# Backtest

A backtest simulates a real portfolio that trades to an index's target
weights. It answers what an index's level cannot: what the money did once
trades cost something, cash ran short, data had holes and names were delisted.

It lives in `beacon.backtest`. `Backtest` is the front door; `BacktestEngine`
is the simulation underneath it, and `BacktestResult` is what both return.

## Running a backtest

`Backtest` holds the assumptions that stay fixed across runs (capital, costs,
the book's currency, modifiers, a benchmark, the data source, the cache).
Each `run()` takes the subject: an index definition and a window.

```python
import logging

from beacon.backtest import Backtest
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

fetcher = dataset.data_fetcher()  # frozen sample data, held in memory

definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)

backtest = Backtest(initial_capital=1_000_000.0,
                    transaction_cost_bps=5.0,
                    data_provider=fetcher)
result = backtest.run(definition, start="2023-01-03", end="2024-12-31")

print(result.summary())
```

A run does two things in order:

1. **Calculate the index**, or reuse a cached calculation of it (see
   [The result cache](#the-result-cache)). This is the `IndexCalculator`
   described in [Methodology](methodology.md), and it produces the
   `IndexResult` whose rebalance snapshots are the target weights.
2. **Simulate** a portfolio trading to those weights with `BacktestEngine`.

`end` is required; `start` defaults to the definition's base date. With no
`data_provider`, each run resolves the process's data source at run time, so
`beacon.use()` after construction is honoured. A calculation that comes back
empty (no levels, all levels zero, or no constituent ever held) raises
`CalculationError` instead of producing a backtest of nothing; the usual cause
is a missing or unresolvable `universe_identifiers`.

`run(definition, ..., optimised=True, optimisation_config=...)` trades an
optimised version of the index instead, built and calculated the same way a
stored optimised index is. See [Optimiser](optimiser.md).

## What happens each day

The engine walks the index calendar's sessions from `start` to `end`. Each
day it:

1. **Applies splits.** A split, reverse split or stock dividend with its
   ex-date since the last session multiplies the shares held by its ratio,
   and the per-share cost and price divide by it, so the holding's value is
   unchanged. It is not a trade and costs nothing. A cancelled action is
   ignored.
2. **Marks** every holding at that day's price, converted into the book's
   currency.
3. **Settles delistings.** A holding past its last listed date (reference
   data's `DATE_TO`) is sold into cash at the last price the portfolio saw,
   with no transaction cost: an acquisition or a failure is not a trade
   crossed in a market. A delisted holding with no usable last price is
   written off. The cash waits for the next rebalance.
4. **Rebalances**, if the day is a key in the index's weight snapshots. The
   snapshots are keyed by the date weights take effect, so an announcement lag
   needs nothing from the engine.
5. **Records** the day: positions, weights, cash and NAV go into the
   portfolio's books.

A rebalance first removes stale names from the target (see
[Prices](#prices)), then asks each modifier whether to skip, generates the
trades, lets each modifier adjust them, and executes sells before buys.

### Trades and costs

- **Sells**: a held name absent from the target is sold in full; one above
  its target value is trimmed to it.
- **Buys**: a target name below its target value is topped up.

Target values are the portfolio's value before the rebalance times each
target weight. Each trade costs `notional * transaction_cost_bps / 10_000`.

A buy the cash cannot cover is **partially filled**: the engine buys what the
cash affords, cost included, and records the rest as an `UnfilledOrder` in
`result.unfilled`. If what the cash affords is worth less than 0.01, nothing
is bought and the order is recorded with a filled quantity of zero. Because
buys are sized before costs are paid, the last buy of a rebalance with
non-zero costs usually comes up short by about that rebalance's costs, so
`unfilled` is routinely non-empty when costs are on; the shortfall stays in
cash.

A target name that cannot be priced at all (one already delisted, say) is
not bought, and is not recorded in `unfilled`.

### Which days are traded

The run steps onto the sessions of the definition's `calendar`. A rebalance
date in the schedule that is not a session (a holiday in a schedule built on
another venue's calendar, or written by hand) is still traded, on the date it
names, at the prices of the last session on or before it. The date also gets
its own row in the NAV series. `result.rebalance_pricing` holds one
`RebalancePricing(date, priced_from)` per rebalance that went ahead; the two
dates differ only when the market was shut. A skipped rebalance has no row.

An index calculated by py-beacon never produces such a date, because its
schedule and the engine read the same calendar.

## Prices

The engine reads the `CLOSE` column unless `price_column` says otherwise, and
refuses before the first trade if the data has no such column.

**Currency.** Prices are stored in each name's listing currency (reference
data's `CURRENCY`). Each is converted into the book's currency with the FX
rate on or before the day. A missing rate raises `CalculationError` rather
than valuing the holding unconverted.

**Missing bars.** A date inside the data's coverage with no bar for a name
resolves to the name's last bar on or before it:

- If the calendar says the market was **closed**, nothing is missing and
  nothing is recorded.
- If the calendar says it was **open**, the data has a hole. The last price
  is carried forward and recorded as a `PriceGap(date, asset_id, priced_from)`
  in `result.price_gaps`, one per name per day.

A date outside the data's coverage raises `CalculationError`. A delisted name
is never carried forward past its last listed date.

**Stale names.** When the fetcher is opened with `max_price_staleness_days`,
a rebalance drops every target name that has not traded within that many
calendar days, and renormalises the remaining weights so the book stays fully
invested. The dropped name is then sold through the ordinary path at its last
price. With no threshold (the default) nothing is dropped.

## Modifiers

A `BacktestModifier` changes rebalance behaviour without changing the index.
It implements two methods:

- `should_skip_rebalance(date, portfolio, target_weights)`: return `True` to
  skip the rebalance entirely.
- `adjust_trades(trades, date, portfolio)`: return a changed list of
  `TradeInstruction`s.

Modifiers run in the order given. Two ship:

- **`DriftThresholdModifier(threshold)`** skips a rebalance when every name's
  weight is within `threshold` of its target (0.02 means 2 percentage
  points), saving turnover.
- **`ExpressionScreen(expression, fetcher, on_missing=False)`**, in
  `beacon.backtest.rules`, drops names that fail an
  [expression](expressions.md), evaluated point in time at every rebalance.
  Buys of an excluded name are cancelled and a holding of it is sold at the
  price it was last marked at, with no cost. The freed weight is **not**
  redistributed: it stays in cash. A name with no value for the expression
  fails unless `on_missing=True`.

```python
from beacon.backtest import DriftThresholdModifier
from beacon.backtest.rules import ExpressionScreen
from beacon.expressions import data

drifting = Backtest(
    initial_capital=1_000_000.0, data_provider=fetcher,
    modifiers=[DriftThresholdModifier(0.02)],
).run(definition, start="2023-01-03", end="2024-12-31")

screened = Backtest(
    initial_capital=1_000_000.0, data_provider=fetcher,
    modifiers=[ExpressionScreen(data.reference.sector != "Utilities", fetcher)],
).run(definition, start="2023-01-03", end="2024-12-31")

print("Rebalances traded:", len(drifting.rebalance_pricing),
      "of", len(result.rebalance_pricing))
print("Weights without CCC:", screened.portfolio.get_weights())
print("Cash:", round(screened.portfolio.cash.iloc[-1], 2))
```

The drift run trades only its first rebalance. Share counts never change in
the sample data, so the market-cap weights move with prices exactly as the
portfolio's do and there is never more than 2 points to correct. The screened run never holds
CCC, the one utility, and keeps CCC's weight in cash. Combining the two needs
care: an excluded name stays away from its target, so the drift threshold
never skips.

## Reading the result

`BacktestResult` keeps the books whole rather than flattening them into
fields. Each comparator is a `Book` with the same surface: `levels`,
`weights` (dates by identifiers) and `returns`.

| Attribute | What it holds |
|---|---|
| `result.portfolio` | The simulated `Portfolio`, frozen: `nav`, `cash`, `positions`, `weights`, `transactions`, `initial_capital` |
| `result.trading_nav` | `portfolio.nav` without its opening row (initial capital on the eve of the first trading day). Every metric uses this. |
| `result.index.target` | The calculated index being aimed at. Its `source` is the `IndexResult`. |
| `result.index.optimised` | The optimised index's own calculation, on an optimised run; otherwise `None` |
| `result.index.tracked` | The book the engine traded toward: `optimised` if present, else `target` |
| `result.benchmark` | The benchmark of record, when `benchmark=` was given (an `IndexResult` or a level series) |
| `result.unfilled` | Buys not filled in full |
| `result.price_gaps` | Days a name was marked at a carried price on an open session |
| `result.rebalance_pricing` | The session each rebalance priced from |

```python
print(result.trading_nav.tail(3))
print(result.index.target.levels.tail(3))
print(len(result.portfolio.transactions), "transactions,",
      len(result.unfilled), "partial fills,",
      len(result.price_gaps), "price gaps")
```

`summary()` returns total and annualised return, volatility, Sharpe ratio
(zero risk-free rate), maximum drawdown and, when the run tracked an index,
`tracking_error` and `tracking_difference`. Returns are daily and annualised
over 252 periods. `get_tracking_error()` is the annualised standard deviation
of daily NAV returns minus `index.tracked` returns; `get_tracking_difference()`
is the cumulative NAV return minus the cumulative index return.

The tracking metrics, volatility and maximum drawdown are computed from
`trading_nav`, which starts at the end of the first day, after the opening
purchase. The cost of that purchase is therefore not in them, and with costs
on, the tracking difference can come out slightly positive. `total_return`
and `annualised_return` are measured from the initial capital and do include
it.

`result.against(other)` compares the NAV with any other result, book,
`IndexResult` or level series and returns excess return, tracking error, beta
and correlation. It stores nothing, so the benchmark of record stays what the
run was given. `result.asset("AAA")` gives one name's trades, holding periods
and weight against target. For charts, see `result.plot`.

## Index level and NAV

At zero cost, for a price-return index with the book in the index's
currency, the NAV follows the index exactly: it is the index level rescaled
to the initial capital, and the two agree to about 15 significant digits.
There is no tracking error to explain.

```python
exact = Backtest(initial_capital=1000.0, data_provider=fetcher).run(
    definition, start="2023-01-03", end="2024-12-31")

nav = exact.trading_nav
level = exact.index.target.levels.reindex(nav.index)
print("Largest daily gap:", float((nav / level - 1).abs().max()))
```

Differences come from what the index does not model or models differently:

- **Costs.** The index trades for free.
- **Partial fills.** With costs on, the last buy of a rebalance is usually
  short and the difference sits in cash.
- **Distributions.** A `TOTAL_RETURN` or `NET_TOTAL_RETURN` index reinvests
  dividends; the engine receives no distributions and earns only the price
  return. It falls behind by about the dividend yield.
- **Price gaps.** On an open session with no bar the engine carries the name's
  last price, while the index values that name at zero for the day. The two
  differ by about the name's weight on that day and agree again once the bar
  returns.
- **Delistings between rebalances.** The engine settles into cash and holds
  it until the next rebalance.
- **Currency.** A book kept in a different currency from the index sees
  exchange-rate moves the index does not (see below).
- **Modifiers.** A skipped rebalance or a screened-out name leaves the book
  away from the index's weights.

## The book's currency

The simulated book is kept in `Backtest(currency=...)`, which defaults to
`"USD"`. It does **not** follow the index definition's `currency`: a GBP index
run with the default is valued in dollars, so its NAV picks up the GBP/USD
moves and no longer matches the level. Pass the index's currency to keep the
two aligned:

```python
sterling = IndexDefinition(
    index_id="SAMPLE-GBP", index_name="Sample Index in Sterling",
    base_date="2023-01-03", base_value=1000.0, currency="GBP",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "FFF"],
)

in_gbp = Backtest(initial_capital=1000.0, currency="GBP",
                  data_provider=fetcher).run(sterling, end="2024-12-31")
print("Tracking error in GBP:", in_gbp.get_tracking_error())
```

An `IndexFund` has no currency setting and always keeps its book in USD (see
[Funds and ETFs](funds.md)).

## The result cache

Calculating the index is usually most of a run's time, and the calculation
depends only on the definition, the data, the window and the library version.
`Backtest` therefore keeps calculated `IndexResult`s in an `IndexResultCache`
(`beacon.index.cache`) and reuses one when all four match. A sweep over costs
or capital calculates the index once.

- **Keyed completely or not at all.** The key covers every field of the
  definition, its rules and weighting scheme, the store behind the fetcher
  (path plus a stamp of its manifest), the window and the version. Anything
  that cannot be keyed is never cached: in-memory data, a rule or scheme not
  registered in the catalogue, or a parameter that does not serialise to JSON.
  `beacon.index.cache.explain_uncacheable(...)` says why.
- **No invalidation.** Changed inputs produce a different key. Rewriting a
  store, even with identical content, changes its stamp and misses.
- **Location.** `cache=None` (the default) uses `index_cache` in the
  platform's app-data folder, which needs the `platformdirs` package; without
  it runs are uncached. Pass `IndexResultCache(path)` to put it elsewhere.
  There is no switch that turns caching off; in-memory data is simply never
  cached.
- **Housekeeping.** Entries are written atomically, a corrupt entry is a miss
  and is removed, and the cache is pruned to 512 MB, least recently used
  first. `clear()` empties it.

Only data read from a store on disk is cacheable. This example saves the
sample data as a store in the current folder and keeps the cache beside it:

```python
from pathlib import Path

from beacon.data import store
from beacon.index.cache import IndexResultCache

store.save(fetcher, Path("sample-store"))
stored = store.load(Path("sample-store"))
cache = IndexResultCache(Path("index-cache"))

for cost in (0.0, 5.0, 10.0):  # one calculation, three simulations
    run = Backtest(initial_capital=1_000_000.0, transaction_cost_bps=cost,
                   data_provider=stored, cache=cache).run(
        definition, start="2023-01-03", end="2024-12-31")
    print(cost, "bps:", round(run.summary()["tracking_difference"], 6))

print("Cached calculations:", len(list(cache.root.iterdir())))
```

## Using the engine directly

`BacktestEngine` takes the same assumptions plus an `IndexResult` to trade
toward, with no cache and no calculation. Pass `calendar=`: without one the
engine steps onto Monday to Friday and judges a missing bar against the
data's own sessions. `beacon.testing.weights.index_result_from_weights` turns
a plain `{date: {identifier: weight}}` schedule into an `IndexResult` the
engine accepts. Here the second rebalance falls on 4 July, when the NYSE is
shut:

```python
from beacon.backtest import BacktestEngine
from beacon.testing.weights import index_result_from_weights

schedule = index_result_from_weights({
    "2023-01-03": {"AAA": 0.5, "CCC": 0.5},
    "2023-07-04": {"AAA": 0.25, "BBB": 0.25, "CCC": 0.5},
})

engine_result = BacktestEngine(
    start_date="2023-01-03", end_date="2023-09-29",
    initial_capital=1_000_000.0, data_provider=fetcher,
    index_result=schedule, calendar="XNYS",
).run()

for row in engine_result.rebalance_pricing:
    print(row.date.date(), "priced from", row.priced_from.date())
```

`index_result_from_weights` gives the result a flat stand-in level, so
tracking metrics against it mean nothing; use it for schedules, not for
comparisons.

See the [API reference](../reference/backtest.md) for every parameter.
