---
title: Backtest
description: "Simulating a portfolio that trades to an index, and reading what it did."
---

# Backtest

A backtest simulates a real portfolio that trades to an index's target
weights. It answers what an index's level cannot: what the money did once
trades cost something, cash ran short, data had holes and names were delisted.

It lives in `beacon.backtest`. `Backtest` is the front door; `BacktestEngine`
is the simulation underneath it, and `BacktestResult` is what both return.

## Running a backtest

`Backtest` holds what stays fixed across runs (capital, costs, the book's
currency, [modelling assumptions](#modelling-assumptions), modifiers, a
benchmark, the data source, the cache).
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

**Stale names.** When `max_price_staleness_days` is set (on the data source
or in the [modelling assumptions](#modelling-assumptions)), a rebalance
removes every target name that has not traded within that many calendar days.
It is the first screen at every rebalance (see
[Implementation](#implementation-screens-and-redistribution)), and the
removed name is then sold through the ordinary path at its last price. With
no threshold (the default) nothing is removed.

## Dividends

A holding is entitled to a cash distribution if it is held at the start of
the ex-date, before that day's trades. The cash arrives on the pay date (the
ex-date when the data gives none), converted into the book's currency that
day and net of `withholding_tax_rate` from the
[modelling assumptions](#modelling-assumptions). A holding sold before the
pay date is still paid. Cancelled distributions are ignored, and the price
data must already show the ex-date drop, as an unadjusted feed does.

`Backtest(dividends=...)` decides what happens to the cash:

| Policy | The cash |
| --- | --- |
| `"accumulate"` (default) | stays in the book and is invested at the next rebalance |
| `"reinvest"` | buys the current holdings, in proportion to their value, the day it arrives |
| `"distribute"` | is paid out of the book; returns add it back, so performance is still total return |

Each payment is a `CashFlow` in `result.portfolio.cash_flows`: a `DIVIDEND`
per name paid, and a negative `DISTRIBUTION` when the cash is paid out. With
`"distribute"`, `trading_nav` is the NAV after the payout, while
`get_returns()` and `summary()` add the payout back.

## Implementation: screens and redistribution

An `Implementation` says how the index is carried out at your size, without
changing the index. Each rebalance runs as stages:

1. **Target**: the index's weights, as published.
2. **Screens**: stale names are removed first, then each screen in order
   removes the names it rejects. A removed name is sold if held and never
   bought.
3. **Redistribution**: the removed weight is spread across the remaining
   names in proportion to their weights (`redistribution="pro_rata"`, the
   default), or left in cash (`redistribution="cash"`).
4. **Capacity**: each name is cut to its [caps](#capacity) at the book's
   size, and positions too small to keep are dropped.
5. **Trades**, generated from the result, then any [modifiers](#modifiers).

| Screen | Removes a name when |
| --- | --- |
| `MarketCapScreen(min_cap, exit=None, float_adjusted=False)` | its market cap, in the book's currency, is below the level |
| `LiquidityScreen(min_traded_value=..., exit=None, lookback_days=63)` | its average daily close times volume, in the book's currency, is below the level |
| `LiquidityScreen(min_volume=..., exit=None, lookback_days=63)` | its average daily share volume is below the level |
| `MinimumPriceScreen(min_price, exit=None)` | its last close, in the book's currency, is below the level |
| `ListingAgeScreen(min_days)` | fewer than `min_days` calendar days have passed since its first price in the data |
| `ExclusionScreen(identifiers=..., sectors=..., regions=...)` | it is listed, or its `SECTOR` or `REGION` on the date is |
| `ExpressionScreen(expression, on_missing=False)` | it fails the [expression](expressions.md); money fields are in the book's currency |

**Buffers.** The threshold screens take an optional `exit` level below the
entry level. A name not held must reach the entry level to come in; a held
name stays until it falls below `exit`, so a name near the threshold does not
flip in and out at every rebalance. Without `exit`, both levels are the
entry level.

A threshold screen rejects a name it cannot value (no price or no share
count) unless built with `on_missing=True`. An `ExpressionScreen` naming a
field the data does not have refuses the run before it starts.

```python
from beacon.backtest import (
    ExpressionScreen,
    Implementation,
    MarketCapScreen,
)
from beacon.expressions import data

screened = Backtest(
    initial_capital=1_000_000.0, data_provider=fetcher,
    implementation=Implementation(screens=[
        ExpressionScreen(data.reference.sector != "Utilities"),
        MarketCapScreen(min_cap=5e10, exit=4e10),
    ]),
).run(definition, start="2023-01-03", end="2024-12-31")

first = screened.rebalance_steps[0]
print(first.removed)                         # {'CCC': 'ExpressionScreen'}
print({name: round(weight, 3) for name, weight in first.weights.items()})
```

`result.rebalance_steps` records each rebalance's stages as a
`RebalanceStep`: the `target`, the names `removed` and the screen that
removed each (`"stale price"` for a stale one, `"MinimumPosition"` for one
too small to keep), the names `capped` and the weight each was cut to, the
`weights` traded to, and the `cash_weight` they leave. Screens and caps are
re-evaluated at every rebalance from data dated on or before it.

### Capacity

A cap limits a position without excluding the name, measured against the
book's value at each rebalance, so a position shrinks smoothly as the fund
grows rather than vanishing at a threshold.

| Cap | A position's value is at most |
| --- | --- |
| `OwnershipCap(max_share)` | `max_share` of the name's free-float market cap |
| `LiquidityCap(days, participation=0.2, lookback_days=63)` | `days` times its average daily traded value times `participation` |
| `WeightCap(max_weight)` | `max_weight` of the book |

Each name is held at no more than its tightest cap. The excess of a capped
name is spread across the names still under their caps, in proportion to
their weights, until none is over; under `redistribution="cash"` it stays in
cash, as it does when every name is capped. A name a cap cannot value (no
free float, no volume) is not limited by it.

`MinimumPosition(value=None, weight=None)` drops a position below either
level and redistributes its weight, and the caps are applied again
afterwards.

```python
from beacon.backtest import LiquidityCap, MinimumPosition, OwnershipCap

large = Backtest(
    initial_capital=50_000_000_000.0, data_provider=fetcher,
    implementation=Implementation(
        caps=[OwnershipCap(0.05), LiquidityCap(days=5)],
        minimum_position=MinimumPosition(weight=0.01)),
).run(definition, start="2023-01-03", end="2024-12-31")

print(large.rebalance_steps[0].capped)
```

## Modifiers

A `BacktestModifier` changes rebalance behaviour after the trades are
generated. It implements two methods:

- `should_skip_rebalance(date, portfolio, target_weights)`: return `True` to
  skip the rebalance entirely.
- `adjust_trades(trades, date, portfolio)`: return a changed list of
  `TradeInstruction`s.

Modifiers run in the order given. **`DriftThresholdModifier(threshold)`**
ships: it skips a rebalance when every name's weight is within `threshold` of
its target (0.02 means 2 percentage points), saving turnover. Deciding which
names may be held is a screen's job; a screen passed as a modifier is refused
with a message saying where it goes.

```python
from beacon.backtest import DriftThresholdModifier

drifting = Backtest(
    initial_capital=1_000_000.0, data_provider=fetcher,
    modifiers=[DriftThresholdModifier(0.02)],
).run(definition, start="2023-01-03", end="2024-12-31")

print("Rebalances traded:", len(drifting.rebalance_pricing),
      "of", len(result.rebalance_pricing))
```

The drift run trades only its first rebalance. Share counts never change in
the sample data, so the market-cap weights move with prices exactly as the
portfolio's do and there is never more than 2 points to correct.

## Reading the result

`BacktestResult` keeps the books whole rather than flattening them into
fields. Each comparator is a `Book` with the same surface: `levels`,
`weights` (dates by identifiers) and `returns`.

| Attribute | What it holds |
|---|---|
| `result.portfolio` | The simulated `Portfolio`, frozen: `nav`, `cash`, `positions`, `weights`, `transactions`, `initial_capital` |
| `result.trading_nav` | `portfolio.nav` without its opening row (initial capital on the eve of the first trading day): one row per simulated day. |
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

Every metric starts from the initial capital. `get_returns()` has one return
per simulated day, and the first runs from the capital to the first day's
close, so the cost of the opening trades is in the tracking metrics,
volatility and maximum drawdown, as it is in `total_return`. On that first day
the index's return is zero, since it starts at its base level.

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
  a dividend on its ex-date. The book is paid it on the pay date and, by
  default, invests it at the next rebalance, so it holds the cash in between
  (see [Dividends](#dividends)). With `dividends="reinvest"` and the pay date
  on the ex-date, the two agree exactly. Set `withholding_tax_rate` to the
  index's rate to track a net-total-return index.
- **Price gaps.** On an open session with no bar, both carry the name's last
  close and record the day in their own `price_gaps`, so a gap does not
  separate them.
- **Delistings between rebalances.** The engine settles into cash and holds
  it until the next rebalance.
- **Currency.** A book kept in a different currency from the index sees
  exchange-rate moves the index does not (see below).
- **Modifiers.** A skipped rebalance or a screened-out name leaves the book
  away from the index's weights.

## The book's currency

The simulated book is kept in the index's currency: a GBP index's backtest
values its holdings, cash, costs and NAV in pounds, so its NAV follows the
level. `result.currency` says which currency a run's book is in.

Pass `Backtest(currency=...)` to keep the book in another currency, as a
dollar investor tracking a sterling index would. The NAV then picks up the
GBP/USD moves the index does not see, which shows in the tracking figures:

```python
sterling = IndexDefinition(
    index_id="SAMPLE-GBP", index_name="Sample Index in Sterling",
    base_date="2023-01-03", base_value=1000.0, currency="GBP",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "FFF"],
)

in_gbp = Backtest(initial_capital=1000.0,
                  data_provider=fetcher).run(sterling, end="2024-12-31")
in_usd = Backtest(initial_capital=1000.0, currency="USD",
                  data_provider=fetcher).run(sterling, end="2024-12-31")
print(in_gbp.currency, in_usd.currency)   # GBP USD
print(in_gbp.get_tracking_error() < in_usd.get_tracking_error())   # True
```

An index that does not record its currency, such as one built by hand, gets
a book in USD. An `IndexFund` follows the same rule.

## Modelling assumptions

`ModellingAssumptions` gathers what a run takes as given about markets and
data, so a result can say what it assumed. Every field is optional:

| Field | Unset means | What it decides |
| --- | --- | --- |
| `fx_policy` | The data source's (`"CARRY_FORWARD"` unless set) | The FX rate on a day a pair printed none |
| `max_price_staleness_days` | The data source's (no limit unless set) | How long a name may go without trading before it is dropped; 0 is no limit |
| `free_float_backfill_days` | The data source's (90 unless set) | How far a free float carries over blank cells |
| `cash_rate` | 0 | The annual rate cash earns, accrued daily, ACT/365 |
| `risk_free_rate` | 0 | The rate the Sharpe ratio is measured against |
| `periods_per_year` | 252 | How returns, volatility and tracking error are annualised |
| `withholding_tax_rate` | 0 | The share of each dividend withheld before the book receives it |

The first three are **data treatment**: they decide how data is read, so they
reach the index calculation too. A backtest hands its assumptions to the
calculator, and the index and the simulation read data the same way. The
others are **simulation conventions**, which do not affect an index.

Set a default for the whole process once, and override any field for one
backtest. A backtest's own fields win field by field, so the example below
keeps the process-wide FX policy:

```python
from beacon import ModellingAssumptions, use_modelling_assumptions

use_modelling_assumptions(ModellingAssumptions(fx_policy="EXACT_DAY"))

earning = Backtest(initial_capital=1_000_000.0, data_provider=fetcher,
                   modelling_assumptions=ModellingAssumptions(cash_rate=0.02)
                   ).run(definition, end="2024-12-31")
print(earning.modelling_assumptions.fx_policy,
      earning.modelling_assumptions.cash_rate)   # EXACT_DAY 0.02

use_modelling_assumptions(None)   # back to nothing set
```

`result.modelling_assumptions` records every field resolved: unset data
treatment filled from the data source, unset conventions from the defaults.
Interest on cash appears in `result.portfolio.cash_flows`, one `CashFlow` per
day, beside the trades in `transactions`.

An `IndexResult` records the data treatment it was calculated under in its own
`modelling_assumptions`. When `BacktestEngine` is given an index calculated
under different data treatment from its own, it runs and logs a warning
naming each setting that differs. The result cache keys on the data
treatment, so a calculation under one FX policy is never reused for another.

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
