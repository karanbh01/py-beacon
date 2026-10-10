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
currency, the [implementation](#implementation-screens-and-redistribution),
the [flows and vehicle](#flows-units-and-the-fee),
[modelling assumptions](#modelling-assumptions), modifiers, a benchmark, the
data source, the cache). Each `run()` takes the subject: an index definition
and a window.

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
2. **Accrues** interest on the cash and the vehicle's management fee since
   the last session, and receives any [dividends](#dividends) due.
3. **Marks** every holding at that day's price, converted into the book's
   currency.
4. **Settles delistings.** A holding past its last listed date (reference
   data's `DATE_TO`) is sold into cash at the last price the portfolio saw,
   with no transaction cost: an acquisition or a failure is not a trade
   crossed in a market. A delisted holding with no usable last price is
   written off. The cash waits for the next rebalance.
5. **Deals flows** at the day's NAV per unit (see
   [Flows, units and the fee](#flows-units-and-the-fee)).
6. **Rebalances**, if the day is a key in the index's weight snapshots. The
   snapshots are keyed by the date weights take effect, so an announcement lag
   needs nothing from the engine. On any other day it invests an inflow and
   works orders an execution limit left.
7. **Records** the day: positions, weights, cash and NAV go into the
   portfolio's books, after paying what fee the cash allows.

A rebalance first removes stale names from the target (see
[Prices](#prices)), then asks each modifier whether to skip, generates the
trades, lets each modifier adjust them, and executes sells before buys.

### Trades and costs

- **Sells**: a held name absent from the target is sold in full; one above
  its target value is trimmed to it.
- **Buys**: a target name below its target value is topped up.

Target values are the portfolio's value before the rebalance times each
target weight. Each trade costs `notional * transaction_cost_bps / 10_000`,
plus market impact when the implementation sets one (see
[Costs and execution](#costs-and-execution)).

Buys are sized so that they and their costs fit the cash the sells leave:
when they would need more, they are scaled down together. A rebalance
therefore fills in full and leaves the book fully invested, short of its
target weights only by the costs.

A buy the cash still cannot cover (a modifier enlarged it, say) is
**partially filled**: the engine buys what the cash affords, cost included,
and records the rest as an `UnfilledOrder` in `result.unfilled` with the
reason `"cash"`. If what the cash affords is worth less than 0.01, nothing is
bought and the order is recorded with a filled quantity of zero.

A target name that cannot be priced on the day (one already delisted, say)
is not bought. It is recorded in `unfilled` with the reason `"no price"`, a
requested quantity of zero and, as its shortfall, the value the rebalance
aimed to hold; its weight stays in cash.

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
| `"reinvest"` (default) | buys the current holdings, in proportion to their value, the day it arrives, as a total-return index assumes |
| `"cash"` | stays in the book as cash and is invested at the next rebalance |
| `"distribute"` | is paid out of the book; returns add it back, so performance is still total return |

`"reinvest"` and `"cash"` both keep the income in the book, as an
accumulating share class does, and differ only in when it is invested.
`"distribute"` pays it out, as a distributing share class does. `"cash"`
avoids a round of small trades, and their costs, on every pay date.

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

### Costs and execution

`MarketImpact(coefficient=1.0, lookback_days=63)` charges every trade for
its size, on top of the fixed basis points, by the square-root law:

```text
impact = coefficient x daily volatility x sqrt(trade value / average daily traded value)
```

as a fraction of the trade's value, with the volatility and the traded value
measured over the `lookback_days` before the trade, in the book's currency.
A name with no history or no volume pays no impact. Without impact a
backtest's return does not depend on its size; with it, the same weights
cost a large fund more, in proportion to the square root of its size.

`ExecutionLimit(participation=None, days=None)` caps how much of an order
trades in one day: at most `participation` of the day's volume, or the
order spread evenly over `days` trading days, or the smaller of the two.
What cannot trade on the rebalance day is a working order, traded on the
following sessions under the same limit, sells before buys and buys no
further than the cash. A new rebalance replaces any order still working,
and a replaced order, or one still working when the run ends, is recorded
in `unfilled` with the reason `"execution limit"`. A volume of 0 means
nothing trades that day. A blank volume is replaced by the last one reported,
if no older than `volume_backfill_days` in the
[modelling assumptions](#modelling-assumptions), and otherwise by the name's
average daily volume over `lookback_days` (63 by default). A name with no
volume at all is not limited by participation, and the run logs a warning.

```python
from beacon.backtest import ExecutionLimit, MarketImpact

realistic = Backtest(
    initial_capital=5_000_000_000.0, data_provider=fetcher,
    transaction_cost_bps=2.0,
    implementation=Implementation(
        impact=MarketImpact(coefficient=0.8),
        execution=ExecutionLimit(participation=0.1, days=5)),
).run(definition, start="2023-01-03", end="2024-12-31")
```

## Flows, units and the fee

A run can take money in and pay it out. Flows are their own part of the run,
the same whatever the vehicle, so the same flows can be run through different
vehicles. Pass one scenario or a list, which are added together:

| Scenario | What flows |
| --- | --- |
| `DatedFlows({"2024-03-01": 5e6, "2024-09-02": -2e6})` | Amounts on dates; one on a day not simulated arrives the next simulated day |
| `PeriodicFlows(amount=1e6)` or `PeriodicFlows(fraction=-0.01)` | A fixed amount, or a share of the assets, each `frequency` (`"MONTHLY"` by default) |
| `RandomFlows(drift=0.05, volatility=0.2, seed=1)` | A normal share of the assets each flow day, with annual drift and volatility, seeded |
| `PerformanceChasingFlows(sensitivity=0.1, lookback_days=63)` | `base + sensitivity x` the trailing return, as a share of the assets |

Amounts are in the book's currency, positive in and negative out. A periodic
scenario flows on the first simulated day of each period after the first day
of the run, which the initial capital already funds.

**Units.** The initial capital buys units at the vehicle's launch price (1.0
without one). Each flow creates or cancels units at that day's NAV per unit,
before anything is traded for it, so a flow changes the fund's size and not
its NAV per unit. With flows, every return metric is a unit's
(time-weighted), and the summary adds `money_weighted_return`, the annual
internal rate of return of the investors' money, which does depend on when
it arrived. Without flows the units never change and the metrics are
computed exactly as before.

**Investing and paying.** An inflow on a rebalance day is invested by the
rebalance. On any other day it is invested at once, toward the last
rebalance's weights (`Implementation(invest_flows="target")`, the default) or
in proportion to the holdings (`"holdings"`); under an execution limit the
buying is worked over the following days. An outflow is paid from cash first,
then by selling every holding pro rata. It is paid the day it is dealt, so
those sales are not held back by an execution limit, and one larger than the
fund is cut to the fund. The trading costs of a flow are borne by the fund.

**Cash buffer.** `Implementation(cash_buffer=0.02)` keeps 2% of the book in
cash: each rebalance invests the rest, an inflow tops the buffer up before
buying, and an outflow draws on it first.

**The vehicle and its fee.** `Vehicle(management_fee_bps=20)` charges 0.20% a
year, accrued every calendar day on the net assets (ACT/365). It is owed as a
liability: the NAV is net of it, a rebalance invests only what is not owed,
and it is paid from cash as soon as there is cash, recorded as a `FEE` cash
flow. `Vehicle(launch_price=100.0)` sets the NAV per unit at launch.

```python
from beacon.backtest import (
    DatedFlows,
    Implementation,
    PeriodicFlows,
    Vehicle,
)

growing = Backtest(
    initial_capital=1_000_000.0, data_provider=fetcher,
    implementation=Implementation(cash_buffer=0.02),
    flows=[PeriodicFlows(amount=50_000.0), DatedFlows({"2024-06-03": -200_000.0})],
    vehicle=Vehicle(management_fee_bps=20, launch_price=100.0),
).run(definition, start="2023-01-03", end="2024-12-31")

print(growing.aum.iloc[-1], growing.nav_per_unit.iloc[-1])
print(len(growing.flows), "flows;",
      round(growing.summary()["money_weighted_return"], 4), "money-weighted")
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
| `result.trading_nav` | `portfolio.nav` without its opening row (initial capital on the eve of the first trading day), net of any fee owed: one row per simulated day. Also `result.aum`. |
| `result.nav_per_unit` | The NAV over the units outstanding: a unit's performance, whatever money flowed |
| `result.units_outstanding` | Units outstanding each simulated day |
| `result.flows` | Each flow as dealt: `date`, `amount`, `nav_per_unit` and the `units` created or cancelled |
| `result.fees_payable` | The management fee owed and not yet paid at the end of each day |
| `result.index.target` | The calculated index being aimed at. Its `source` is the `IndexResult`. |
| `result.index.optimised` | The optimised index's own calculation, on an optimised run; otherwise `None` |
| `result.index.tracked` | The book the engine traded toward: `optimised` if present, else `target` |
| `result.benchmark` | The benchmark of record, when `benchmark=` was given (an `IndexResult` or a level series) |
| `result.unfilled` | Orders not filled in full, each with its `reason`: `"cash"`, `"no price"` or `"execution limit"` |
| `result.price_gaps` | Days a name was marked at a carried price on an open session |
| `result.rebalance_pricing` | The session each rebalance priced from |

```python
print(result.trading_nav.tail(3))
print(result.index.target.levels.tail(3))
print(len(result.portfolio.transactions), "transactions,",
      len(result.unfilled), "unfilled orders,",
      len(result.price_gaps), "price gaps")
```

`summary()` returns total and annualised return, volatility, Sharpe ratio
(zero risk-free rate), maximum drawdown and, when the run tracked an index,
`tracking_error` and `tracking_difference`; with flows, every figure is a
unit's and `money_weighted_return` is added. Returns are daily and annualised
over 252 periods. `get_tracking_error()` is the annualised standard deviation
of daily NAV returns minus `index.tracked` returns; `get_tracking_difference()`
is the cumulative NAV return minus the cumulative index return.

Every metric starts from the initial capital. `get_returns()` has one return
per simulated day, and the first runs from the capital to the first day's
close, so the cost of the opening trades is in the tracking metrics,
volatility and maximum drawdown, as it is in `total_return`. On that first day
the index's return is zero, since it starts at its base level.

`result.against(other)` compares the NAV (per unit, with flows) with any
other result, book,
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
- **Fees, flows and cash.** A vehicle's management fee lowers the NAV, a cash buffer holds part of the book out of the market, and trading for flows costs the fund; the index has none of these. Compare a unit's NAV, not the NAV, when money flowed.
- **Execution.** Costs leave the book a little under its target weights,
  an execution limit trades into them over several days, and a name with no
  price leaves its weight in cash.
- **Distributions.** A `TOTAL_RETURN` or `NET_TOTAL_RETURN` index reinvests
  a dividend on its ex-date. The book is paid it on the pay date and, by
  default, reinvests it that day, so the two agree exactly when the pay date
  is the ex-date. With `dividends="cash"` the book holds the cash until the
  next rebalance (see [Dividends](#dividends)). Set `withholding_tax_rate` to
  the index's rate to track a net-total-return index.
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
a book in USD.

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
| `volume_backfill_days` | 5 | How many calendar days a name's last reported volume stands in for a blank one under an execution limit |

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
  it runs are uncached. Pass `IndexResultCache(path)` to put it elsewhere,
  or `cache=False` to turn caching off. In-memory data is never cached
  either way.
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
