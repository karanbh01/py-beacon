# Concepts overview

py-beacon keeps three questions apart: what an index is (its methodology),
how its history is computed (the calculation), and how a portfolio that
tracks it behaves (the backtest). All three read their data through one
`DataFetcher`.

```
Data          DataFetcher: prices, reference data, corporate actions, FX, features
     |
     v
Methodology   IndexDefinition: universe, selection, weighting, capping, treatment, schedule
     |
     v
Calculation   IndexCalculator.run() -> IndexResult
     |
     v
Backtest      Backtest.run() -> BacktestResult
```

Each layer depends only on the ones above it in this picture, so a
methodology can be calculated without being traded, and a calculation can be
reused by many backtests.

## Data

Every calculation asks a `DataFetcher` for what it needs, as of a date:
prices, shares outstanding, free float, reference fields, corporate actions,
exchange rates and features. Three settings on the fetcher decide what
happens when data is missing: whether a missing exchange rate carries the
last one forward, how long a name may go without trading before it is
treated as stale, and how long a reported free float carries forward over
blanks. [Data](data.md) covers the data model and these settings.

Rules and universes name the data they read with
[expressions](expressions.md), such as `data.market.market_cap > 1e9`, so a
screen written in Python and one built in the Beacon app are the same thing.

## Methodology

An `IndexDefinition` holds the static rules of an index. The
[universe](universe.md) is the set of identifiers it may draw from, either a
fixed list or a filter. The [methodology](methodology.md) then has four
concerns:

- **Selection.** Eligibility rules narrow the universe to the constituents
  on each rebalance date. A name must pass every rule.
- **Weighting.** A weighting scheme assigns the constituents' weights:
  `EqualWeighted`, or `MarketCapWeighted`, optionally by free-float market
  cap. An optional `max_constituent_weight` then caps any one name,
  redistributing the excess and repeating until nothing breaches the cap.
- **Treatment.** The divisor keeps the level continuous when a rebalance,
  a delisting or a special dividend changes the index's market value
  without changing what it is worth. The return type decides whether cash
  distributions are reinvested: `PRICE` ignores them, `TOTAL_RETURN`
  reinvests them, and `NET_TOTAL_RETURN` reinvests them after withholding
  tax.
- **Scheduling.** The rebalance frequency (monthly, quarterly, semi-annual
  or annual), the day rule (first or last business day of the month, or the
  third Friday), and the trading calendar, an exchange MIC such as `XNYS`.
  Every index needs a calendar and there is no default, because a wrong one
  would quietly schedule rebalances on another market's holidays. An
  optional lag separates the day a composition is announced from the
  session it takes effect.

## Calculation

`IndexCalculator.run()` walks the sessions of the index's calendar, from the
base date (or a later start date) to the end date. On each rebalance date it
applies the methodology; on every session it values the index and records
what it held. The `IndexResult` holds the index levels, the divisor
history, the constituents and weights decided at each rebalance, the weights
held on every day, and a report for each rebalance where a weight cap bound.

## Backtest

`Backtest` calculates the index (reusing a cached calculation when it can)
and simulates a portfolio trading towards its weights, with transaction
costs. The `BacktestResult` keeps the portfolio's books on
`result.portfolio` and the index on `result.index`, so the two can be
compared. The portfolio's NAV generally differs from the index level, even
without costs; [Backtest](backtest.md) explains why.

## Built on top

- [Funds](funds.md): `IndexFund` and `ETF` run a backtest of their index and
  deduct a management fee.
- [Derivatives](derivatives.md): futures and total return swaps priced off
  an index, an ETF or a stock.
- [Optimiser](optimiser.md): weights found numerically under constraints,
  rather than by a formula, including optimised indices derived from
  another index.
- [Risk model](risk-model.md): how assets move together, with covariance
  shrinkage and factor models.
- [Attribution](attribution.md): where the return and the risk came from,
  with parts that add up exactly to the whole.
- [Charts](charts.md) and [Reports](reports.md): drawing results and
  laying them out as PDF documents.
