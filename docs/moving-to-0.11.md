---
title: Moving to 0.11.0
description: "Every 0.10.0 path on a backtest result and its 0.11.0 replacement, the renamed names, and what changed without a warning."
---

# Moving to 0.11.0

0.11.0 gives a backtest result four parts:

- `result.fund`: the vehicle's recorded data;
- `result.trading`: what the engine did;
- `result.assumptions`: what the run assumed;
- `result.analytics`: every calculation on the run.

The 0.10.0 names still work and give what they gave, but each warns with a `DeprecationWarning` that names its replacement. They are removed in 0.12.0. To find every old name your code still uses, run it with the warnings raised as errors:

```bash
python -W error::DeprecationWarning your_script.py
```

`result.summary()`, `asset()`, `against()`, `with_data()` and `plot` are unchanged, and so are `result.portfolio`, `result.index` and `result.benchmark`.

## The fund's data

`result.fund` is a `FundRecord`. A part the vehicle doesn't have is None: `fund.etf` for a fund that is not listed, and likewise `fund.swap`, `fund.futures`, `fund.cash_band` and `fund.hedging`.

| 0.10.0 | 0.11.0 |
|---|---|
| `result.nav_per_unit` | `result.fund.nav_per_unit` |
| `result.trading_nav` | `result.fund.trading_nav` |
| `result.aum` | `result.fund.aum` |
| `result.units` | `result.fund.units` |
| `result.units_outstanding` | `result.fund.units_outstanding` |
| `result.launch_price` | `result.fund.launch_price` |
| `result.currency` | `result.fund.currency` |
| `result.flows` | `result.fund.flows` |
| `result.share_classes` | `result.fund.share_classes` |
| `result.fees_payable` | `result.fund.fees.payable` |
| `result.fees_paid` | `result.fund.fees.paid` |
| `result.dividends_receivable` | `result.fund.dividends.receivable` |
| `result.dividends_entitled` | `result.fund.dividends.entitled` |
| `result.market` | `result.fund.etf.market` |
| `result.swap_value` | `result.fund.swap.value` |
| `result.swap_resets` | `result.fund.swap.resets` |
| `result.counterparty_exposure` | `result.fund.swap.counterparty_exposure` |
| `result.futures_notional` | `result.fund.futures.notional` |
| `result.futures_exposure` | `result.fund.futures.exposure` |
| `result.futures_trades` | `result.fund.futures.trades` |
| `result.futures_rolls` | `result.fund.futures.rolls` |
| `result.futures_days` | `result.fund.futures.days` |
| `result.cash_band_events` | `result.fund.cash_band.events` |
| `result.hedge_value` | `result.fund.hedging.value` |
| `result.performance_levels()` | `result.fund.nav_per_unit`, or `result.fund.trading_nav` for a run without flows |

## The engine's data

`result.trading` is a `TradingRecord`.

| 0.10.0 | 0.11.0 |
|---|---|
| `result.rebalance_steps` | `result.trading.steps` |
| `result.rebalance_pricing` | `result.trading.pricing` |
| `result.unfilled` | `result.trading.unfilled` |
| `result.total_unfilled_value` | `result.trading.unfilled_value` |
| `result.price_gaps` | `result.trading.price_gaps` |
| `result.total_costs` | `result.trading.total_costs` |
| `result.replication` | `result.trading.replication` |
| `result.active` | `result.trading.active` |

## The analytics

`result.analytics` was the fund's `PerformanceAnalytics`. It is now the run's `Analytics`, which has:

- `analytics.portfolio`: a whole portfolio's figures in nine categories, `performance`, `risk`, `relative`, `holdings`, `trading`, `flows`, `income`, `vehicle` and `attribution`;
- `analytics.instrument`: the figures a name at a time;
- `analytics.reports`: reports, each a set of statistics worked out together.

Each analytic is a method named by its statistic. Called bare, it gives the fund over the whole run. Keywords pick the form:

| Keyword | What it picks |
|---|---|
| `of=` | the portfolio (`"fund"`, `"index"`, `"benchmark"` or `"all"`) |
| `start=`, `end=` | the window |
| `by=` | calendar periods (`"month"`, `"quarter"` or `"year"`) |
| `trailing=` | windows ending on the last day, such as `["1Y", "3Y"]` |
| `rolling=`, `step=` | a rolling window |
| `against=` | the comparator |
| `as_of=` | one day of an analytic that runs through time |
| `annualise=` | whether an annualised figure is given over any window |

A single figure is a float. Each keyword that adds a dimension makes it a Series, then a DataFrame.

`result.risk`, `result.attribution` and `result.portfolio_analytics` were the classes `RiskAnalytics`, `Attribution` and `PortfolioAnalytics`. Their new homes give the same figures as statistics, a method each. The classes are still there, built from the run's data: `Attribution(result.analytics.run_data)`, `PortfolioAnalytics(result.analytics.run_data)` and `RiskAnalytics(result.analytics.run_data, fetcher)`, where `fetcher` is the data the run was made from.

| 0.10.0 | 0.11.0 |
|---|---|
| `result.analytics` (`PerformanceAnalytics`) | `PerformanceAnalytics(result.analytics.run_data)`, or the methods under `result.analytics.portfolio` |
| `result.portfolio_analytics` | `result.analytics.portfolio` |
| `result.risk` | `result.analytics.portfolio.risk` |
| `result.attribution` | `result.analytics.portfolio.attribution` |
| `result.run_data` | `result.analytics.run_data` |
| `result.analysis_request` | `result.analytics.request` |
| `result.analysis_timings` | `result.analytics.timings` |
| `result.analysis_block` | `result.analytics.to_dict()` |
| `result.modelling_assumptions` | `result.assumptions` |
| `result.get_returns()` | `result.analytics.portfolio.performance.returns()` |
| `result.get_annual_returns()` | `result.analytics.portfolio.performance.total_return(by="year")` |
| `result.money_weighted_return()` | `result.analytics.portfolio.performance.money_weighted_return()` |
| `result.get_tracking_error()` | `result.analytics.portfolio.relative.tracking_error()` |
| `result.get_tracking_difference()` | `result.analytics.portfolio.relative.tracking_difference()` |
| `result.information_ratio()` | `result.analytics.portfolio.relative.information_ratio()` |
| `result.active_share()` | `result.analytics.portfolio.relative.active_share()` |
| `result.active_attribution()` | `result.analytics.portfolio.attribution.active_attribution()` |
| `result.market_return()` | `result.analytics.portfolio.vehicle.market_return()` |
| `result.market_summary()` | `result.analytics.portfolio.vehicle.market_summary()` |
| `result.active_summary()` | `relative.information_ratio()`, `relative.active_share().mean()` and `relative.ex_ante_tracking_error().mean()`, under `result.analytics.portfolio` |

## Renamed names

What a strategy holds is called a portfolio throughout, and the names that said "book" follow. The old names warn until 0.12.0.

| 0.10.0 | 0.11.0 |
|---|---|
| `Book` | `PortfolioRecord` |
| `IndexBooks` | `IndexRecords` |
| `ShareClassBook` | `ShareClassRecord` |
| `book=` on the analysis classes' methods | `of=` |
| `BOOKS` | `PORTFOLIOS` |
| `book_net_assets` on `ShareClassRecord` | `base_net_assets` |
| `RiskOfLoss.books()` | `RiskOfLoss.portfolios()` |
| `class_tracking(book=)` | `class_tracking(share_class=)` |
| `books=` on the performance functions | `portfolios=` |
| `beacon.analysis.risk.forecast.book_returns` | `portfolio_returns` |
| `StoredSource(book=)` and `StoredSource.book` | `currency=` and `currency` |
| `currency_parts(book=)` | `base=` |
| `book_value` on `plan()`, `CapacityCap.max_value()` and `MinimumPosition.too_small()` | `portfolio_value` |

`PortfolioRecord`, `IndexRecords`, `FundRecord` and `TradingRecord` are exported from `beacon.backtest`. The analysis classes that took `book=` are `Attribution`, `PortfolioAnalytics`, `ConcentrationAnalytics`, `RiskOverTime`, `FactorAnalytics`, `LiquidityAnalytics` and `RiskOfLoss`, and `beacon.analysis.risk.forecast`'s functions took it too. A method given both `of=` and `book=` raises `TypeError`. The performance functions that took `books=` are `risk_rating`, `portfolio_risk_return`, `crisis_paths`, `return_distribution`, `characteristics_over_time`, `style_box` and `style_classification`. A share class's `base_net_assets` is its net assets in the fund's base currency, since `net_assets` is in the class's own. A capacity cap of your own that still names `max_value()`'s third parameter `book_value` keeps working, as the engine passes it by position.

## Changes without a warning

These can't warn, so code that relies on them needs changing now:

- **The portfolio column.** The tables with a column for each row's portfolio name it `portfolio`, not `book`: `risk_rating`, `portfolio_risk_return`, `crisis_paths` and `style_classification`. The characteristics table's column levels are `portfolio` and `figure`. Each statistic's `figure_names` starts with `portfolio`, and the server's `names` key is `portfolio`. The values, such as `fund` and `index`, are unchanged. The server renames the column in a run saved before 0.11.0 as it reads it.
- **A share class's net assets in the base currency.** The server's share-class field `book_net_assets` is `base_net_assets`, and the `class_days` table's `BOOK_NET_ASSETS` column is `BASE_NET_ASSETS`. A run or job saved before 0.11.0 is read with the new names.
- **The Lorenz curve.** `lorenz()`'s `share_of_book` column is `share_of_portfolio`.
- **The server's record schemas.** In `openapi.json`, `PortfolioBookPayload` is `PortfolioRecordPayload` and `BookPayload` is `RecordPayload`. The fields on the wire are unchanged, so only types generated from the schema change.
- **Short windows.** A figure annualised over too short a window is left blank, unless `annualise=True` asks for it. Annualised rates need at least a year. Ratios scaled by the square root of a year, such as the Sharpe ratio, need at least six months.
- **Removed statistics.** `trailing_statistics` is the `summary` statistic table every analysis holds, or `analytics.table([...], trailing=[...], of="all")`. `member_risk_return` is the per-name table at weekly returns, `result.analytics.instrument.table(frequency="weekly")`.
- **A statistic's part.** A statistic's `part` is where it sits in `result.analytics`: one of the nine categories, or `instrument`.
- **Active attribution's shape.** `result.active_attribution()` gave an object with a list of `contributions` and a residual. `result.analytics.portfolio.attribution.active_attribution()` gives a table instead: a row for each name and each group, with its `contribution`, `average_weight`, `total_return` and `level`. Costs and cash are not attributed to any name, so it has no residual row.
- **The catalogue's entry class.** `beacon.analysis.reports.statistics.Statistic` is `StatisticSpec`, so `Statistic` (`beacon.analysis.Statistic`) is the enum of statistic ids alone.
- **Not a dataclass.** `BacktestResult` is a plain class, built with the same keywords, so `dataclasses.replace()` and `fields()` no longer apply to it.

The [changelog](reference/changelog.md) lists every change in 0.11.0.
