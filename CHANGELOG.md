# Changelog

What changed in each release of py-beacon. The newest release is first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the version numbers follow [Semantic Versioning](https://semver.org/).
Before 1.0, any release may change the API.

## [Unreleased]

## [0.1.0] - 2026-09-24

The first release.

### Added

- Build an index from rules and a weighting scheme: equal weight, market cap or free-float market cap, with optional weight caps.
- Calculate the index level on a real exchange calendar, as price, total return or net total return.
- Adjust the index for dividends, special dividends and delistings.
- Write selection rules as expressions, such as `data.market.market_cap > 1e9`, including rules on company fundamentals and other features.
- Backtest a portfolio that trades to the index weights, with transaction costs, drift thresholds and a benchmark.
- Read results with tracking error, attribution, risk, concentration and drift, and draw them as charts.
- Optimise an index under constraints (position limits, group limits, turnover, number of names), with the efficient frontier and factor exposures.
- Estimate risk models with shrinkage, and split risk and active risk by constituent.
- Price index futures, ETF futures and total return swaps, with carry, roll and a sensitivity grid.
- Hold names in several currencies. Prices are converted with FX rates.
- Load data from files, generate a realistic synthetic dataset, or download prices with yfinance. Data is kept in a local store that reports its coverage and age.
- Three settings, shown on `/health`: whether a missing FX rate carries forward, when a stale price drops a name, and how long a free float carries forward (90 days by default).
- A local API server for the Beacon desktop app, covering data, indices, universes, backtests, the optimiser, risk, derivatives and PDF reports. Long tasks run as jobs with live progress.
- This changelog, served at `GET /changelog` so the app can show what changed.
- Optional extras keep the core install small: `data`, `excel`, `pdf`, `optimise`, `plot` and `server`.

### Notes

- An index or backtest never uses a price, rate or free float dated after the day it is working on.
- Requires Python 3.11 or later.

[Unreleased]: https://github.com/karanbh01/py-beacon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/karanbh01/py-beacon/releases/tag/v0.1.0
