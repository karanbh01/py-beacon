# Changelog

What changed in each release of py-beacon. The newest release is first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the version numbers follow [Semantic Versioning](https://semver.org/).
Before 1.0, a breaking change raises the middle number, as in 0.1 to 0.2.

## [Unreleased]

### Added

- Documentation at https://pybeacon.dev: a guide to each part of py-beacon (data, universes, methodology, expressions, backtests, funds, derivatives, the optimiser, risk, attribution, charts, reports and the server), and a complete Python reference. Every example in it runs.
- Named data stores. Register a data folder under a name you choose, see every store at `GET /data/stores`, and switch which one the engine serves. The engine remembers the active store and serves it again on the next start.
- Generate synthetic data from the engine: `POST /data/synthetic` creates a new store with the size, dates and seed you choose (end date defaulting to today), and serves it when it is ready. It runs as a job with progress, and gives exactly the same data as `python -m beacon.synthetic` with the same settings.
- `python -m beacon.synthetic --progress` prints a line at each stage, for a program running it.
- Extend a generated store to today without changing its history: `python -m beacon.synthetic --extend PATH`, or `beacon.synthetic.extend`. Prices, exchange rates, listings, dividends, splits and features carry on from where the store stops, and the rows already there are never changed. The same store extended to the same date always gives the same data. Stores generated from this version on keep the settings this needs beside the data.
- Import your own data from CSV files or an Excel workbook: `POST /data/import` checks every row and saves a new store, or refuses with one finding per problem naming its sheet, row and column. `GET /data/import/template` downloads a blank template, and `beacon.data.importing.load_files` does the same from Python. Dates are read as YYYY-MM-DD only, so a date cannot be read as the wrong day.
- The `server` extra now includes `excel`, for Excel import.
- Refresh a store from its own source: `POST /data/stores/{id}/refresh` extends synthetic data to today, reads a folder or database again, and saves what changes. If the store is being served, the engine serves the refreshed data. Each store in `GET /data/stores` says what a refresh would do.
- A folder store can choose to refresh from Yahoo Finance instead (`refresh_from`), which downloads new prices and saves them into the folder. It is never the default.
- A data store can be a Postgres database. Give it tables or views named `market`, `reference`, and optionally `fx`, `corporate_actions` and `features`, with the import template's columns; views over your own tables work. The engine only reads, over a read-only connection, and checks every row as an import does. The password is never stored: name an environment variable that holds it. Needs the new `postgres` extra.
- The engine can start without any data and load a store later. Loading runs as a job with progress, and the event socket announces `data.loaded` when the new data is being served.

### Changed

- `POST /data/coverage/{dataset}/sync` is deprecated. It now refreshes the store being served from that store's own source, and no longer downloads from Yahoo Finance unless the store chose it. Its body is ignored, and its job is a `refresh:{store_id}` job. Use `POST /data/stores/{id}/refresh`.
- A request that needs data when none is loaded now answers 409 `NO_DATA_LOADED`, saying what could not be done. It answered 500 `CONFIGURATION_ERROR` before.
- `/health` says which store is being served and whether one is loading.
- `/health` and the data events carry `data_version`, a token that changes whenever the data being served changes. A client compares it to tell whether what it cached is still current.

### Fixed

- A constituent delisted the session before a rebalance no longer takes its weight out of the index level. It leaves first, as on any other day.
- `SelectionResult.excluded_by` names the right step when prices go stale. With stale names dropped, it credited each exclusion to the step before it, and a stale name to the last rule.
- A damaged data file no longer stops the engine with a crash. It is refused with a message naming the file, and a damaged store found at startup is skipped with a warning.

## [0.1.1] - 2026-09-25

### Added

- `DataFetcher.fx_route` says how a currency pair is converted: from a stored rate, its inverse, or a cross through USD.

### Fixed

- A currency with no stored rate of its own is now converted using the inverse of the reverse pair, or a cross through USD. Before, an index in pounds holding US shares refused because only GBP to USD was stored.
- The quickstart in the README and on the docs home page runs again.

## [0.1.0] - 2026-09-24

The first release.

### Added

- Build an index from rules and a weighting scheme: equal weight, market cap or free-float market cap, with optional weight caps.
- Calculate the index level on a real exchange calendar, as price, total return or net total return.
- Adjust the index for dividends, special dividends and delistings.
- Write selection rules as expressions, such as `data.market.market_cap > 1e9`, including rules on company fundamentals and other features.
- Backtest a portfolio that trades to the index weights, with transaction costs, drift thresholds and a benchmark.
- Read results with tracking error, attribution, risk, concentration and drift, and draw them as charts.
- Hold names in several currencies. Prices are converted with FX rates.
- Load data from files, generate a realistic synthetic dataset, or download prices with yfinance. Data is kept in a local store that reports its coverage and age.
- Three settings, shown on `/health`: whether a missing FX rate carries forward, when a stale price drops a name, and how long a free float carries forward (90 days by default).
- A local API server for the Beacon desktop app, covering data, indices, universes, backtests, the optimiser, risk, derivatives and PDF reports. Long tasks run as jobs with live progress.
- This changelog, served at `GET /changelog` so the app can show what changed.
- Optional extras keep the core install small: `data`, `excel`, `pdf`, `optimise`, `plot` and `server`.

### Notes

- An index or backtest never uses a price, rate or free float dated after the day it is working on.
- Requires Python 3.11 or later.

[Unreleased]: https://github.com/karanbh01/py-beacon/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/karanbh01/py-beacon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/karanbh01/py-beacon/releases/tag/v0.1.0
