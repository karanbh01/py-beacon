# Changelog

What changed in each release of py-beacon. The newest release is first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the version numbers follow [Semantic Versioning](https://semver.org/).
Before 1.0, a breaking change raises the middle number, as in 0.1 to 0.2.

## [Unreleased]

### Added

- Dividends in the backtest. A holding is paid each cash distribution on the pay date, for the shares held at the start of the ex-date, converted into the book's currency and net of `withholding_tax_rate` (a new modelling assumption). `Backtest(dividends=...)` accumulates the cash until the next rebalance (the default), reinvests it the same day, or distributes it, with returns adding distributions back. Each payment is recorded in `portfolio.cash_flows`.
- `Implementation`: how a backtest carries out its index at its size, without changing the index. Its screens decide which names may be held at each rebalance (`MarketCapScreen`, `LiquidityScreen`, `MinimumPriceScreen`, `ListingAgeScreen`, `ExclusionScreen`, `ExpressionScreen`), with optional exit levels (buffers) so names near a threshold do not flip in and out. Removed weight is spread pro rata across the remaining names or held as cash. Pass it as `Backtest(implementation=...)`.
- `BacktestResult.rebalance_steps`: for each rebalance, the target, the names removed and the screen that removed each, and the weights traded to.
- `ModellingAssumptions`: what a backtest takes as given about markets and data in one object. FX policy, stale-price threshold and free-float backfill (shared with the index calculation), and cash rate, risk-free rate and periods per year. Set a process-wide default with `use_modelling_assumptions`, and override it per backtest with `Backtest(modelling_assumptions=...)`. Results and index results record what they assumed. The defaults change no result.
- Cash can earn interest (`cash_rate`), recorded in `Portfolio.cash_flows`.
- `IndexResult.currency` and `BacktestResult.currency`: the currency an index's levels and a backtest's book are in. A backtest run from the engine reports its `currency` too.

### Changed

- A backtest over data with cash distributions now receives them, so its NAV and every figure from it rise by about the dividend yield. It earned the price return only, and trailed a total-return index by about that much.
- `ExpressionScreen` is a screen, passed in `Implementation(screens=[...])`, and no longer takes a `fetcher`. It was a trade modifier that cancelled buys of a failing name, left its weight in cash and only trimmed a failing holding. A failing name is now removed from the target, sold in full and its weight redistributed. It moved from `beacon.backtest.rules` to `beacon.backtest`, and a screen passed in `modifiers` is refused with a message saying where it goes.
- A backtest keeps its book in the index's currency unless `currency` is passed. It defaulted to USD whatever the index's currency, so a euro or sterling index was valued in dollars and its NAV picked up exchange-rate moves the index does not have. This includes backtests run from the engine and by an `IndexFund`.

### Fixed

- The index result cache keys on the data source's FX policy, stale-price threshold and free-float backfill. It keyed on the store alone, so changing one of those settings could reuse an index calculated under the old one.

## [0.3.1] - 2026-10-01

One fix: futures and roll pricing refuse bad input as a 422 rather than a 404.

### Fixed

- Futures and roll pricing answer 422 `INVALID_ARGUMENT`, not 404 `DATA_NOT_FOUND`, for a negative or missing time to expiry, an expiry before the valuation date, and a back expiry that is not after the front.

## [0.3.0] - 2026-09-30

Money in one currency across multi-currency indices, prices carried over gaps, expressions that refuse fields the data does not have, and engine state that survives. Two breaking changes: `data.actions` is gone from expressions, and some requests the engine cannot answer as put answer 422 instead of 404.

### Added

- `DataFetcher.fetch_prices` reads prices for several instruments at once, converted into one currency day by day under the dataset's FX policy.
- `IndexResult.price_gaps` and `beacon.index.PriceGap`: the days a held name had no bar and was valued at its last close. `PriceGap` is still importable from `beacon.backtest`.
- The asset view reports `currency`, which its returns, beta and tracking error are measured in (the index's), and `price_currency`, which its price series is in (the name's own).
- A feature import's response says whether the rows were `saved` into the served store, and gives the new `data_version`.
- `BacktestResult.get_annual_returns()`: calendar-year returns that compound to the whole run's return.
- `TotalReturnSwap` takes `underlying_type` (`INDEX`, the default, `ETF` or `EQUITY`). It was always `INDEX`.
- Each release from `GET /changelog` carries its `summary`, the prose between its heading and its first section.
- A risk model request takes an optional `currency`, and a risk model and an optimisation run report the currency they were measured in.

### Changed

- Requests that cannot be answered as put answer 422 `INVALID_ARGUMENT` instead of 404 `DATA_NOT_FOUND`: an unknown price interval, an adjusted series without `CLOSE`, and an empty or oversized feature batch.
- An expression with a field the data does not have is refused with `ExpressionError` by `universe.where`, `ExpressionScreen` and an index's `ExpressionRule`, naming the field and suggesting close matches. Before, it selected nothing.
- `IndexDefinition` refuses an unknown calendar or rebalancing frequency when it is built, naming the valid values and suggesting close calendar codes. Before, the mistake surfaced only when rebalance dates were first computed.
- The API schema states more of what it accepts: `since` on `/changelog` is a dotted version, and `/beacon/compare` needs at least two `ids`, each a valid identifier. These are now refused as `VALIDATION_ERROR` by the schema check. The boolean fields of a synthetic-data or import request no longer accept `0`, `1` or strings.

### Removed

- `data.actions` in expressions. No expression could read its fields, so every name was missing a value; it now raises `UnknownDatasetError`, and the field catalogue and `GET /data/fields` no longer list an `actions` namespace. Read corporate actions from the fetcher's `corporate_actions`.

### Fixed

- Money is compared in one currency wherever names are compared or added up. `LiquidityRule`'s traded-value floor, the engine's risk models, optimisation runs, the weights pane (drift, risk contributions, active risk), attribution and the asset view all read each name's prices in its own currency, so a multi-currency index mixed yen with dollars and left exchange-rate returns and risk out. They now convert into the index's currency.
- An index values a held name with no bar on a session at its last close, as the backtest engine does, and lists the day in `IndexResult.price_gaps`. It used to value the name at zero, so the level dipped by the name's weight for the day and recovered when the bar returned.
- An expression's `market_cap` inside an index is in the index's currency, as `MarketCapRule`'s bounds are. Outside an index it stays in USD.
- `POST /data/features` now saves the rows into the served store when it is a folder, so they survive a restart, and changes `data_version` and sends a `data.freshness` event for `features`. Before, a client caching on `data_version` missed the change and a restart lost the rows.
- The engine always keeps the saved results its views read: the latest backtest of each index, every optimisation run and the latest estimate of each risk model. The limit of 50 saved jobs now applies only to the rest, so a run of loads, refreshes or renders no longer makes those views answer 404.
- `data.market.free_float` and `free_float_market_cap` in an expression carry the free float forward as far as the data's `free_float_backfill_days`, as every other read does, rather than 10 days.
- Attribution for a capped index no longer fails when its window starts after the index's first rebalance. The cap drag is measured over the same periods as the contributions.
- A cancelled dividend is no longer reinvested by a total-return index, counted in the trailing dividend and yield, or applied to adjusted closes; a cancelled split is no longer applied to adjusted closes either.
- The engine checks for a token before it loads any data, so a missing token is reported at once rather than after a large store has loaded.
- `http://127.0.0.1` and `http://[::1]` on any port are allowed origins, as `http://localhost` already was.
- A wrong or missing token on the event socket closes it with code 1008 and a reason, as intended, instead of refusing the handshake with an HTTP 403 a browser cannot read.
- `BacktestPlots.annual_returns` measures each year from the previous year's close, and the first from the initial capital. It measured each year from its own first close, so every year after the first lost its first day's return.
- A chart drawn before `beacon.plot.use()` gets the light colours on matplotlib's white background, not the dark mode's.
- `OptimisationPlots.frontier` starts the capital market line at the rate the frontier was traced at, unless another is passed, so the line is tangent.
- The Excel reports accept a `pathlib.Path` as well as a string, and the holdings report writes `valuation_date` into its sheet instead of only logging it.
- `futures_roll_return` uses 365-day years, like every other year fraction in the derivatives. It used 365.25, so its rates came out smaller than the rest by a factor of 365/365.25.
- The sample dataset in `beacon.testing.dataset` stores its GBPUSD rate in `RATE`, so `fx_pairs` and the coverage view list it. A pair stored without `RATE` still converts but now logs a warning that it is not listed.
- A report template with an unknown page setting answers 422 instead of 500, and so does a futures or swap price whose inputs overflow.
- A plain `OPTIONS` request answers 405 with an `Allow` header listing every method its path supports. It listed only one route's methods on a path with several, such as `/data/features`. CORS preflights are unchanged.
- The nightly API fuzz run passes again: it leaves out operations that start heavy work, and skips the check that a schema-valid request is accepted only where a rule spans fields or depends on the data.
- Generating or extending synthetic data from an isolated engine (`python -I`, as the Beacon app runs it) keeps the child process isolated too, so it cannot load packages from the user's own site-packages.

## [0.2.0] - 2026-09-26

Choose where the engine's data comes from: named stores, synthetic data you can extend to today, CSV and Excel import, and read-only Postgres. Also documentation at pybeacon.dev, and fixes to index levels and backtests around splits and delistings.

### Added

- Documentation at https://pybeacon.dev: a guide to each part of py-beacon (data, universes, methodology, expressions, backtests, funds, derivatives, the optimiser, risk, attribution, charts, reports and the server), and a complete Python reference. Every example in it runs.
- The engine can start without any data and load a store later. Loading runs as a job with progress, and the event socket announces `data.loaded` when the new data is being served.
- Named data stores. Register a data folder under a name you choose, see every store at `GET /data/stores`, and switch which one the engine serves. The engine remembers the active store and serves it again on the next start.
- Generate synthetic data from the engine: `POST /data/synthetic` creates a new store with the size, dates and seed you choose (end date defaulting to today), and serves it when it is ready. It runs as a job with progress, and gives exactly the same data as `python -m beacon.synthetic` with the same settings.
- `python -m beacon.synthetic --progress` prints a line at each stage, for a program running it.
- Extend a generated store to today without changing its history: `python -m beacon.synthetic --extend PATH`, or `beacon.synthetic.extend`. Prices, exchange rates, listings, dividends, splits and features carry on from where the store stops, and the rows already there are never changed. The same store extended to the same date always gives the same data. Stores generated from this version on keep the settings this needs beside the data.
- Import your own data from CSV files or an Excel workbook: `POST /data/import` checks every row and saves a new store, or refuses with one finding per problem naming its sheet, row and column. `GET /data/import/template` downloads a blank template, and `beacon.data.importing.load_files` does the same from Python. Dates are read as YYYY-MM-DD only, so a date cannot be read as the wrong day.
- The `server` extra now includes `excel`, for Excel import.
- Refresh a store from its own source: `POST /data/stores/{id}/refresh` extends synthetic data to today, reads a folder or database again, and saves what changes. If the store is being served, the engine serves the refreshed data. Each store in `GET /data/stores` says what a refresh would do.
- A folder store can choose to refresh from Yahoo Finance instead (`refresh_from`), which downloads new prices and saves them into the folder. It is never the default.
- A data store can be a Postgres database. Give it tables or views named `market`, `reference`, and optionally `fx`, `corporate_actions` and `features`, with the import template's columns; views over your own tables work. The engine only reads, over a read-only connection, and checks every row as an import does. The password is never stored: name an environment variable that holds it. Needs the new `postgres` extra.

### Changed

- A backtest's return metrics start from its initial capital: `get_returns()` has one return per simulated day, the first from the capital to the first close. The tracking difference, tracking error, volatility and drawdown now include the cost of the opening trades, which they used to leave out, so a run with costs no longer shows a positive tracking difference for that reason.
- In a backtest result from the engine, `returns`, `drawdown` and `annual_returns` start from the initial capital too, so they agree with the metrics. `returns` now has one value per day of `level`, the first from the capital to the first close. `level` is unchanged.
- `POST /data/coverage/{dataset}/sync` is deprecated. It now refreshes the store being served from that store's own source, and no longer downloads from Yahoo Finance unless the store chose it. Its body is ignored, and its job is a `refresh:{store_id}` job. Use `POST /data/stores/{id}/refresh`.
- A request that needs data when none is loaded now answers 409 `NO_DATA_LOADED`, saying what could not be done. It answered 500 `CONFIGURATION_ERROR` before.
- `/health` says which store is being served and whether one is loading.
- `/health` and the data events carry `data_version`, a token that changes whenever the data being served changes. A client compares it to tell whether what it cached is still current.

### Fixed

- `IndexFund.calculate_nav` keeps the fund's trading cost when it extends a run to a later date. It used to re-run at zero cost.
- A split, reverse split or stock dividend no longer moves an index level or a backtest's NAV. On the ex-date the units or shares held change by the ratio, as the price does. Before, a split between rebalances cut the level and NAV by the split, including in synthetic data, which splits every year.
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

[Unreleased]: https://github.com/karanbh01/py-beacon/compare/v0.3.1...HEAD
[0.3.1]: https://github.com/karanbh01/py-beacon/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/karanbh01/py-beacon/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/karanbh01/py-beacon/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karanbh01/py-beacon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/karanbh01/py-beacon/releases/tag/v0.1.0
