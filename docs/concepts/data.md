---
title: Data
description: "The data model, the three data settings, currencies, and every way to get data in."
---

# Data

Every calculation in py-beacon reads its data through one object, a
`DataFetcher`. Index construction, backtests, screens, attribution and the
engine's HTTP API all ask it the same questions, so they all get the same
answers. This page covers what the fetcher holds, the three settings that
decide how it reads, how currencies are converted, and the ways to get data
into it.

## The data model

A fetcher holds up to four datasets. Only market data is required.

| Dataset | Container | Keyed by | What it holds |
| --- | --- | --- | --- |
| Market data | `MarketData` | `IDENTIFIER`, `DATE` | Prices, volume, shares outstanding, free float, and FX rates |
| Reference data | `ReferenceData` | `IDENTIFIER`, `DATE_FROM` to `DATE_TO` | Names, currencies, exchanges, classifications |
| Corporate actions | `CorporateActions` | `IDENTIFIER`, `EX_DATE` | Dividends, splits and structural events |
| Features | `FeatureData` | `IDENTIFIER`, `DATE` | Any other datapoint: fundamentals, alternative data, your own values |

Each container is built from a pandas DataFrame with `from_dataframe`, and
checks its required columns when it is built. Column names are upper case.

### Market data

Long form: one row per identifier per date, with `IDENTIFIER` and `DATE`
columns. Every series has the same shape, whatever it is: an equity's price, a
currency pair's exchange rate and a short-term interest rate are all in
`CLOSE`, and reference data says which is which (see
[What a series is](#what-a-series-is)). `CLOSE` is the price every calculation
reads unless it is told otherwise (the index calculator and the backtest engine
both take a `price_column`). The other known columns are optional:

| Column | Meaning |
| --- | --- |
| `OPEN`, `HIGH`, `LOW` | The day's other prices |
| `VOLUME` | Shares traded, used by liquidity rules and `adv_3m` |
| `SHARES_OUTSTANDING` | Used for market caps and market-cap weighting |
| `FREE_FLOAT` | The investable fraction of the shares, between 0 and 1 |

Any other column is kept, and can be read with `fetch_market_data` or named as
a `price_column`. A date with no row for a name means it did not trade that
day. When loaded, the frame is indexed on `(IDENTIFIER, DATE)` and sorted.

### Reference data

Descriptive fields that change rarely, stored point in time: each row is valid
from `DATE_FROM` to `DATE_TO`, both inclusive. A blank or missing `DATE_TO`
means the row is still current. A name that changes sector has two rows, and a
question about a date gets the row in force on that date.

`NAME` and `CURRENCY` are the columns every series needs, and an equity also
needs its `EXCHANGE`: an index builds each universe member from them. Anything
else (`SECTOR`, `REGION`, `COUNTRY_LISTING`, `ISIN` and so on) is kept and can
be screened on. A name whose every row has a `DATE_TO` is treated as delisted
after the latest one.

#### What a series is

These columns say what each series is, and so how its `CLOSE` is read. Values
are matched in any case.

| Column | Meaning |
| --- | --- |
| `ASSET_CLASS` | `Equity`, `Currency` or `Fixed Income`. Blank, or no column at all, is `Equity`. Any other value is kept and read like an equity. |
| `SECURITY_TYPE` | For a currency or fixed income series, required: `Spot` for a currency pair, `Short Rate` for a short-term interest rate. Optional for an equity (`Common Stock`, `ETF`, ...). |
| `BASE_CURRENCY`, `QUOTE_CURRENCY` | A spot pair's two currencies. `CLOSE` is units of the quote currency per unit of the base, so a `GBP`/`USD` pair at 1.27 means one pound buys 1.27 dollars. Its `CURRENCY` must be the quote currency. |
| `TENOR` | A short rate's tenor, such as `ON`, `1M` or `3M`. Optional. |
| `DAY_COUNT` | A short rate's day count: `ACT/360` (the default) or `ACT/365`. |
| `TRADABLE` | Whether a fund can hold the series: yes or no (`TRUE`/`FALSE`, `Y`/`N` and 1/0 also work). |

A short rate's `CURRENCY` is the currency it is for, and its `CLOSE` is an
annual decimal: 0.0525 for 5.25%. A rate above 1 is refused as probably a
percent; a negative rate is fine.

Currency pairs and short rates are references, not holdings, so they are not
tradable unless `TRADABLE` says so. A series that is not tradable is never
offered as a universe member: `instrument_identifiers` and
`tradable_identifiers` leave it out, and `is_tradable` says which a series is.

`pair_reference` and `short_rate_reference` build these rows:

```python
from beacon.data import pair_reference, short_rate_reference

pair_reference("GBPUSD")                       # GBP quoted in USD, read from the name
pair_reference("Cable", base="GBP", quote="USD")
short_rate_reference("SONIA", "GBP", tenor="ON", day_count="ACT/365")
```

### Corporate actions

One row per action: `IDENTIFIER`, `EX_DATE`, `TYPE` and `VALUE`, with optional
`PAY_DATE` and `STATUS` (`announced`, `paid` or `cancelled`). What `VALUE`
means depends on the type:

| Kind | Types | `VALUE` is |
| --- | --- | --- |
| Cash | `DIVIDEND`, `SPECIAL_DIVIDEND`, `RETURN_OF_CAPITAL` | An amount per share. Amounts add up. |
| Ratio | `SPLIT`, `REVERSE_SPLIT`, `STOCK_DIVIDEND` | A multiplier on the share count. Ratios compound. |
| Structural | `RIGHTS_ISSUE`, `SPIN_OFF`, `MERGER` | Nothing that can be added up |

Any other type is refused when the data is loaded. `fetch_trailing_dividend`
and `fetch_trailing_dividend_yield` sum ordinary dividends over the trailing
twelve calendar months: an action dated exactly one year before the as-of date
has rolled out, and one dated on it is in. How an index adjusts for actions is
covered in [Methodology](methodology.md).

### Features

Everything that is not a price, a reference field or an action shares one
table: `IDENTIFIER`, `DATE`, `TYPE`, `FIELD` and `VALUE`, with an optional
free-text `DETAIL`. `TYPE` names the dataset a value came from, so `revenue`
from a vendor and `revenue` from your own model stay separate.

`DATE` is **when the value became knowable**, not the period it describes. Q1
revenue published on 15 May is dated 15 May, and the period goes in `DETAIL`.
A read on a date sees the latest value dated on or before it, so a backtest
standing on 1 April cannot see that Q1 figure. Storing the period end in
`DATE` would let every screen see numbers before anyone had them.

A restatement is a new row with a later `DATE`, and both rows are kept, so a
backtest standing between them sees the original figure. Two rows with the
same identifier, date, type and field are a duplicate, and the last one wins.

`fetch_feature` also refuses a value that is too old: by default one more than
730 days before the date counts as missing. Pass `max_age_days` to change the
limit, or None to remove it.

### The fetcher

`DataFetcher(market_data, reference_data=None, corporate_actions=None,
features=None, ...)` wraps the containers. A fetcher with no actions or
features holds empty ones, so asking it never needs a check for None first.
The questions it answers include:

| Method | Answers |
| --- | --- |
| `fetch_market_data(identifier, start, end)` | Market rows: indexed by `DATE` for one identifier, by `(IDENTIFIER, DATE)` for a list |
| `fetch_price(identifier, date)` | One close, or None if the name did not trade |
| `fetch_reference_data(identifier, date)` | The reference rows in force on the date |
| `fetch_classification(identifier, date, scheme="SECTOR")` | One classification on the date |
| `fetch_shares_outstanding(identifier, date)` | That day's value, or None |
| `fetch_free_float_factor(identifier, date)` | The free float in force, carried forward (see [below](#free-float-backfill)), or None |
| `fetch_corporate_actions(identifier, start, end, types)` | Actions by ex-date |
| `fetch_feature(identifier, field, date, feature_type)` | The feature value knowable on the date, or None |
| `fx_rate_on(from_ccy, to_ccy, date)` | The rate converting one currency into another, or None |

Here is a small dataset built by hand: two names, one of which moves sector,
and two FX pairs.

```python
import logging

import pandas as pd

from beacon.data import DataFetcher, MarketData, ReferenceData, pair_reference

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

days = ["2024-01-02", "2024-01-03", "2024-01-04"]

market = MarketData.from_dataframe(pd.DataFrame({
    "IDENTIFIER": ["AAA"] * 3 + ["BBB"] * 3 + ["GBPUSD"] * 2 + ["EURUSD"] * 2,
    "DATE": days + days + [days[0], days[2]] * 2,
    "CLOSE": [100.0, 101.0, 102.0, 50.0, 50.5, 51.0, 1.27, 1.26, 1.09, 1.10],
    "SHARES_OUTSTANDING": [1e6] * 3 + [2e6] * 3 + [None] * 4,
    "FREE_FLOAT": [0.9, None, None, 0.8, 0.8, 0.8, None, None, None, None],
}))

reference = ReferenceData.from_dataframe(pd.DataFrame([
    {"IDENTIFIER": "AAA", "NAME": "Alpha plc", "CURRENCY": "GBP", "EXCHANGE": "XLON",
     "SECTOR": "Energy", "DATE_FROM": "2020-01-01"},
    {"IDENTIFIER": "BBB", "NAME": "Beta Inc", "CURRENCY": "USD", "EXCHANGE": "XNYS",
     "SECTOR": "Technology", "DATE_FROM": "2020-01-01", "DATE_TO": "2024-01-03"},
    {"IDENTIFIER": "BBB", "NAME": "Beta Inc", "CURRENCY": "USD", "EXCHANGE": "XNYS",
     "SECTOR": "Financials", "DATE_FROM": "2024-01-04"},
    pair_reference("GBPUSD", date_from="2020-01-01"),
    pair_reference("EURUSD", date_from="2020-01-01"),
]))

fetcher = DataFetcher(market, reference)

print(fetcher.fetch_price("AAA", "2024-01-03"))                  # 101.0
print(fetcher.fetch_classification("BBB", "2024-01-03"))         # Technology
print(fetcher.fetch_classification("BBB", "2024-01-04"))         # Financials
print(fetcher.fetch_shares_outstanding("BBB", "2024-01-02"))     # 2000000.0
print(fetcher.instrument_identifiers)                            # ['AAA', 'BBB']
print(fetcher.fx_pairs)                                          # ['EURUSD', 'GBPUSD']
```

Features go in the same way:

```python
from beacon.data.features import FeatureData

features = FeatureData.from_dataframe(pd.DataFrame({
    "IDENTIFIER": ["AAA", "AAA"],
    "DATE": ["2024-02-15", "2024-05-15"],   # when each figure was published
    "TYPE": "fundamentals",
    "FIELD": "revenue",
    "VALUE": [1.2e9, 1.3e9],
    "DETAIL": ["FY2023", "Q1 2024"],
}))

fetcher = DataFetcher(market, reference, features=features)

print(fetcher.fetch_feature("AAA", "revenue", "2024-02-14"))     # None: not yet published
print(fetcher.fetch_feature("AAA", "revenue", "2024-04-01"))     # 1200000000.0
print(fetcher.fetch_feature("AAA", "revenue", "2024-06-03"))     # 1300000000.0
```

`MarketData(path)` and `ReferenceData(path)` also read a CSV or Excel file
directly, but without the checks described in
[Importing CSV or Excel](#importing-csv-or-excel). Prefer the importer for
files you did not write yourself.

## Three settings

Three choices change the numbers a calculation produces, and neither answer
is right for everyone. Each is set once on the fetcher and applies to every
calculation that reads through it, so an index and its backtest cannot
disagree about them. `beacon.data.store.load` takes the same three arguments,
so a store can be opened under any of them. The engine serves every store
with the defaults and reports the settings in force on `GET /health`.

| Setting | Default | Allowed values |
| --- | --- | --- |
| `fx_policy` | `"CARRY_FORWARD"` | `"CARRY_FORWARD"` or `"EXACT_DAY"` |
| `max_price_staleness_days` | None (keep every name) | None, or a whole number of days, 1 or more |
| `free_float_backfill_days` | 90 | A whole number of days, 0 or more |

A value outside these raises `ValueError` when the fetcher is built.

### FX policy

How a rate is read on a day the pair did not print one.

- **`CARRY_FORWARD`** uses the last rate published on or before the day. That
  is the rate anyone could have dealt at, and it suits valuing a portfolio
  every day.
- **`EXACT_DAY`** answers only when the pair printed a rate on that very day.
  Choose it when a conversion stands for a cash event on a specific date, such
  as a dividend, where an approximate rate is worse than being told there is
  none.

Neither ever uses a rate dated after the day, so a date before a pair's first
rate has no rate under either policy. When there is no rate, a calculation
that needs one refuses with a `CalculationError` naming the pair, and a screen
treats the value as missing. Nothing converts at an assumed rate of 1.

```python
print(fetcher.fx_rate_on("GBP", "USD", "2024-01-03"))   # 1.27, carried from 2 January
print(fetcher.fx_rate_on("GBP", "USD", "2024-01-01"))   # None: before the first rate

exact = DataFetcher(market, reference, fx_policy="EXACT_DAY")
print(exact.fx_rate_on("GBP", "USD", "2024-01-03"))     # None: no rate printed that day
```

### Stale prices

`max_price_staleness_days` is how many calendar days a name may go without a
price before it is no longer worth holding. A name that stops trading without
its reference data recording an end date otherwise keeps its last price
forever, and an index goes on holding it at that frozen price.

With a threshold set, a name whose last price on or before a date is more than
that many days old is:

- dropped from index selection before any eligibility rule runs, and recorded
  as its own step in the selection results;
- dropped from a backtest's target weights at each rebalance, with the
  remaining weights rescaled to sum to 1, so the rebalance sells the holding
  in the ordinary way.

A name with no price at all is not reported as stale: that is a different
problem, handled where the price is needed. `stale_identifiers` shows
what a threshold would drop:

```python
strict = DataFetcher(market, reference, max_price_staleness_days=5)

print(sorted(strict.stale_identifiers(["AAA", "BBB"], pd.Timestamp("2024-01-08"))))  # []
print(sorted(strict.stale_identifiers(["AAA", "BBB"], pd.Timestamp("2024-01-12"))))  # ['AAA', 'BBB']
```

### Free-float backfill

Free float changes on corporate events and index reviews, not daily, so a
blank `FREE_FLOAT` cell usually means nothing was reported that day rather than
that the float changed. `fetch_free_float_factor` therefore uses that day's
value if there is one, and otherwise the last value reported within the
previous `free_float_backfill_days` calendar days. It never uses a later value.

The default of 90 days covers a quarterly review cycle with room to spare.
Widen it if your source reports free float less often; set 0 to use only a
value dated that day. There is no unlimited setting: a float last seen a year
ago is not the float.

Every float-adjusted calculation (free-float market-cap weighting, the index's
market values and the special-dividend adjustment) reads through this, and
refuses with a `CalculationError` when there is no value within the window.
Using the full market cap instead would weight the name as though every share
were freely traded.

```python
print(fetcher.fetch_free_float_factor("AAA", "2024-01-04"))   # 0.9, from 2 January

no_backfill = DataFetcher(market, reference, free_float_backfill_days=0)
print(no_backfill.fetch_free_float_factor("AAA", "2024-01-04"))   # None
```

## Currencies and FX pairs

An FX pair is a market series of its own, its rate in `CLOSE`, with a reference
row that gives it `ASSET_CLASS` `Currency`, `SECURITY_TYPE` `Spot` and its
`BASE_CURRENCY` and `QUOTE_CURRENCY` (see [What a series is](#what-a-series-is)).
A pair is found by its two currencies, so its identifier can be anything:
`GBPUSD`, `GBP/USD WMR 4pm` or a vendor's code. A series named like a pair but
with no such reference row is not a pair. `fx_pairs` lists the pairs that have
market data. Two pairs for the same two currencies are refused.

Market data with a `RATE` column, the way pairs were stored before 0.7.0, is
refused with a message saying what to change.

`fx_rate_on` finds a rate in this order:

1. **Direct**: the stored pair itself.
2. **Inverse**: one over the stored reverse pair, so `GBPUSD` also gives USD to
   GBP. A stored rate of zero or below has no inverse, and that day has no
   rate.
3. **Cross via USD**: two legs through the dollar, each found by rule 1 or 2,
   so `GBPUSD` and `EURUSD` together give GBP to EUR.

A currency converts into itself at 1. `fx_route` says which route a pair takes,
or None when there is none:

```python
print(fetcher.fx_route("GBP", "USD"))   # direct
print(fetcher.fx_route("USD", "GBP"))   # inverse
print(fetcher.fx_route("GBP", "EUR"))   # cross via USD
print(fetcher.fx_route("GBP", "JPY"))   # None
print(round(fetcher.fx_rate_on("GBP", "EUR", "2024-01-02"), 4))   # 1.1651
```

A cross follows the FX policy. Under `CARRY_FORWARD` each leg is the rate in
force on the day; under `EXACT_DAY` a cross exists only on days both legs
printed. `fx_rates_on` answers the same question for many days at once, and
`fetch_fx_rates` returns a stored pair's raw series without inverting or
crossing it.

### Short rates

A currency's short-term interest rate is a series too, described as
`Fixed Income`/`Short Rate`. `short_rate_on(currency, date)` gives the rate in
force on a day (the last on or before it, never a later one), and
`fetch_short_rates` the series. Where a currency has several, the one month
rate is read, else the overnight, else the first by identifier; the choice is
logged. `short_rate(currency)` says which series it is, with its tenor and day
count.

## Getting data in

There are four ways in, and each ends in a `DataFetcher`.

| Source | Use | Needs |
| --- | --- | --- |
| A data store folder | Saving and reloading data, and what the engine serves | Nothing extra |
| CSV files or an Excel workbook | Bringing your own data, checked row by row | Nothing extra |
| A Postgres database | Reading data you already hold, in place | Nothing extra |
| Yahoo Finance | Downloading real prices | `data` extra |

### Data stores

A data store is a folder holding a `manifest.json` and one Parquet file per
dataset: `market.parquet` (always), and `reference.parquet`,
`corporate_actions.parquet` and `features.parquet` when the data has them. FX
pairs are rows in the market file. The manifest records the format version and
where the rows came from (`synthetic`, `yfinance`, `imported`, `local`, or any
name you pass). Any Parquet reader opens the files, for example
`pandas.read_parquet`.

Stores saved before py-beacon 0.9.6 are gzipped CSV. The first load converts
one to Parquet in place and removes the CSV files, after the Parquet files
are written. The data reads the same either way. A store in a folder that
cannot be written to stays as CSV, and loads from it each time with a
warning.

`beacon.data.store.save(fetcher, path, source="local")` writes one, and
`beacon.data.store.load(path, ...)` reads it back, with the three settings as
keyword arguments. Saving the same data twice produces the same bytes. A
folder that is not a store, a store written by a newer py-beacon, or a damaged
file raises a `ConfigurationError` saying which.

```python
from pathlib import Path

from beacon.data import store

store.save(fetcher, Path("my-store"))
print(sorted(file.name for file in Path("my-store").iterdir()))
# ['features.parquet', 'manifest.json', 'market.parquet', 'reference.parquet']

loaded = store.load(Path("my-store"), fx_policy="EXACT_DAY")
print(loaded.source, loaded.fx_policy)   # local EXACT_DAY
```

`store.default_path()` is the folder the engine loads when it is given no
other data. See [Serving data](../serving-data.md)
for how the engine chooses and registers stores.

### Importing CSV or Excel

`beacon.data.importing` loads your own files in a fixed layout. Give CSV files
named after their sheet (`market.csv`, `reference.csv`, ...), or one Excel
workbook (`.xlsx` or `.xlsm`) whose sheets carry those names. Names are
matched ignoring case and spaces, so a sheet called "Corporate Actions" is the
`corporate_actions` sheet and a column called "close" is `CLOSE`.

| Sheet | Required columns | Optional columns |
| --- | --- | --- |
| `market` (required) | `IDENTIFIER`, `DATE`, `CLOSE` | `OPEN`, `HIGH`, `LOW`, `VOLUME`, `SHARES_OUTSTANDING`, `FREE_FLOAT` |
| `reference` (required) | `IDENTIFIER`, `NAME`, `CURRENCY`, `DATE_FROM`, and `EXCHANGE` for an equity | `DATE_TO`, `EXCHANGE`, `ASSET_CLASS`, `SECURITY_TYPE`, `BASE_CURRENCY`, `QUOTE_CURRENCY`, `TENOR`, `DAY_COUNT`, `TRADABLE` |
| `corporate_actions` | `IDENTIFIER`, `EX_DATE`, `TYPE`, `VALUE` | `PAY_DATE`, `STATUS` |
| `features` | `IDENTIFIER`, `DATE`, `FIELD`, `VALUE` | `TYPE` (default `imported`), `DETAIL` |

Extra columns are kept, so the reference sheet can carry `SECTOR`, `COUNTRY`
or anything else. Currency pairs and short rates go in the market sheet like
any other series, with a reference row each saying what they are. There is no
`fx` sheet any more: one is refused, with a message saying where its rows go.

`importing.template("xlsx")` returns a blank workbook with every sheet's
columns and example rows (an equity, a currency pair and a short rate);
`template("csv")` returns a zip of CSV files.

Every row is checked before anything is loaded:

- Dates must be written `YYYY-MM-DD` (a time after the date is allowed, as
  Excel date cells carry one). `01/02/2024` is refused rather than guessed,
  because it means January on one machine and February on another.
- Numbers must be numbers; `CLOSE` must be above zero (a short rate's may be
  negative, but not above 1) and `FREE_FLOAT` between 0 and 1.
- Required cells must be filled, a `DATE_TO` cannot be before its
  `DATE_FROM`, and an action's `TYPE` and `STATUS` and a rate's `DAY_COUNT`
  must be known values (in any case).
- Each reference row must describe its series properly: an equity has an
  `EXCHANGE`, a spot pair has both currencies and is quoted in its quote
  currency, a short rate has a currency, and no two pairs quote the same
  currencies.
- No two rows may share a key (such as `IDENTIFIER` and `DATE` in the market
  sheet), and every identifier in the market, actions and features sheets must
  be in the reference sheet.

If anything fails, `load_files` raises `DataImportError` and loads nothing.
The error lists every problem at once (up to 200, with the total), each naming
its sheet, its row as a spreadsheet shows it (the header is row 1) and its
column, so the files can be fixed in one pass. `error.findings` gives the same
problems as dictionaries, which is what the engine sends a client.

```python
from beacon.data import importing
from beacon.data.importing import DataImportError

pd.DataFrame({"IDENTIFIER": ["AAA", "AAA", "BBB"],
              "DATE": ["2024-01-02", "2024-01-03", "02/01/2024"],
              "CLOSE": [100.0, 101.0, -1.0]}).to_csv("market.csv", index=False)
pd.DataFrame({"IDENTIFIER": ["AAA"], "NAME": ["Alpha plc"], "CURRENCY": ["GBP"],
              "EXCHANGE": ["XLON"], "DATE_FROM": ["2020-01-01"]}
             ).to_csv("reference.csv", index=False)

try:
    importing.load_files(["market.csv", "reference.csv"])
except DataImportError as error:
    for problem in error.problems:
        print(problem.sheet, problem.row, problem.column, problem.message)

# market 4 CLOSE CLOSE must be above zero.
# market 4 DATE DATE '02/01/2024' is not a date written as YYYY-MM-DD.
# market 4 IDENTIFIER 'BBB' is not in the reference sheet.
```

`load_files` returns a fetcher with the default settings. To choose them, or to
save the result as a store, pass the fetcher on as in the sections above.

### Postgres

A Postgres store is tables or views in one schema, named after the sheets
above (`market` and `reference`, and optionally `corporate_actions` and
`features`), with the same columns. To use data you already hold, create views
with those names and columns over your own tables. The rows go through the
same checks as an import.

py-beacon only reads: it connects with a read-only session and never creates,
changes or deletes anything. The password is never stored. Name an
environment variable that holds it (`password_env`), and it is read each time
py-beacon connects; leave it as None for a server that needs no password.

<!-- not run: needs a Postgres server -->
```python
from beacon.data.postgres import PostgresSource, load

source = PostgresSource(host="localhost", database="markets",
                        user="analyst", password_env="MARKETS_PASSWORD",
                        schema="public", port=5432)
data = load(source, fx_policy="CARRY_FORWARD")
```

`load` passes the three settings on to the fetcher. A database that cannot be
reached, a missing password variable, or a table that cannot be read raises
`DataImportError`, as a bad row does.

### Yahoo Finance

`beacon.data.ingest` downloads daily history and reference fields, one
identifier at a time. A name that fails is recorded in the result's `failed`
mapping, with the reason, and the run carries on. Prices are unadjusted
(`Close`, not `Adj Close`), and a name whose download has no close price is
recorded as failed.

```bash
pip install "py-beacon-kit[data]"
```

<!-- not run: downloads from Yahoo Finance -->
```python
from beacon.data import DataFetcher, MarketData, ReferenceData, ingest

names = ["AAPL", "MSFT"]
prices = ingest.ingest_market_data(names, ingest.yfinance_downloader(),
                                   start="2024-01-01", end="2024-12-31")
details = ingest.ingest_reference_data(names,
                                       ingest.yfinance_reference_downloader())
print(prices.summary())

downloaded = DataFetcher(MarketData.from_dataframe(prices.market),
                         ReferenceData.from_dataframe(details.reference))
```

Reference rows are stamped valid from 1900-01-01, because a download carries
no history of when a field changed. Any function with the signature
`(identifier, start, end) -> DataFrame` can stand in for the downloader.

## Synthetic data

`beacon.synthetic` generates a whole market to work with: anonymised companies
(`Company A` and so on, every ticker starting `CMP`, so nothing can be
mistaken for a real listing), with prices that behave like equities
(volatility clustering, fat tails, crises at their real dates, and names that
move together by sector and region). It produces every dataset:

- market data with OHLC, volume, shares outstanding and free float;
- reference data with sectors, regions, identifiers and listing details;
- dividends and splits that match the prices;
- FX pairs for the six non-dollar listing currencies (EUR, HKD, JPY, GBP, CAD,
  AUD), each stored against USD, such as `EURUSD`;
- features: fundamental ratios and a little alternative data, derived from
  the prices so a valuation screen and a price screen agree;
- names listing and delisting over time, at 3% a year each by default.

`SyntheticConfig` sets what to generate: `assets` (512 by default), `start`
and `end` (a fixed window, 2019-12-31 to 2024-12-31, by default), `seed`,
`risk_free_rate`, `equity_premium`, `delisting_rate`, `listing_rate`,
`features` and `calendar` (the exchange whose sessions the data has bars on,
`XNYS` by default). The same config always gives the same data. `generate`
returns the dataset, and `write` generates one and saves it as a store:

```python
from beacon.synthetic import SyntheticConfig, extend, generate, write

config = SyntheticConfig(assets=20, start="2024-01-02", end="2024-06-28", seed=7)

synthetic = generate(config).fetcher()
print(synthetic.fx_pairs)          # ['AUDUSD', 'CADUSD', 'EURUSD', 'GBPUSD', 'HKDUSD', 'JPYUSD']
print(synthetic.feature_types())   # ['alternative', 'fundamentals']

write(config, Path("synthetic-store"))
```

Nothing should depend on its exact values: the model's parameters are tuned as
it improves.

### From the command line

```bash
python -m beacon.synthetic                              # 6,000 names, ten years to today
python -m beacon.synthetic --assets 200 --start 2020-01-02 --end 2024-12-31 --seed 1 --out my-store
```

The command line defaults to 6,000 names over the ten years ending today, and
writes to the folder the engine loads by default (pass `--out` to choose
another). The seed fixes the draw but not the dates, so pass both `--start`
and `--end` when you need the same data twice. `--extended-universe` (10,000
names) and `--long-history` (back past every modelled crisis) each roughly
double the work, and the command warns before a run that needs a lot of
memory. `--no-features` skips the features.

### Extending a store

Regenerating with a later end date changes every past price, because the whole
history is drawn from one random stream. `extend` instead carries a generated
store on to a new date and leaves every row already in it as it was:

```python
added = extend(Path("synthetic-store"), end="2024-07-31")
print(added.first, added.last, added.sessions)   # 2024-07-01 2024-07-31 22
```

Each name carries on from its last close with the parameters it was generated
with; FX pairs, listings, delistings, dividends, splits and features continue
too. Three kinds of existing record change, because they describe a state
rather than a day: a name that delists gets its end date, a dividend whose pay
date arrives becomes paid, and each name's next earnings date moves on.
Extending the same store to the same date always gives the same data. With no
`end`, it extends to today.

```bash
python -m beacon.synthetic --extend my-store              # to today
python -m beacon.synthetic --extend my-store --end 2025-06-30
```

The generator's parameters are saved in a `synthetic` folder inside the store.
A store without it (one you imported, say) loads normally but cannot be
extended: `extend` raises `ValueError`.

## The sample dataset

`beacon.testing.dataset` is a tiny, fixed dataset for examples and tests. It
never changes, and it is identical to the last digit on every platform, so a
number printed in an example or asserted in a test stays true.

- Six names from 2023-01-02 to 2025-12-31 on New York Stock Exchange sessions:
  `AAA` and `BBB` move together, `CCC` is defensive, `DDD` is volatile, `EEE`
  lags, and `FFF` trades in GBP.
- Market data with OHLC, volume, shares outstanding and free float, and the
  `GBPUSD` pair as its own identifier.
- Reference data with names, currencies, exchanges and sectors, valid for the
  whole span. No corporate actions or features.

```python
from beacon.testing import dataset

sample = dataset.data_fetcher()
print(dataset.UNIVERSE)                                      # ('AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF')
print(sample.fetch_classification("FFF", "2024-06-03"))      # Financials
print(dataset.prices().loc["2024-01-02", "AAA"])             # 137.896804
```

It also offers `prices()`, `returns()`, `fx_rates()`, `sectors()`,
`equal_weights()`, and `index_result_from_weights` for building a backtest
schedule by hand. Use it for anything that needs a known number, and the
synthetic generator for anything that needs to look like a market.

## See also

- [Expressions and screens](expressions.md): selecting names with the data
  described here.
- [Serving data](../serving-data.md): how the engine loads and refreshes
  stores.
- Reference: [Data](../reference/data.md),
  [Synthetic data](../reference/synthetic.md),
  [Sample dataset](../reference/testing.md).
