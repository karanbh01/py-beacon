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
columns. `CLOSE` is the price every calculation reads unless it is told
otherwise (the index calculator and the backtest engine both take a
`price_column`). The other known columns are optional:

| Column | Meaning |
| --- | --- |
| `OPEN`, `HIGH`, `LOW` | The day's other prices |
| `VOLUME` | Shares traded, used by liquidity rules and `adv_3m` |
| `SHARES_OUTSTANDING` | Used for market caps and market-cap weighting |
| `FREE_FLOAT` | The investable fraction of the shares, between 0 and 1 |
| `RATE` | An exchange rate, on an FX pair's rows only (see [Currencies](#currencies-and-fx-pairs)) |

Any other column is kept, and can be read with `fetch_market_data` or named as
a `price_column`. A date with no row for a name means it did not trade that
day. When loaded, the frame is indexed on `(IDENTIFIER, DATE)` and sorted.

### Reference data

Descriptive fields that change rarely, stored point in time: each row is valid
from `DATE_FROM` to `DATE_TO`, both inclusive. A blank or missing `DATE_TO`
means the row is still current. A name that changes sector has two rows, and a
question about a date gets the row in force on that date.

`NAME`, `CURRENCY` and `EXCHANGE` are the columns every name needs: an index
builds each universe member from them. Anything else (`SECTOR`, `REGION`,
`COUNTRY_LISTING`, `ISIN` and so on) is kept and can be screened on. A name
whose every row has a `DATE_TO` is treated as delisted after the latest one.

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

from beacon.data import DataFetcher, MarketData, ReferenceData

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

days = ["2024-01-02", "2024-01-03", "2024-01-04"]

market = MarketData.from_dataframe(pd.DataFrame({
    "IDENTIFIER": ["AAA"] * 3 + ["BBB"] * 3 + ["GBPUSD"] * 2 + ["EURUSD"] * 2,
    "DATE": days + days + [days[0], days[2]] * 2,
    "CLOSE": [100.0, 101.0, 102.0, 50.0, 50.5, 51.0, None, None, None, None],
    "SHARES_OUTSTANDING": [1e6] * 3 + [2e6] * 3 + [None] * 4,
    "FREE_FLOAT": [0.9, None, None, 0.8, 0.8, 0.8, None, None, None, None],
    "RATE": [None] * 6 + [1.27, 1.26, 1.09, 1.10],
}))

reference = ReferenceData.from_dataframe(pd.DataFrame({
    "IDENTIFIER": ["AAA", "BBB", "BBB"],
    "NAME": ["Alpha plc", "Beta Inc", "Beta Inc"],
    "CURRENCY": ["GBP", "USD", "USD"],
    "EXCHANGE": ["XLON", "XNYS", "XNYS"],
    "SECTOR": ["Energy", "Technology", "Financials"],
    "DATE_FROM": ["2020-01-01", "2020-01-01", "2024-01-04"],
    "DATE_TO": [None, "2024-01-03", None],
}))

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

An FX pair is stored in the market data as an identifier of its own, named
`FROMTO` in upper case, with the rate in `RATE`. `GBPUSD` at 1.27 means one
pound buys 1.27 dollars. The `RATE` column is what marks a pair:
`fx_pairs` lists the identifiers that have a `RATE` value, and
`instrument_identifiers` lists everything else. A pair may also carry its rate
in `CLOSE`, so that it can be charted like any other identifier.

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

## Getting data in

There are four ways in, and each ends in a `DataFetcher`.

| Source | Use | Needs |
| --- | --- | --- |
| A data store folder | Saving and reloading data, and what the engine serves | Nothing extra |
| CSV files or an Excel workbook | Bringing your own data, checked row by row | `excel` extra for Excel |
| A Postgres database | Reading data you already hold, in place | `postgres` extra |
| Yahoo Finance | Downloading real prices | `data` extra |

### Data stores

A data store is a folder holding a `manifest.json` and one gzipped CSV per
dataset: `market.csv.gz` (always), and `reference.csv.gz`,
`corporate_actions.csv.gz` and `features.csv.gz` when the data has them. FX
pairs are rows in the market file. The manifest records the format version and
where the rows came from (`synthetic`, `yfinance`, `imported`, `local`, or any
name you pass). The files open in any text editor after `gunzip`.

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
# ['features.csv.gz', 'manifest.json', 'market.csv.gz', 'reference.csv.gz']

loaded = store.load(Path("my-store"), fx_policy="EXACT_DAY")
print(loaded.source, loaded.fx_policy)   # local EXACT_DAY
```

`store.default_path()` is the folder the engine loads when it is given no
other data; it needs the `server` extra. See [Serving data](../serving-data.md)
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
| `reference` (required) | `IDENTIFIER`, `NAME`, `CURRENCY`, `EXCHANGE`, `DATE_FROM` | `DATE_TO` |
| `fx` | `PAIR` (such as `GBPUSD`), `DATE`, `RATE` | |
| `corporate_actions` | `IDENTIFIER`, `EX_DATE`, `TYPE`, `VALUE` | `PAY_DATE`, `STATUS` |
| `features` | `IDENTIFIER`, `DATE`, `FIELD`, `VALUE` | `TYPE` (default `imported`), `DETAIL` |

Extra columns are kept, so the reference sheet can carry `SECTOR`, `COUNTRY`
or anything else. The `fx` sheet's pairs become market-data identifiers.

`importing.template("xlsx")` returns a blank workbook with every sheet's
columns and one example row (it needs the `excel` extra, as reading a workbook
does); `template("csv")` returns a zip of CSV files.

Every row is checked before anything is loaded:

- Dates must be written `YYYY-MM-DD` (a time after the date is allowed, as
  Excel date cells carry one). `01/02/2024` is refused rather than guessed,
  because it means January on one machine and February on another.
- Numbers must be numbers; `CLOSE` and `RATE` must be above zero, and
  `FREE_FLOAT` between 0 and 1.
- Required cells must be filled, a `DATE_TO` cannot be before its
  `DATE_FROM`, a `PAIR` must be six letters (such as `GBPUSD`), and an action's `TYPE`
  and `STATUS` must be known values (in any case).
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
above (`market` and `reference`, and optionally `fx`, `corporate_actions` and
`features`), with the same columns. To use data you already hold, create views
with those names and columns over your own tables. The rows go through the
same checks as an import.

py-beacon only reads: it connects with a read-only session and never creates,
changes or deletes anything. The password is never stored. Name an
environment variable that holds it (`password_env`), and it is read each time
py-beacon connects; leave it as None for a server that needs no password.

```bash
pip install "py-beacon-kit[postgres]"
```

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
