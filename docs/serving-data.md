# Serving data

py-beacon includes a local engine: an HTTP server that the Beacon desktop app,
or any other client, starts on the same machine. The engine holds the data and
runs the calculations (index levels, backtests, attribution, risk), and the
client asks for results over HTTP.

This page covers starting the engine and where its data comes from. For
authentication, jobs, live events and errors, see
[The engine's HTTP API](server.md).

## Quick start

```bash
pip install "py-beacon-kit[server]"
python -m beacon.synthetic                     # 6,000 names over ten years
python -m beacon.server --port 0 --token dev
```

The first command writes a synthetic data store to the app-data folder. The
second takes no data argument: when no data store is registered yet, the
engine registers the one in the app-data folder and serves it.

You can also start the engine with no data at all and load some while it runs:
generate synthetic data, register a folder or database, or import files. See
[Loading data while the engine runs](#loading-data-while-the-engine-runs).

## Why the client talks HTTP

The client never opens the data files. It does not know the store is gzipped
CSV, so the format can change without any change to the client.

The data is not the product; the calculations are. A client reading the store
directly would have to reimplement the index maths, and then two
implementations would have to agree.

## Starting the engine

```bash
python -m beacon.server --port 0 --token <secret>
```

| Option | Default | What it does |
| --- | --- | --- |
| `--host` | `127.0.0.1` | The interface to bind. Keep it on loopback: the engine has no TLS and trusts its bearer token alone. |
| `--port` | `0` | The port to bind. `0` lets the operating system pick a free one, which the engine then announces. |
| `--token` | `$BEACON_API_TOKEN` | The bearer token every request must carry. Required: with neither the option nor the variable, the engine exits. |
| `--data` | none | A data-store folder to serve, instead of the registered stores. |
| `--documents` | the app-data folder | Where the engine keeps what it saves (see below). |
| `--cors-origin` | `beacon://app`, `app://` | An exact origin allowed to call the engine from a browser. Repeatable, and replaces the defaults. `http://localhost` on any port is always allowed. |

| Environment variable | Used when |
| --- | --- |
| `BEACON_API_TOKEN` | `--token` is not given. |
| `BEACON_DATA_PATH` | `--data` is not given. A data-store folder to serve. |
| `BEACON_CORS_ORIGINS` | `--cors-origin` is not given. Comma-separated origins; replaces the defaults. |

### What the engine saves

Everything the engine saves goes under the `--documents` folder, one
subfolder per collection: indices, universes, watchlists, constraint sets,
report templates, backtest records, job results, rendered reports, the list of
registered data stores, and the data stores the engine creates itself
(generated or imported).

Without `--documents`, this is the platform app-data folder; on Windows,
`%LOCALAPPDATA%\beacon\beacon`. The default synthetic data store sits in the
same folder, under `market-store`, and `beacon.data.store.default_path()`
returns its path.

!!! note "Microsoft Store Python"
    With the Microsoft Store build of Python, Windows redirects this folder
    into the Python package's own storage, under
    `%LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.x_...\LocalCache\Local\beacon`.
    Look there if the folder above is empty.

Pass `--documents` with a temporary folder when you try the engine out, so
nothing is written into the real app-data folder.

## The startup handshake

1. **The client starts the engine**, with `--port 0` and a token it generated.

2. **The engine loads its data.** It reads the data store it will serve (see
   [Where the data comes from](#where-the-data-comes-from)) before it binds
   a port, so data is already loaded when the client first calls.

3. **The engine binds a port and announces it.** The first line on stdout is

    ```
    BEACON_PORT=52612
    ```

    flushed at once, because the client is waiting for it. The socket is
    already listening, so the client can call straight away: a request that
    arrives before the server has started waits in the socket's queue.

4. **The engine logs where its data came from**, on stderr:

    ```
    INFO:__main__:Data source: the data store 'Synthetic data' (C:\...\market-store).
    INFO:__main__:Allowed origins: beacon://app, app:// (plus localhost on any port).
    ```

    Everything after the port line, on stdout or stderr, is ordinary logging.

5. **The client makes ordinary requests**, with the same token:

    ```
    GET http://127.0.0.1:52612/health
    Authorization: Bearer <secret>
    ```

If the engine cannot start, it prints `error: <reason>` on stderr and exits
with code 2 before announcing a port. That happens when no token is given, or
when `--data` or `BEACON_DATA_PATH` names data that cannot be read. So a
client that sees the process exit before a `BEACON_PORT=` line should report
stderr.

The engine does not watch the process that started it. The client stops it
when it is done. Jobs still running are lost; finished job results are saved
(see [Jobs](server.md#jobs)).

## Where the data comes from

At startup the engine takes the first of these that applies:

| | Source | If it cannot be read |
| --- | --- | --- |
| 1 | `--data <folder>` | The engine exits with code 2. Naming data that cannot be read is a mistake worth stopping for. |
| 2 | `$BEACON_DATA_PATH` | The engine exits with code 2, for the same reason. |
| 3 | The active registered store: the one served last time | The engine logs a warning and starts with no data, so a store that was moved or damaged never stops it from starting and offering another. |
| 4 | If no store is registered at all but the app-data folder holds one, that store, registered as "Synthetic data" (or "My data" if it was not generated) and made active | As for 3. |
| 5 | Nothing | The engine starts with no data. |

Data named by `--data` or `BEACON_DATA_PATH` is served but not registered as
a store, so it has no store id and cannot be refreshed.

The "Data source" log line names the branch that ran, for example
`Data source: no data loaded: 'My data' is unreadable.`, so an empty client
can be explained by reading the log.

## When there is no data

The engine still starts, and everything that does not read data works.
`GET /health` reports `data_source.configured: false`. A request that needs
data answers 409 with the code `NO_DATA_LOADED`, which is what a client should
branch on to offer loading data. `GET /data/identifiers` and
`GET /data/coverage` answer normally, with nothing in them.

## Loading data while the engine runs

A data store holds one dataset: a py-beacon data folder or a Postgres
database. The engine keeps a list of registered stores
(`GET /data/stores`), serves one at a time, and remembers which, so the next
start serves the same one.

| To | Call | Answer |
| --- | --- | --- |
| Register a data folder or a Postgres database | `POST /data/stores` | 201 with the store. The folder must be an absolute path holding `manifest.json` and `market.csv.gz`; a database is read in full before it is registered. |
| Serve a registered store | `POST /data/stores/{id}/activate` | 202 with a `load:{id}` job. |
| Generate synthetic data into a new store | `POST /data/synthetic` | 202 with a `generate:{id}` job. The store is served when it is ready, unless `activate` is false. |
| Import CSV files or an Excel workbook into a new store | `POST /data/import` | 201 with the new store, plus a `load:{id}` job when it is being served. `GET /data/import/template` gives the layout. |
| Rename a store or change its refresh source | `PATCH /data/stores/{id}` | 200 with the store. |
| Forget a store | `DELETE /data/stores/{id}` | 204. A folder you registered is left as it is; a store the engine created (`managed: true`) has its folder deleted. The store being served cannot be forgotten (409). |

Loading runs as a job because reading a large store takes seconds. Until it
finishes, the engine serves the previous data, and `/health` shows
`data_source.loading: true`. Only one load runs at a time: another activation
while one is loading is refused with 409 `CONFLICT`. Generation and import
started during a load still create their store, but leave it for you to
activate.

When the new data is being served, the event socket announces `data.loaded`
with the store and a new `data_version`. A client should then refetch
anything it derived from the data. A request already running when the switch
happens finishes with the data it started with.

Generation runs `python -m beacon.synthetic` in a child process with every
setting spelled out, so the same settings give the same data whichever way it
was made. Cancelling the job stops the child and removes the half-written
store.

An import checks every row before saving anything. If a row is wrong, the
answer is 422 `INVALID_RULE` with one finding per problem, each naming its
sheet, row and column.

A Postgres store needs the `postgres` extra. The engine only reads it, and
never stores the password: the store names an environment variable that
holds it.

## Refreshing a store

`POST /data/stores/{id}/refresh` brings a store up to date from its own
source, as a `refresh:{id}` job:

| Store | What a refresh does |
| --- | --- |
| Synthetic data | Extends it to today, or to `end` in the request body, keeping every day it holds. |
| A folder | Reads it again, picking up files changed outside the engine. |
| A database | Reads its tables again. |
| Imported files | Nothing. The request is refused with 409; import the files again instead. |

A folder or database that is not being served is refused with 409 too: it is
read afresh whenever it is activated. So is a store that is already
refreshing, and, for the store being served, a refresh while another load is
running.

A folder store can refresh from Yahoo Finance instead, by setting
`refresh_from` to `yfinance` when it is registered or later with `PATCH`.
Its refresh then downloads new prices for its instruments and saves them into
the folder. This needs the `data` extra and is never the default. Synthetic
data and databases cannot choose it.

Whatever a refresh changes is saved, so it survives a restart. If the store is
the one being served, the engine serves the refreshed data and announces it
with `data.loaded` and `data.freshness`, both carrying the new `data_version`.
Each store in `GET /data/stores` says what a refresh would do in its
`refresh` field (`extend`, `reread` or `download`), which is null when there
is nothing to refresh.

`POST /data/coverage/{dataset}/sync` is deprecated. It refreshes the store
being served, whichever dataset is named, and ignores its body.

## Generating a store from the command line

`python -m beacon.synthetic` produces market-like data: a factor model with
GJR-GARCH volatility, Student-t innovations and negative skew, plus reference
data, shares outstanding, free float, dividends and splits that agree with the
prices, and exchange rates.

It also generates **features**: four fundamental ratios (`pe_ratio`,
`pb_ratio`, `eps`, `debt_to_equity`) quarterly, and two alternative series
(`x_sentiment`, `wikipedia_views`) monthly. The ratios are derived from the
price path rather than drawn beside it, so `pe_ratio x eps` is exactly the
close at the period end. Announcement lags vary by name and quarter, and
coverage is deliberately incomplete, so a point-in-time read has a ragged edge
to resolve.

| Flag | Default |
| --- | --- |
| `--assets` | 6,000 |
| `--start` / `--end` | The ten years ending today |
| `--seed` | 42 |
| `--risk-free-rate` / `--equity-premium` | 0.03 and 0.06, annualised |
| `--calendar` | `XNYS`: the exchange whose trading days have prices |
| `--out` | The app-data store the engine finds by itself |
| `--extended-universe` | Off. Widens the universe to 10,000 names. |
| `--long-history` | Off. Reaches back past every crisis the generator models. |
| `--no-features` | Off. Skips the ratios and alternative data (about 8% of rows). |
| `--progress` | Off. Prints a `BEACON_PROGRESS <fraction> <stage>` line per stage, for a program running the command. |
| `--extend PATH` | Off. Extends an existing store instead of generating one. |

Both expansion flags widen a default rather than overrule a value you gave:
an explicit `--assets` beats `--extended-universe`, and an explicit `--start`
beats `--long-history`. `--long-history` starts a year before the earliest
crisis the generator models, rather than a fixed number of years back, so the
crises stay in range as today moves.

The dates default to today because data ending months ago shows as stale in
every freshness indicator. Pass both `--start` and `--end` when you need the
same data on another day: the seed fixes the draw, not the calendar. The same
seed and dates give byte-identical files.

Nothing generated resembles a real company. Names are `Company A` and so on,
and every ticker starts with `CMP`, so a collision with a real listing is
impossible.

### Size and memory

The default run peaks at about 2.5 GB of memory. Each expansion flag roughly
doubles the work; both together are about five times the default and need
around 10 GB. The command prints its row count and memory estimate before it
starts, and warns when the estimate reaches 4 GB. Narrow the universe with
`--assets` or the window with `--start` on a smaller machine.

Delisted names have their rows removed rather than kept as empty rows, so a
store holds fewer rows than its universe size times its trading days.

### Extending a store to today

Generating again with a later end date would change every past price, because
the whole history is drawn from one random stream. To bring a store up to
date without that, extend it:

```bash
python -m beacon.synthetic --extend PATH               # up to today
python -m beacon.synthetic --extend PATH --end 2026-06-30
```

The market carries on from where the store stops: each name from its last
close and share count, each exchange rate from its last rate, with new
listings, delistings, dividends, splits and features at the same rates as
before. Rows already in the store never change; new rows are added after
them. The only records that change are ones describing a state: a name that
delists gets its end date, a dividend whose pay date arrives becomes paid, and
next earnings dates move forward.

The same store extended to the same date always gives the same data. A store
generated before py-beacon 0.1.2 cannot be extended, because it lacks the
generator settings saved beside the data; generate a new one.

## Reading the data

The endpoint-by-endpoint reference is at [pybeacon.dev/api](https://pybeacon.dev/api/).
A few things are worth knowing before you use it.

### Market caps come in pairs

`GET /data/reference` can return `market_cap` and `free_float_market_cap`,
each twice:

| Field | What it is |
| --- | --- |
| `market_cap_local` | Price times shares as the exchange reports it, in `local_currency` |
| `market_cap` | The same figure converted, in `market_cap_currency` |

`?currency=EUR` sets what the converted figure is in; it is `USD` when you say
nothing. Use the currency of the index you are comparing against, because a
cap and a weight in one row only compare while they share a unit. The local
figure is never converted away: it is a fact about the company, and what you
would check against another source. A missing FX rate nulls only the
converted figure.

### Adjusted prices

`GET /data/prices/{identifier}?adjusted=true` adds an `ADJ_CLOSE` column:
`CLOSE` back-adjusted for splits and dividends.

An adjusted series is not a price. It answers "what would a holder have
made", so a chart of it is a total-return chart. It is adjusted backwards, so
its last value equals the last raw close and only history moves. That also
means the whole series shifts when a new action lands, so do not treat a
cached adjusted series as fixed. It is computed per request for that reason.

### Browsing a whole dataset

`GET /data/tables/{dataset}` pages through `market`, `reference`,
`corporate_actions` or `features`, as the usual `{index, columns, data}` frame
plus a `total`. Pages are at most 1,000 rows, the order is stable, and an
`offset` past the end gives an empty page.

`identifiers` narrows it to some instruments, comma-separated or repeated.
That is how to read one instrument's history, such as every feature value it
has carried:

```
GET /data/tables/features?identifiers=CMPA&limit=1000
```

The filter applies before paging, so `total` counts the filtered rows. An
identifier the dataset does not hold adds no rows rather than failing. There
is no sorting or filtering beyond that; use the expression API for those.

### Notes for a client

**Branch on a corporate action's `kind`, not its `type`.** `kind` is `cash`,
`ratio` or `structural`, and says what `value` means. A type the client has
never seen would otherwise be rendered as whatever its list defaults to.

**Take staleness from `/data/coverage`.** Each dataset there carries its
`frequency` and `stale_after_seconds`, so a client does not need thresholds of
its own that can drift from the engine's.

**Read `identifiers_union`, not a sum.** Per-dataset counts overlap. Currency
pairs are market identifiers, so they are counted under `market` too; the
`fx` row answers "are exchange rates held" and reports a null
`cache_size_bytes`, because its rows live in the market file.

**A currency pair is an ordinary identifier.** `EURUSD` answers on
`/data/prices` and appears in `/data/identifiers`. It has no reference data.
`RATE` is filled on a pair and null on every instrument, which is how to tell
them apart.

**`null` means unknown, not zero.** A missing `pay_date`, `adv_3m` or
`risk_contribution` is null rather than a placeholder, so a client can leave
the field out rather than show a dash, which would read as "there is none".

**CORS matters only if a browser page calls the engine directly.** Requests
from the app's main process are not browser requests, so CORS never applies
to them.
