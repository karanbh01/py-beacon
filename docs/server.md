---
title: Server guide
description: "Using the engine's HTTP API: authentication, jobs, live events and error codes."
---

# The engine's HTTP API

The local engine (`python -m beacon.server`) answers HTTP on a loopback port.
This page explains how the API works as a whole: authentication, the endpoint
groups, jobs, live events and errors. The fields of each endpoint are in the
[API reference](/api/index.md), generated from the engine's
OpenAPI document. Starting the engine and choosing its data are covered in
[Serving data](serving-data.md).

The examples on this page run the engine in-process with FastAPI's
`TestClient`, which needs `httpx`:

```bash
pip install "py-beacon-kit[server]" httpx
```

## Authentication

Every request carries the token the engine was started with, as a bearer
token:

```
Authorization: Bearer <token>
```

The token comes from `--token`, or from `BEACON_API_TOKEN` when the option is
not given, and the engine refuses to start without one. The client that
starts the engine generates it, so the engine can tell its own client from
any other process on the machine: binding to loopback keeps other machines
out, not other programs.

Every HTTP route needs the token, `/health` and `/changelog` included. A
missing or wrong token answers 401 with the code `UNAUTHORIZED` and a
`WWW-Authenticate: Bearer` header.

The event socket cannot carry a header, because browsers cannot set one on a
WebSocket handshake, so it takes the token as a query parameter:
`/ws?token=<token>`. With a wrong or missing token the socket opens and is
closed at once with code 1008 and the reason `Invalid bearer token.`, which a
browser can read, unlike a refused handshake.

```python
from pathlib import Path

from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app

config = ServerConfig(auth_token="dev", storage_root=Path("engine-documents"))
auth = {"Authorization": "Bearer dev"}
client = TestClient(create_app(config))

refused = client.get("/health")
assert refused.status_code == 401
assert refused.json()["error"]["code"] == "UNAUTHORIZED"

health = client.get("/health", headers=auth).json()
print(health["version"], health["data_source"])
```

`storage_root` is the Python form of `--documents`: where the engine saves its
documents. Point it at a folder of your own when trying things out.

## Starting up

A client needs three things from startup (the full sequence is in
[Serving data](serving-data.md#the-startup-handshake)):

- Read stdout until the line `BEACON_PORT=<port>`. It is the first line, and
  it is printed only after the startup data has loaded.
- If the process exits before printing it, startup failed: stderr holds a
  line starting `error:`, and the exit code is 2.
- Call `GET /health` to learn what is being served. `data_source.configured`
  says whether any data is loaded, `data_source.store_id` and `store_name`
  say which, `data_source.loading` says whether a new store is being loaded,
  and `data_source.data_version` identifies the data (see
  [Data versions](#data-versions)). `version` is the engine's version.

## Endpoint groups

| Prefix | What it is for |
| --- | --- |
| `/health` | Whether the engine is up, what data it serves, and the modelling settings in force (`fx_policy`, `max_price_staleness_days`, `free_float_backfill_days`). |
| `/changelog` | The engine's release notes. See [Release notes](#release-notes). |
| `/data/identifiers`, `/data/prices`, `/data/reference`, `/data/corporate-actions`, `/data/features`, `/data/fields`, `/data/tables` | Reading the loaded data: search names, price history, reference fields (including derived ones such as market caps), corporate actions, features, the field catalogue, and raw pages of each dataset. `POST /data/features` adds feature rows to the served data and saves them into its store when that is a folder; a Postgres store is read-only, so there they last until other data is loaded or the engine stops. |
| `/data/coverage` | What each dataset holds, how old it is, and when it goes stale. |
| `/data/stores` | Registered data stores: list, register, rename, forget, activate, refresh. |
| `/data/synthetic` | Generate synthetic data into a new store, as a job. |
| `/data/import` | Import CSV files or an Excel workbook into a new store, and download the template. |
| `/data/watchlists` | Saved lists of identifiers. |
| `/universes` | Saved universes: the pools of names an index selects from, and their members on a date. A `GLOBAL` universe covering the loaded data is kept up to date automatically and cannot be edited. |
| `/indices` | Index definitions: save, validate, preview, the rebalance schedule, the rule types and calendars an editor can offer, deriving an optimised index from a saved one, and an index's level calculated on its own (`GET /indices/{id}/level`). |
| `/strategies` | Stored strategies: what a backtest holds and how it trades. List, read, save, check without saving, and delete; backtest one (a job), check a backtest request without running it, and read the results of its latest run: the record, overview, weights, attribution, one asset, a comparison of several strategies, and every strategy's latest run. See [Stored strategies](#stored-strategies) and [Backtest requests](#backtest-requests). |
| `/backtest/options` | Everything a backtest request may carry: each family's types and fields, the vehicle presets, and the modelling assumptions with their defaults. See [Backtest requests](#backtest-requests). |
| `/optimise` | Constraint sets, and optimisation runs (a job) with their frontier and exposures. |
| `/risk-models` | Estimating a risk model (a job) and reading the estimated models. |
| `/reports` | Report templates, rendering a report to PDF (a job), and downloading the PDF. |
| `/derivatives` | Pricing index futures and total return swaps, and an index's futures term structure and roll. |
| `/jobs` | Polling and cancelling jobs. |
| `/ws` | The event socket. |

An endpoint that needs data answers 409 `NO_DATA_LOADED` when none is
loaded. The rest, such as listing saved indices or reading `/health`, work
without data.

## Backtest requests

Every backtest runs a [stored strategy](#stored-strategies), which holds what
is held, the index it is measured against and how it trades.
`POST /strategies/{id}/backtest` takes the rest: the dates, the initial
capital and the trading cost, and, optionally, every other part of a
[backtest](concepts/backtest.md):

| Field | What it says |
| --- | --- |
| `flows` | Money arriving and leaving, as a list of scenarios, or with share classes a list for each class by name. |
| `share_classes` | The fund's share classes, sharing one book: each a `name`, and optionally a `currency`, `hedged`, `hedge_ratio`, `hedge_band`, `management_fee_bps` and `distribution`. With classes, `initial_capital` is by class name, each in its class's currency. See [Funds](concepts/funds.md). |
| `vehicle` | A preset (`uk_oeic`, `ucits_etf`, ..., or null for the generic vehicle) and the settings to change. See [Fund vehicles and pricing](concepts/fund-vehicles.md). |
| `modelling_assumptions` | What the run takes as given; each unset field takes the engine's default. |
| `dividends`, `currency` | The dividend policy, and the book's currency. |

A request without them runs exactly as one with only the dates, capital and
cost. A body carrying `strategy` or `implementation` is refused: those are
the stored strategy's. Configurable objects use one shape, `{"type": ..., "params": {...}}`,
and `GET /backtest/options` lists every type with its fields, and every setting a
request can carry, `run_settings` describing the dates, capital, cost,
currency, dividends and benchmark: their types,
defaults, choices, units (`fraction` is shown as a percentage, `money` is in
the book's currency) and bounds. It needs no data.

A vehicle's `settings` are laid over its preset's: a typed object (`pricing`)
or a list (`limits`) replaces the preset's whole, a plain object (`market`)
merges key by key, an omitted key keeps the preset's value, and null means
none, refused on a setting that cannot be none. Settings that apply only to
exchange-traded vehicles say so in `applies_to` and are refused on any other.

A request whose settings cannot be built is refused with 422
`INVALID_RULE` before any job starts, with a finding for each problem at its
field's path, such as `vehicle.settings.pricing.params.threshold`.
`POST /strategies/{id}/backtest/validate` takes the same body and answers
the fully resolved settings, or the findings, without running anything.

The run's result carries its resolved `settings`, starting with the
`strategy_id`, `strategy_name` and `index_id` it ran, and, as the run uses them,
`nav_per_unit`, `units` and the `flows` ledger, an ETF's `market` quotes, the
`replication` or `active` record of each rebalance, the implementation's
`rebalance_steps` and the `unfilled` orders. `nav_per_unit`, `units` and every
`market` series share `level`'s dates; with flows, `level` is NAV per unit,
so a flow is not drawn as a gain.

With share classes the result carries `share_classes`, one record per class
in the request's order, and `hedge_value`, the hedged classes' forwards
together in the book's currency. A class's record is in its own currency, its
series on `level`'s dates: its net assets (and `book_net_assets`, in the
book's currency, which add up to the fund's), units, NAV per unit, the fees
and distributions of each day and their totals, its flows as dealt, and for a
hedged class its `hedge_value` and each trade of its `hedges`. Its `metrics`
give its total return, tracking difference and tracking error against the
index in its currency: converted at each day's rate for an unhedged class,
and hedged the same way for a hedged one, so a hedged class's tracking
difference leaves out the interest rate differential its hedge earns or
pays. `GET /backtest/options` describes each class field in
`share_class_settings`, and a problem with one is a finding at its path, such
as `share_classes.1.hedge_band` or `initial_capital.EUR`.

A synthetic fund's result carries its swap: `swap_value` each day (owed to the
fund when positive, and part of the NAV), `counterparty_exposure` (what the
fund would lose if the counterparty failed, as a fraction of NAV) and each of
`swap_resets` (its date, the value `settled`, the `reason` and the `notional`
it was reset to). A fund holding index futures, through futures-based
replication or by equitising a cash buffer, carries `futures_notional` and
`futures_exposure` each day (in money, 0 before they open, and as a fraction
of NAV) and each of `futures_rolls`: the date, the new `expiry`, the
`contracts` held and their `price` (each contract on one unit of the index,
so contracts times price is the `notional`), the roll's `cost` and the
`dividend_yield` it was priced with. These share `level`'s dates, money is in
the book's currency, and each is null or empty for a fund without them.

A run's `comparisons` compare it with its `index` (the index it tracked, or the
one an active strategy is built against) and its `benchmark` (the request's
external benchmark, or null). Each gives the comparator's `level`, rebased to
100 on the first of the days both have, its `own` figures over those days
(total and annualised return, volatility, Sharpe ratio, maximum drawdown and
monthly hit rate, defined as the run's metrics are) and the run's `relative`
figures against it: relative and annualised relative return, tracking error,
information ratio (the annualised relative return over the tracking error),
correlation, beta and the share of calendar months the run beat it. The
relative figures against the index agree with the run's `tracking_error` and
`tracking_difference`. The run's own metrics include `hit_rate_monthly`, the
share of calendar months with a positive return.

`GET /strategies/{id}/periods?frequency=annual|quarterly|monthly` gives the
latest run's figures in each calendar year, quarter or month (named `2025`,
`2025 Q3` or `2025-07`): the run's total return, volatility, Sharpe ratio,
maximum drawdown and share of positive days, and against its `index` and its
`benchmark` the same `own` and `relative` figures as its comparisons, or null.
Each period runs from the previous period's close and the first from the
run's start, so the periods compound to the whole run's figures.

Runs are kept per strategy, so two strategies on one index each keep their
own. `GET /strategies/{id}/overview`, `/record`, `/weights`, `/attribution`
and `/assets/{identifier}` read the strategy's latest run, and answer 404
until it has one. Each names the strategy and the index it holds.
`GET /strategies/{id}/result` gives the latest run's whole payload, the one
its job answered with, so a client needs no job id to read it again, and it is
kept across a restart.
`GET /strategies/compare?ids=a&ids=b` puts several strategies' latest runs on
one axis, and `GET /strategies/runs` lists every strategy's latest saved run,
newest first. A strategy's weights view takes `benchmark`, an index to
measure tracking error against, which is calculated on its own.

## An index on its own

`GET /indices/{id}/level` calculates a stored index alone, with no portfolio,
costs or vehicle, so an index's history can be shown before anything is
backtested. It answers the index's `level` from its base value, its headline
`metrics` (total and annualised return, volatility, Sharpe ratio and maximum
drawdown, defined as a backtest's are) and its weights at each rebalance in
`rebalances`. `start` and `end` set the window; by default it runs from the
base date to the last date the loaded data covers. The calculation goes
through the cache backtests use, so a backtest over the same dates reuses it.

## A strategy's holdings workflow

`GET /strategies/{id}/holdings?date=YYYY-MM-DD` shows the latest run's
holdings at a date, one row per name, from the universe to the units held. It
reads the rebalance in force on `date` (`rebalance_date`; the run's last day
when no date is given) and lists every rebalance of the run in
`rebalance_dates`.

`stages` are the workflow's steps in order, each with its `key`, the `label` a
column reads, its `kind` (`universe`, `index_rule`, `index_cap`, `strategy`,
`implementation_screen`, `cash_buffer`, `implementation_cap`) and the count
`remaining` after it. Stale prices are screened in every run; the capacity step
appears where something could cap. Each row gives the name's `excluded_at`
stage, its index weight before and after the index's cap, its weight in the
backtest's benchmark index, the strategy's own weight, its weight after each
implementation stage (`implementation`, by stage key), the `portfolio_weight`
the rebalance traded to, and as of `date` its `current_weight`, `weight_drift`
(current less portfolio, null on the rebalance date) and `shares_held`, with
its reference columns (see below). A held weight is the units at the day's
price over the day's NAV, what the book recorded. A run saved before 0.9.3 has
no saved stages: its rows show the index's weights alone until it is run
again.

## Reference columns on holding rows

Every row that names a holding carries its reference columns as they stood on
that row's date, by the rule `GET /data/reference?date=` uses, so a past
holding shows the classification of its time rather than today's. That covers
the weights view's rows (as of `asof`), each rebalance of a run's result (as of
its date) with its unfilled orders and price gaps, the record's rebalances
and orders likewise and each of its weights panels as `reference_windows`
(each name's columns with the span over which they held, so any day of the
panel reads the classification of its time), attribution's rows (as of
the window's end), an optimisation run's rows (`GET /optimise/runs/{id}`) and
its frontier (as of the target's date), and a constituent preview's assets (as
of the preview date).

`reference_fields`, a comma-separated list such as `name,sector`, chooses the
columns: any column of the loaded reference data, keyed in lower case. Without
it a row carries the name, asset class, sector and the dataset's country column
(`country_domicile`, else `country`, else `country_listing`); an empty value
carries none. A column that says what a series is (`asset_class`,
`security_type`, `tradable`) answers as the engine reads it, so a dataset with
no `ASSET_CLASS` column answers `Equity`. A documented descriptive column the
dataset lacks (`sector`, `industry`, `sub_industry`, `region`, `country`,
`country_domicile`, `country_listing`) answers null, and any other unknown
column is refused with 422 `INVALID_RULE`. A row is `reference: {...}`, or
null when the name had no reference row on its date; a map of weights (a
rebalance, the frontier) carries `reference: {identifier: {...}}` beside it.
`GET /data/fields` lists the columns, each with a `label` to show.

On a long history, ask for only the columns a view needs: the default four
add about 46 KB to each rebalance of 500 names.

## Stored strategies

A stored strategy is what a backtest holds and how it trades, kept beside
the index definitions:

```json
{
  "id": "tracker",
  "name": "Sampled tracker",
  "description": null,
  "index_id": "TECH10",
  "strategy": {"type": "index_tracking",
               "replication": {"type": "SampledReplication", "params": {"holdings": 5}}},
  "implementation": {"cash_buffer": 0.01}
}
```

`index_id` is the stored index the strategy holds: the one tracked, or an
active strategy's benchmark. `strategy` and `implementation` take the shapes a
[backtest request](#backtest-requests) gives them, and a strategy that only
holds its index as defined is `{"type": "index"}`, the default. Everything
else a backtest needs, such as the period, the capital, the flows and the
vehicle, belongs to the backtest, and a strategy carrying one of those is
refused.

`POST /strategies` saves a new strategy, refusing an id already taken with
409 `CONFLICT`, and `PUT /strategies/{id}` saves one under its id, replacing
any stored there. Both check the strategy as a backtest checks its settings
and refuse it with 422 `INVALID_RULE`, a finding for each problem at its
path, such as `strategy.replication.type` or `index_id` for an index that is
not stored. `POST /strategies/validate` answers the same findings without
saving.

`GET /strategies` and `GET /strategies/{id}` give each strategy a `problem`
when it cannot be run as it stands: `INDEX_NOT_FOUND` when its index is not
stored, such as one removed on another machine, and `INDEX_UNREADABLE` when
the index is stored but cannot be read. It is null otherwise.

An index a strategy holds cannot be deleted: `DELETE /indices/{id}` answers
409 `IN_USE`, with the strategies in `detail.used_by`. The same goes for an
index whose optimised child a strategy holds, since the delete would take the
child with it. Deleting a strategy removes its runs with it, and the response
says how many went.

An optimisation run (`POST /optimise/runs`) solves for a target's weights on a
rebalance date, and the target is an index or a strategy, exactly one of
`index_id` and `strategy_id`. An index's weights are calculated, so it needs
no backtest. A strategy's are what its latest run chose at that rebalance:
the index's for a strategy that holds its index, the replication's for a
tracker, the construction's for an active strategy. Each row of the result
gives the `target_weight`, and the result names its target and
`target_date`. A risk model without `identifiers` takes its names from an
`index_id`, calculated, or from what a `strategy_id`'s latest run held. A
built-in report template, such as the factsheet, is generated from a
`strategy_id`'s latest run and named after the strategy.

## Jobs

Work that takes more than a moment runs as a job: the request answers
**202 Accepted** at once with the job's state, and the work carries on in the
engine.

| Job kind | Started by | Result |
| --- | --- | --- |
| `backtest:{strategy_id}` | `POST /strategies/{strategy_id}/backtest` | The backtest run. |
| `optimise:{run_id}` | `POST /optimise/runs` | The solved portfolio. |
| `risk:{model_id}` | `POST /risk-models/{model_id}/estimate` | The estimated model. |
| `render:{render_id}` | `POST /reports/render` | Where the PDF is; download it from `GET /reports/renders/{render_id}`. |
| `load:{store_id}` | `POST /data/stores/{id}/activate`, or an import | The store now served. |
| `generate:{store_id}` | `POST /data/synthetic` | The new store, and whether it is being served. |
| `refresh:{store_id}` | `POST /data/stores/{id}/refresh` | What the refresh did. |

Everything that can be checked is checked before the job starts, so a bad
request fails with an ordinary error rather than as a job that fails a moment
later.

A job's state has the same shape everywhere: in the 202 answer, from
`GET /jobs/{job_id}`, and on the event socket.

| Field | Meaning |
| --- | --- |
| `job_id` | The id to poll. |
| `kind` | One of the kinds above. |
| `status` | `pending`, `running`, `succeeded`, `failed` or `cancelled`. The last three are final. |
| `progress` | 0.0 to 1.0. |
| `message` | What the job is doing now. |
| `result` | The result, only once the job has succeeded; null otherwise. |
| `error` | When the job failed, the same `{code, message, detail}` an HTTP error carries (see [Errors](#errors)); null otherwise. |

Poll `GET /jobs/{job_id}` until `status` is final, or watch the event socket.
`GET /jobs` lists every job. `DELETE /jobs/{job_id}` cancels a job that is
still running; it answers with the job's state either way, so cancelling one
that has just finished is not an error. A job from before a restart cannot be
cancelled and answers 404.

Finished jobs are saved under `--documents`, so `GET /jobs/{job_id}` still
answers after the engine restarts. A job still running when the engine
stops is lost.

Some endpoints read their answers from saved job results: a strategy's
overview, weights, attribution, asset and compare views (from its latest
backtest), `/optimise/runs/{run_id}/...`, and `/risk-models`. An optimisation
run or a risk model aimed at a strategy reads its latest backtest too. The
result each of these reads is always kept: the latest successful backtest of
each strategy, every successful optimisation run, and the latest estimate of
each risk model. A backtest's result goes when its strategy is deleted or a
newer backtest replaces it.

Of the other finished jobs, the engine keeps the 50 most recent and deletes
older ones. `/strategies/{id}/record` and `/strategies/runs` read a record
kept per strategy, which the limit does not touch.

## Events

`/ws?token=<token>` is a WebSocket that sends a JSON message for every event,
to every client connected at the time. It sends nothing back and takes
nothing in; events are not replayed, so a client that connects late should
read the current state over HTTP.

Each message has a `type`:

| `type` | When | Carries |
| --- | --- | --- |
| `job` | A job starts, reports progress, or finishes | The job's state, as in the table above. |
| `data.loaded` | The engine starts serving a store: after activation, generation, import, or a refresh of the store being served | `store` (`id` and `name`) and the new `data_version`. |
| `data.freshness` | The store being served was refreshed, or features were imported | `dataset` (`market` for a refresh, `features` for an import) and `detail`, holding `store` (null for data not registered as a store), `rows_added` (rows downloaded or imported, or null) and the new `data_version`. |

A client that falls behind loses the oldest undelivered messages rather than
slowing the engine: each client's queue holds 100. Progress frames are
snapshots, so a lost one costs nothing, and `GET /jobs/{job_id}` always has
the current state.

### Data versions

`data_version` is an opaque token that changes whenever the data being
served changes: at startup, on every store load (the same store loaded again
included), when a refresh changes the store being served, and on every
feature import. A new value is
never one used before, even across restarts.

Compare it only for equality. `/health` reports the current value, and the
`data.loaded` and `data.freshness` events carry the new one. If it differs
from the value a client cached against, anything the client derived from the
data (name lists, coverage, previews) is stale.

This example registers a small store, activates it while listening on the
socket, and reads the finished job:

```python
from beacon.data import store
from beacon.testing import dataset

folder = store.save(dataset.data_fetcher(), Path("sample-store").resolve())

with TestClient(create_app(config)) as client:
    registered = client.post("/data/stores",
                             headers=auth,
                             json={"name": "Sample", "path": str(folder)})
    store_id = registered.json()["id"]

    with client.websocket_connect("/ws?token=dev") as socket:
        accepted = client.post(f"/data/stores/{store_id}/activate", headers=auth)
        assert accepted.status_code == 202

        while True:
            event = socket.receive_json()
            print(event["type"], event.get("status"), event.get("data_version"))

            if event["type"] == "job" and event["status"] != "running":
                break

    job = client.get(f"/jobs/{accepted.json()['job_id']}", headers=auth).json()
    print(job["status"], job["result"])

    health = client.get("/health", headers=auth).json()
    print(health["data_source"]["store_name"], health["data_source"]["data_version"])
```

The `with` block keeps the test client's event loop running between
requests, which a job needs. A real client makes the same calls over HTTP.

## Errors

Every error answer, whatever its status, has one shape:

```json
{
  "error": {
    "code": "NO_DATA_LOADED",
    "message": "No data is loaded, so this data cannot be read. Load a data store first.",
    "detail": {"purpose": "this data cannot be read"}
  }
}
```

`code` is stable and is what a client branches on. `message` is for people
and may change. `detail` is null or an object with structured context; its
contents depend on the code.

| Status | `code` | Meaning |
| --- | --- | --- |
| 400 | `BAD_REQUEST` | The request could not be read at all. |
| 401 | `UNAUTHORIZED` | The bearer token is missing or wrong. |
| 404 | `NOT_FOUND` | No such route. |
| 404 | `DATA_NOT_FOUND` | The thing asked for does not exist: an index, a job, an instrument's data, a store. |
| 405 | `METHOD_NOT_ALLOWED` | The path exists but not with this method. See [Wrong methods](#wrong-methods). |
| 409 | `NO_DATA_LOADED` | The request needs data and none is loaded. Load a store, then retry. |
| 409 | `STORE_FORMAT_OUTDATED` | The data store was saved in a format this version no longer reads, such as a store saved before 0.7.0 with exchange rates in a `RATE` column. Regenerate or re-import it. `GET /data/stores` says the same in each store's `problem`. |
| 409 | `IN_USE` | A delete refused while other documents use the one named, such as an index a stored strategy holds. `detail.used_by` lists them. |
| 409 | `CONFLICT` | The engine's state refuses the request: an id already taken, a store already loading or refreshing, forgetting the store being served, a store with nothing to refresh, editing the read-only `GLOBAL` universe, a report template with no blocks. |
| 422 | `VALIDATION_ERROR` | The request does not match its schema. `detail.errors` lists each problem with its location (`loc`), message (`msg`) and `type`. |
| 422 | `INVALID_RULE` | A definition or document was refused. When there are several problems, `detail.findings` lists each one with `path`, `rule_id`, `severity`, `code` and `message`. |
| 422 | `INVALID_IDENTIFIER` | An id that cannot be used, for example one containing `/`, or a name reserved for an endpoint. |
| 422 | `INVALID_EXPRESSION` | A malformed expression. |
| 422 | `INVALID_ARGUMENT` | A request that cannot be answered as put, such as an end date before the start, an unknown price interval, an adjusted series without `CLOSE`, more than 1,000 identifiers at once, or a futures expiry before its valuation date. |
| 422 | `FROZEN_PORTFOLIO` | An attempt to change a finished backtest's books. |
| 500 | `CALCULATION_ERROR` | A calculation refused deliberately; the message says what to change. |
| 500 | `UNEXPECTED_CALCULATION_FAILURE` | Something failed that should not have. `detail.original_type` names the exception, so a crash can be told from a refusal. |
| 500 | `CONFIGURATION_ERROR`, `REPORTING_ERROR`, `NO_DATA_SOURCE`, `BEACON_ERROR` | Other failures inside the engine. |
| 501 | `NOT_IMPLEMENTED` | The endpoint exists but does not work yet. |
| 503 | `MISSING_DEPENDENCY` | An optional package the request needs is not installed. The message names the extra to install. |

Another status without its own code carries `HTTP_ERROR`.

A failed job's `error` uses the same codes, so one renderer serves both. A
failed job saved without a code carries `UNCLASSIFIED_FAILURE`.

The findings in `detail.findings` have codes of their own, such as
`NOT_A_DATA_STORE` or `UNKNOWN_RULE_TYPE`. They point at the field or rule
that caused each problem, and `POST /indices/validate` and
`POST /optimise/constraint-sets/validate` return the same findings without
saving anything.

```python
client = TestClient(create_app(config))  # a new engine, serving no data

answer = client.get("/data/prices/CMPA", headers=auth)
assert answer.status_code == 409
print(answer.json()["error"]["code"])

answer = client.post("/data/stores",
                     headers=auth,
                     json={"name": "Nowhere", "path": "relative/folder"})
assert answer.status_code == 422
for finding in answer.json()["error"]["detail"]["findings"]:
    print(finding["code"], finding["message"])
```

### Wrong methods

A method a path does not support answers 405 `METHOD_NOT_ALLOWED`, with an
`Allow` header listing the methods it does support. This holds even where a
fixed path sits beside a parameterised one: `PUT /indices/validate` is a
405, not an attempt to save an index called `validate`.

```python
answer = client.put("/indices/validate", headers=auth, json={})
assert answer.status_code == 405
print(answer.headers["Allow"], answer.json()["error"]["message"])
```

## Listings that skip what they cannot read

A listing of saved documents leaves out any document it cannot read rather
than failing as a whole, and says how many it left out. This applies to
`GET /indices`, `/universes`, `/data/watchlists`,
`/strategies`, `/strategies/runs`, `/optimise/constraint-sets`, `/reports/templates`,
`/jobs` and `/data/stores`.

Each carries `skipped`, the number left out, and `skipped_causes`, the same
number split by cause:

| Cause | Meaning | What to do |
| --- | --- | --- |
| `unparseable` | The file is damaged: not valid JSON, or a version nothing can migrate. | Restore it from a backup or remove it. |
| `from_newer_build` | A newer py-beacon wrote it. Nothing is wrong with it. | Upgrade the engine. |
| `unrecognised` | Valid JSON that this engine's model does not accept. | The server log names the field. |

A non-zero `skipped` means the listing is incomplete. Each skipped document
answers 404 on its own route.

## Release notes

`GET /changelog` returns the notes for the engine that is running: its
`version` and `entries`, newest first. Each entry has a `version`, a `date`
(null for unreleased changes) and `sections`, each a `heading` such as Added
or Fixed with a list of `items` in Markdown. Entries with no changes are left
out.

`?since=<version>` returns only the releases after that version, so a client
can show what is new since the version it last showed.

```python
notes = client.get("/changelog", headers=auth, params={"since": "0.1.0"}).json()
print(notes["version"], [entry["version"] for entry in notes["entries"]])
```

## CORS

A browser page can call the engine only from an allowed origin. The defaults
are `beacon://app` and `app://`, and `http://localhost`, `http://127.0.0.1`
and `http://[::1]` on any port are always allowed. `--cors-origin` (repeatable) or `BEACON_CORS_ORIGINS`
(comma-separated) replace the defaults. Requests from a desktop app's main
process are not browser requests, so CORS does not apply to them.
