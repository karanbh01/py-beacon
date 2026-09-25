# The engine's HTTP API

The local engine (`python -m beacon.server`) answers HTTP on a loopback port.
This page explains how the API works as a whole: authentication, the endpoint
groups, jobs, live events and errors. The fields of each endpoint are in the
[API reference](https://pybeacon.dev/api/), generated from the engine's
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
`/ws?token=<token>`. A wrong or missing token is refused at the handshake,
which answers 403.

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
| `/data/identifiers`, `/data/prices`, `/data/reference`, `/data/corporate-actions`, `/data/features`, `/data/fields`, `/data/tables` | Reading the loaded data: search names, price history, reference fields (including derived ones such as market caps), corporate actions, features, the field catalogue, and raw pages of each dataset. `POST /data/features` adds feature rows to the served data. |
| `/data/coverage` | What each dataset holds, how old it is, and when it goes stale. |
| `/data/stores` | Registered data stores: list, register, rename, forget, activate, refresh. |
| `/data/synthetic` | Generate synthetic data into a new store, as a job. |
| `/data/import` | Import CSV files or an Excel workbook into a new store, and download the template. |
| `/data/watchlists` | Saved lists of identifiers. |
| `/universes` | Saved universes: the pools of names an index selects from, and their members on a date. A `GLOBAL` universe covering the loaded data is kept up to date automatically and cannot be edited. |
| `/indices` | Index definitions: save, validate, preview, the rebalance schedule, the rule types and calendars an editor can offer, and deriving an optimised index from a saved one. |
| `/beacon` | Running a backtest of a saved index (a job), and reading its results: the latest record, overview, weights, attribution, one asset, and a comparison of several indices. |
| `/optimise` | Constraint sets, and optimisation runs (a job) with their frontier and exposures. |
| `/risk-models` | Estimating a risk model (a job) and reading the estimated models. |
| `/reports` | Report templates, rendering a report to PDF (a job), and downloading the PDF. |
| `/derivatives` | Pricing index futures and total return swaps, and an index's futures term structure and roll. |
| `/jobs` | Polling and cancelling jobs. |
| `/ws` | The event socket. |

An endpoint that needs data answers 409 `NO_DATA_LOADED` when none is
loaded. The rest, such as listing saved indices or reading `/health`, work
without data.

## Jobs

Work that takes more than a moment runs as a job: the request answers
**202 Accepted** at once with the job's state, and the work carries on in the
engine.

| Job kind | Started by | Result |
| --- | --- | --- |
| `backtest:{index_id}` | `POST /beacon/{index_id}/backtest` | The backtest run. |
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
answers after the engine restarts. The engine keeps the 50 most recently
finished jobs of all kinds and deletes older ones. A job still running when
the engine stops is lost.

Some endpoints read their answers from saved job results, and so are subject
to the same limit of 50: an index's overview, weights, attribution, asset and
compare views (from its latest backtest), `/optimise/runs/{run_id}/...`, and
`/risk-models`. An optimisation run also needs a saved backtest of its index.
`/beacon/{index_id}/record` and `/beacon/backtests` read a record kept per
index, which the limit does not touch.

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
| `data.freshness` | The store being served was refreshed | `dataset` (`market`) and `detail`, holding `store`, `rows_added` (rows downloaded, or null) and the new `data_version`. |

A client that falls behind loses the oldest undelivered messages rather than
slowing the engine: each client's queue holds 100. Progress frames are
snapshots, so a lost one costs nothing, and `GET /jobs/{job_id}` always has
the current state.

### Data versions

`data_version` is an opaque token that changes whenever the data being
served changes: at startup, on every store load (the same store loaded again
included), and when a refresh changes the store being served. A new value is
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
| 404 | `DATA_NOT_FOUND` | The thing asked for does not exist: an index, a job, an instrument's data, a store. Some requests the engine cannot answer, such as an unknown price interval, also answer this. |
| 405 | `METHOD_NOT_ALLOWED` | The path exists but not with this method. See [Wrong methods](#wrong-methods). |
| 409 | `NO_DATA_LOADED` | The request needs data and none is loaded. Load a store, then retry. |
| 409 | `CONFLICT` | The engine's state refuses the request: an id already taken, a store already loading or refreshing, forgetting the store being served, a store with nothing to refresh, editing the read-only `GLOBAL` universe, a report template with no blocks. |
| 422 | `VALIDATION_ERROR` | The request does not match its schema. `detail.errors` lists each problem with its location (`loc`), message (`msg`) and `type`. |
| 422 | `INVALID_RULE` | A definition or document was refused. When there are several problems, `detail.findings` lists each one with `path`, `rule_id`, `severity`, `code` and `message`. |
| 422 | `INVALID_IDENTIFIER` | An id that cannot be used, for example one containing `/`, or a name reserved for an endpoint. |
| 422 | `INVALID_EXPRESSION` | A malformed expression. |
| 422 | `INVALID_ARGUMENT` | An argument the library refused, such as an end date before the start. |
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
`/optimise/constraint-sets`, `/reports/templates`, `/beacon/backtests`,
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
are `beacon://app` and `app://`, and `http://localhost` on any port is always
allowed. `--cors-origin` (repeatable) or `BEACON_CORS_ORIGINS`
(comma-separated) replace the defaults. Requests from a desktop app's main
process are not browser requests, so CORS does not apply to them.
