# Serving data to a client

How a desktop client gets numbers out of Beacon, end to end. Written for
whoever is on the other side of the process boundary.

## The short version

```bash
python -m beacon.synthetic --seed 42   # 6,000 names, 10 years, ~18s
python -m beacon.server --port 0 --token dev        # finds it by itself
```

The second command takes no data argument. That is the point: the generator's
default output location and the server's third resolution branch are the same
directory, so the two agree without either knowing about the other. A client
that already spawns `python -m beacon.server --port 0` needs no change.

## The client never reads the data

It talks HTTP to a server it launches. It does not open a file, does not see a
CSV, and does not know the store is gzipped — so the format can change without
anything on the client side changing with it.

That is deliberate. The data is not the product; the *calculations* are. Index
levels, backtests, attribution and risk contributions are all pandas and numpy.
A client reading the store directly would have to reimplement the index maths,
and then there would be two implementations to keep agreeing.

## The startup handshake

1. **The client spawns the server.**

   ```
   python -m beacon.server --port 0 --token <secret>
   ```

   `--port 0` asks the OS for any free port, so the client does not have to
   pick one and hope.

2. **The server announces where it landed.** The socket is bound *before*
   uvicorn starts, so the port is known early, and one line goes to stdout:

   ```
   BEACON_PORT=52612
   ```

   It is flushed immediately, because the client is blocked reading it — a
   buffered stdout would deadlock the handshake. Every later stdout line is
   ordinary logging.

3. **The server loads the store into memory, once.** The files are not read
   again.

4. **The client makes ordinary requests.**

   ```
   GET http://127.0.0.1:52612/data/prices/CMPA
   Authorization: Bearer <secret>
   ```

   The same token that was passed on the command line. That is how the server
   knows the request came from the process that started it rather than from
   anything else on the machine.

The server binds loopback only and lives and dies with the client.

## Where the server looks for data

In order, stopping at the first that answers:

| | Source | Behaviour if unreadable |
| --- | --- | --- |
| 1 | `--data <path>` | **Exit 2.** Asking for a store that cannot be read is a mistake worth stopping for. |
| 2 | `$BEACON_DATA_PATH` | **Exit 2**, same reasoning. |
| 3 | The app-data store | **Warn and continue.** A corrupt store must not leave the client unable to start the server that would replace it. |
| 4 | Nothing | The server starts data-less, as it always did. |

The branch that ran is logged immediately after the port announcement:

```
INFO:__main__:Data source: the app-data store (…/beacon/beacon/market-store).
```

So "why is the client empty" is answered by reading the log rather than by
guessing.

The app-data directory is the platform convention — on Windows,
`%LOCALAPPDATA%\beacon\beacon\market-store`. `beacon.data.store.default_path()`
returns it.

## When there is no data at all

The server **still starts**. `/health` reports `configured: false` and the data
endpoints return `CONFIGURATION_ERROR`, which is what a client should branch
on. Refusing to start would leave the client unable to run the sync that would
populate it — the failure would prevent its own cure.

## Generating a store

`python -m beacon.synthetic` produces market-like data: a factor model with
GJR-GARCH volatility, Student-t innovations and negative skew, plus reference
data, shares outstanding, free float, dividends and splits that agree with the
prices.

It also generates **features** — four fundamental ratios (`pe_ratio`,
`pb_ratio`, `eps`, `debt_to_equity`) quarterly, and two alternative series
(`x_sentiment`, `wikipedia_views`) monthly. The ratios are derived from the
price path rather than drawn beside it, so `pe_ratio x eps` is the close at
the period end, exactly. Announcement lags vary per name per quarter and
coverage is deliberately incomplete, so a point-in-time read has a ragged edge
to resolve against.

| Flag | Default |
| --- | --- |
| `--assets` | 6,000 |
| `--start` / `--end` | The ten years ending **today** |
| `--seed` | 42 |
| `--out` | The app-data store the server auto-loads |
| `--extended-universe` | Off; widens the universe to 10,000 names |
| `--long-history` | Off; reaches back past every crisis the generator models |
| `--no-features` | Off; skips the ratios and alternative data (~8% of rows) |

Both expansion flags widen a default rather than overruling a value you named:
an explicit `--assets` beats `--extended-universe`, and an explicit `--start`
beats `--long-history`.

`--long-history` is anchored to the crisis dates rather than to a round number
of years, because those dates are fixed while a rolling window moves — "25
years back from today" already clipped the start of the dot-com unwind, and by
2030 would have missed it entirely.

Measured peak memory and wall clock, which matter at these sizes:

| Run | Rows | Peak | Time |
| --- | --- | --- | --- |
| default (6,000 × 10y) | 11.8M | 2.5 GB | 19s |
| `--extended-universe` | 26.1M | 3.8 GB | 29s |
| `--long-history` | 32.6M | 4.8 GB | 42s |
| both | ~69M | ~10 GB | ~85s |

The default holds **fewer** rows than the 5,000-name panel that preceded it
(13.0M), despite carrying a thousand more names. Delisted instruments have
their rows removed rather than carried as nulls, so a panel with turnover in
it is smaller than the full grid its universe size implies. The three
expansion figures below predate that change and are therefore upper bounds.

The CLI prints its own estimate before it starts and warns above 4 GB.

The dates default to today deliberately: data ending eighteen months ago reads
as stale in every freshness indicator, which is a true statement about the data
and a misleading one about the application. Pass both dates explicitly when you
want reproducibility across days — the seed fixes the draw, not the calendar.

Same seed and same dates produce byte-identical files, so re-running costs
nothing.

Nothing generated resembles a real company: names are `Company A` … and every
ticker carries a `CMP` prefix, which makes a collision with a real listing
impossible rather than merely improbable.

## A worked example

Spawned with no arguments beyond the port and token, against a store generated
a moment earlier:

```
GET /health
{"status": "ok", "data_source": {"configured": true, "identifiers": 512},
 "cache_age": 0.31}

GET /data/coverage
  market             configured=true  ids=512  source=synthetic  freq=daily
  reference          configured=true  ids=512  source=synthetic  freq=static
  corporate_actions  configured=true  ids=382  source=synthetic  freq=event
  identifiers_union=512   cache_size_bytes=14448159

GET /data/prices/CMPA
  1305 rows: OPEN, HIGH, LOW, CLOSE, VOLUME, SHARES_OUTSTANDING, FREE_FLOAT

GET /data/reference?identifiers=CMPA,CMPB,CMPC&fields=NAME,SECTOR,adv_3m
  three entries in one request, each with NAME, SECTOR and a server-computed ADV

GET /data/corporate-actions/CMPA
  {"ex_date": "2026-05-15T00:00:00", "type": "DIVIDEND", "kind": "cash",
   "value": 0.7514, "pay_date": "2026-06-05", "status": "paid"}
```

## Adjusted prices

`GET /data/prices/{identifier}?adjusted=true` adds an `ADJ_CLOSE` column:
`CLOSE` back-adjusted for **splits and dividends**, the vendor convention.

Two things follow from that, and both matter to what you render:

**An adjusted series is not a price.** It answers "what would a holder have
made", so its level is not what anything traded at that day. A chart of it is
a total-return chart, and labelling it as a price is the mistake the name
invites.

**It is adjusted backwards**, so the last value equals the last raw close and
only history moves. That makes the right-hand edge checkable against any other
source — and it means the whole series shifts when a new action lands, so a
cached adjusted series is not immutable.

Computed per request rather than stored, for that last reason: a stored column
would be wrong from the next dividend onwards, and silently.

## Browsing a whole dataset

`GET /data/tables/{dataset}?offset=&limit=` over `market`, `reference`,
`corporate_actions` and `features`, returning the same `{index, columns, data}`
frame shape as everything else plus a `total`.

Paged, with a maximum of 1,000 rows: the default store is 11.8M market rows.
Ordering is stable, `offset` past the end is an empty page rather than a 404,
and there is deliberately no filtering or sorting — a client that needs those
wants the expression API.

## Notes for a client

**Branch on `kind`, not on `type`.** `kind` is `cash`, `ratio` or `structural`
and is the authoritative answer to what `value` means. Matching type strings
works until a type the list has never seen arrives, at which point it renders
as whichever the list defaults to — confidently, and wrongly.

**Derive staleness from `frequency` and `stale_after_seconds`.** Both travel
with each dataset in `/data/coverage`, so a client holding its own 24h/7d
thresholds is guessing at a property of the data, and the guess diverges from
the engine the moment either changes.

**Read `identifiers_union`, not a sum.** Per-dataset counts overlap; adding
them reports more assets covered than exist. The `fx` row is the clearest
case: currency pairs are market identifiers, so they are counted in `market`
too, and `fx` exists to answer "do we hold exchange rates" rather than to add
to a total. It reports a null `cache_size_bytes` for the same reason — the
rows live in the market file.

**A currency pair is an ordinary identifier.** `EURUSD` answers on
`/data/prices` like anything else, with the rate in `CLOSE`, and appears in
`/data/identifiers`. It carries no reference data, and `RATE` is populated on
a pair and null on every instrument — which is how to tell them apart if you
need to.

**Per-dataset `cache_size_bytes` is that dataset's file.** The store total is
reported once at the top level, and is larger than the sum of the parts because
the manifest is part of the store and not a dataset.

**`null` means unknown, not zero.** A missing `pay_date`, an absent `adv_3m`,
an uncovered `risk_contribution` — all null rather than a placeholder, so a
client can omit the field rather than dash it. A dash reads as "there is none",
which is a different statement.

**CORS only matters if the renderer calls the server directly.** The default
allowed origins are `beacon://app` and `app://`, plus localhost on any port,
and `--cors-origin` overrides them. Route calls through a main process instead
and CORS never arises, because a main-process request is not a browser request.
