# 5. Gzipped CSV for the persisted data store

Date: 2026-08-03 (BN-113)

## Status

Accepted.

## Context

`python -m beacon.server` — the command the desktop client spawns — always
started without data. `ServerConfig` had accepted a `data_fetcher` since the
server was built, but nothing ever passed one, so every data endpoint answered
`CONFIGURATION_ERROR` and sync could not bootstrap: it needs a fetcher to merge
*into*. Giving the launcher a store to find means choosing a format for it.

The store holds three datasets — market data (long-form OHLCV plus shares
outstanding and free float), reference data with validity windows, and a
corporate-action history. At the scale BN-114 generates, that is roughly
645,000 market rows: 512 identifiers over five years of business days.

## Decision

A directory of gzipped CSV files with a JSON manifest:

```
market-store/
  manifest.json            schema_version, source, datasets present
  market.csv.gz
  reference.csv.gz         omitted when the fetcher has none
  corporate_actions.csv.gz omitted when the history is empty
```

Three writer settings are pinned so the same data always produces the same
bytes: `lineterminator="\n"`, gzip `mtime=0`, and sorted manifest keys.

## Alternatives considered

**Parquet.** Faster to read, roughly a third the size, and dtype-preserving so
no date parsing is needed on load. Rejected because it puts pyarrow (~40MB of
compiled wheel) in front of the one thing that must work before anything else
does. The store is read once at startup and 645k rows take about a second and a
half through `read_csv` — not the difference between a usable desktop
application and an unusable one. It is the strongest alternative and the reason
`schema_version` exists: if load time becomes the complaint, the format can
change without stranding anyone's store.

**A single SQLite file.** Queryable without loading everything, and atomic to
write. Rejected because nothing in Beacon queries the store — `DataFetcher`
loads the whole panel into memory and answers from there. The benefit is one
Beacon does not currently collect, and it would put a query dialect between the
data and the containers.

**Uncompressed CSV.** Simpler still, and diffable in git. Rejected on size: 55MB
against 15MB, for a file nobody diffs.

**Reusing `DocumentStore`.** It already versions and migrates JSON on the same
app-data root. Rejected because it is built for small hand-authored documents;
645k rows of JSON would be several hundred megabytes and slower to parse than
the CSV it replaced. The two stores share the app-data root and the
`schema_version` convention, which is the part worth sharing.

## Consequences

- The store opens in a text editor after `gunzip`, so "did the server start
  with data" is answerable by looking rather than by instrumenting a load.
- No new dependency. `platformdirs` is needed only for `default_path()`, and
  only because the *default location* is a platform question — passing an
  explicit path anywhere in the module needs nothing beyond pandas.
- Byte-for-byte reproducibility is testable, which is what BN-114's "same seed
  produces the same store" guarantee rests on. Without the three pinned
  settings that guarantee would hold only within one run on one machine — the
  same class of platform dependence that made BN-95's fixture hash fail on six
  of nine CI cells.
- Dtypes are not preserved: everything round-trips through text, so the loaders
  re-parse dates and floats. Float values survive exactly because pandas writes
  shortest-round-trip repr, but an integer column returns as int64 only if
  every value still parses as one.
- No atomic write. A store written while the server is reading it is a torn
  read. Acceptable today because the writer is a CLI run by a human between
  server starts; if the server ever writes its own store this needs a
  write-to-temp-and-rename, as `DocumentStore` already does.
