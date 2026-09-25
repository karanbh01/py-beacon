# 2. Data freshness is tracked; `cache_age` reports a real number

**Status:** Accepted
**Date:** 2026-07-30
**Issue:** [BN-99] (#118), amending [BN-66] (#82)

## Context

BN-66 asked for coverage that "reflects real cache ages". Nothing in the
library cached anything: `DataFetcher` read from `MarketData` and
`ReferenceData` that were loaded into memory once at startup and never changed
after. So `/health` and `/data/coverage` both reported `cache_age: null`, that
criterion on #82 could not be met, and the WebSocket `data.freshness` event had
nothing real to announce.

BN-99 named the two ways out and warned against drifting into the second by
default:

1. Track when each dataset was last refreshed, and report it.
2. Drop the criterion — keep identifier counts and date spans, leave
   `cache_age` null, and document that as the honest answer.

Option 2 was explicitly defensible *while no ingestion path existed*. If data
can only arrive at startup, its age is the process's uptime, which the client
already knows and does not need an endpoint for.

## Decision

**Option 1.** BN-100 landed an ingestion path immediately before this, which
removes the argument for option 2: data can now arrive after startup, at a
moment nobody outside the server knows, so its age becomes a real question that
only the server can answer.

The sequencing was on purpose. BN-99 was taken *after* BN-100 rather than in
issue order, so the decision was made with an ingestion path in hand instead of
being reasoned about hypothetically.

`DataFetcher` records a refresh timestamp per dataset. Loading stamps one,
every merge from a sync replaces it, and `/health` and `/data/coverage` read
real ages off it.

## Consequences

- **Loading counts as a refresh.** A freshly started server holds data that is
  genuinely seconds old. Reporting "unknown" until someone happened to run a
  sync would be less true, not more careful.
- **Null now means "not loaded", and only that.** `cache_age` is null when a
  dataset is absent — no reference data configured, or no data source at all.
  That is a different statement from "loaded and never refreshed", and
  collapsing the two would put the ambiguity back that this decision removes.
- **Coverage carries the timestamp as well as the age.** An age is only true at
  the instant it was read; a client holding a response for a minute needs the
  timestamp to work out what it now has.
- **A backwards clock reports zero, not a negative age.** A system clock
  adjustment between the two readings is noise, not information about the data.
- **BN-66's criterion on #82 is amended, not dropped.** The test that asserted
  `cache_age is None` was correct when nothing tracked refreshes; it now
  asserts a real age, with the old reasoning recorded in its docstring.
- This is freshness tracking, **not a cache**. Nothing is evicted, nothing has
  a TTL, and no request is served from a different store than before. The name
  `cache_age` is kept because it is the client's existing contract, but what it
  measures is "how old is the data I hold", which is the question anyone
  actually asks.

## What this does not do

There is still no expiry, no background refresh, and no staleness threshold
above which the server refuses to answer. Those need a policy decision about
what "too old" means for a given dataset, and that is a product question rather
than a plumbing one. The timestamps are the prerequisite for any of it.
