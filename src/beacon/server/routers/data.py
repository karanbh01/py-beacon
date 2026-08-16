# src/beacon/server/routers/data.py
"""
Data router: prices and reference data.

Two endpoints from BN-65 are deliberately absent.

`/data/fundamentals` is dropped rather than deferred: fundamentals will be
served by a general *features* endpoint covering any per-instrument datapoint
that is neither reference data nor a corporate action and that can drive a
backtest or an index rule. That endpoint is not designed yet.

`/data/corporate-actions` is served since BN-98 added a history to the data
layer. It returns the raw actions plus the aggregates that need the whole
series — a trailing dividend, its yield, and the compounded split ratio — so a
client does not reimplement the trailing window and get its boundary wrong.
"""
from typing import Annotated

import pandas as pd

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...data.identifiers import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    IdentifierIndex,
    fingerprint,
)
from ...exceptions import ConfigurationError, DataNotFoundError
from ..config import ServerConfig
from ..reference import MAX_BATCH, build_entries, parse_identifiers, parse_list
from ..schemas import (
    SOURCE_USER,
    BatchReferenceResponse,
    CorporateActionsResponse,
    IdentifierMatch,
    IdentifierSearchResponse,
    PricesResponse,
    ReferenceResponse,
    UniverseMembership,
)

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request, Response  # noqa: E402

# Resolutions this router can honestly serve. The stored data is the native
# frequency; the other two are derived by taking each period's last
# observation, which is the standard convention for a price series.
RESAMPLE_RULES = {"weekly": "W", "monthly": "ME"}
NATIVE_INTERVAL = "native"
SUPPORTED_INTERVALS = (NATIVE_INTERVAL, *RESAMPLE_RULES)

# Query parameters declared as Annotated aliases rather than as call defaults:
# FastAPI's `param = Query(...)` form puts a function call in a default, which
# is the pattern B008 exists to catch. Annotated is also the form FastAPI now
# recommends.
StartQuery = Annotated[str | None, Query(description="Inclusive start date, YYYY-MM-DD.")]
EndQuery = Annotated[str | None, Query(description="Inclusive end date, YYYY-MM-DD.")]
IntervalQuery = Annotated[str, Query(description="native, weekly or monthly.")]
ColumnsQuery = Annotated[
    list[str] | None,
    Query(description="Market-data columns to return; all by default.")]
AsOfQuery = Annotated[
    str | None,
    Query(description="Point-in-time date, YYYY-MM-DD. Returns only rows valid then.")]
TypesQuery = Annotated[
    list[str] | None,
    Query(description="Restrict to these action types, e.g. DIVIDEND, SPLIT.")]
QueryQuery = Annotated[
    str | None,
    Query(alias="q",
          description="Fragment to match against identifier and name. Absent "
                      "or empty enumerates instead of searching.")]
LimitQuery = Annotated[
    int,
    Query(ge=1, le=MAX_LIMIT,
          description=f"Maximum rows, at most {MAX_LIMIT}.")]
OffsetQuery = Annotated[
    int, Query(ge=0, description="Rows to skip, for walking an enumeration.")]
DatasetsQuery = Annotated[
    list[str] | None,
    Query(description="Only return identifiers covered by all of these, e.g. "
                      "'market'. Comma-separated or repeated.")]
IdentifiersQuery = Annotated[
    list[str] | None,
    Query(description="Identifiers to look up. Repeat the parameter or "
                      f"comma-separate; at most {MAX_BATCH} per call.")]
FieldsQuery = Annotated[
    list[str] | None,
    Query(description="Reference columns to return, plus derived fields such "
                      "as adv_3m. All stored columns and no derived field by "
                      "default.")]


def _data_fetcher(request: Request) -> DataFetcher:
    """Return the process's data source, or fail with a mapped error.

    Args:
        request: The incoming request.

    Returns:
        DataFetcher: The configured source.

    Raises:
        ConfigurationError: When the process was started without one. This
            maps to 500 rather than 404 — the request was fine, the server is
            not configured to answer it.
    """
    # app.state is untyped, so pin the config back to its real type before
    # reading through it.
    config: ServerConfig = request.app.state.config

    fetcher = config.data_fetcher
    if fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so data endpoints "
            "cannot be served. Restart it with one configured.")

    return fetcher


def _resample(frame: pd.DataFrame,
              interval: str) -> pd.DataFrame:
    """Reduce a date-indexed frame to a coarser interval.

    Args:
        frame: Date-indexed market data.
        interval: One of SUPPORTED_INTERVALS.

    Returns:
        pd.DataFrame: The frame at the requested interval. Returned unchanged
        for the native interval.
    """
    if interval == NATIVE_INTERVAL:
        return frame

    return frame.resample(RESAMPLE_RULES[interval]).last().dropna(how="all")


def _identifier_index(request: Request,
                      fetcher: DataFetcher | None) -> IdentifierIndex:
    """The identifier index for this process, rebuilt only when data changes.

    Cached on app state rather than rebuilt per request: this endpoint is
    called on every keystroke, and pulling names out of pandas each time would
    be doing the expensive part repeatedly for an answer that only moves when a
    sync moves it.

    The cache key is the index's own version, which is a fingerprint of the
    fetcher's refresh timestamps. A sync changes those, the fingerprint stops
    matching, and the next request rebuilds — so a client's suggestions refresh
    with the freshness event it already listens for, and no new invalidation
    mechanism is needed.
    """
    if fetcher is None:
        return IdentifierIndex.empty()

    cached: IdentifierIndex | None = getattr(request.app.state,
                                             "identifier_index", None)
    current = fingerprint(fetcher)

    if cached is not None and cached.version == current:
        return cached

    index = IdentifierIndex.build(fetcher)
    request.app.state.identifier_index = index

    return index



def _memberships(request: Request,
                 identifier: str) -> list[UniverseMembership]:
    """Which universes contain an instrument.

    Answered by scanning the stored universes rather than by an index, which
    is the right shape at this size: a workspace holds a handful of documents,
    and an index would be a second structure to keep in step with them for no
    measurable gain. If universes ever number in the thousands this becomes a
    reverse map built once per request instead.

    Returns an empty list when the store is absent, so an endpoint that could
    always answer this question keeps working on a server configured without
    document storage.
    """
    store = getattr(request.app.state, "universe_store", None)

    if store is None:
        return []

    memberships = [
        UniverseMembership(id=document["id"],
                           name=document.get("name", document["id"]),
                           source=document.get("source", SOURCE_USER))
        for document in store.read_all()
        if identifier in document.get("identifiers", ())
    ]

    # Sorted so the order does not depend on how the filesystem listed the
    # documents, which would make the response unstable between calls.
    return sorted(memberships, key=lambda membership: membership.id)


def build_data_router() -> APIRouter:
    """Build the /data router.

    Returns:
        APIRouter: Router carrying the prices and reference endpoints.
    """
    router = APIRouter(prefix="/data", tags=["data"])

    @router.get("/identifiers", response_model=IdentifierSearchResponse)
    def identifiers(request: Request,
                    response: Response,
                    q: QueryQuery = None,
                    limit: LimitQuery = DEFAULT_LIMIT,
                    offset: OffsetQuery = 0,
                    datasets: DatasetsQuery = None) -> IdentifierSearchResponse:
        # Deliberately does NOT go through `_data_fetcher`, which raises when
        # no source is configured. "Nothing matches" and "this engine is
        # misconfigured" render as very different things in a client, and an
        # empty suggestion list must not look like a broken install — so a
        # data-less server answers 200 with an empty list.
        config: ServerConfig = request.app.state.config
        index = _identifier_index(request, config.data_fetcher)

        found = index.search(query=q,
                             limit=limit,
                             offset=offset,
                             datasets=tuple(parse_list(datasets)))

        # The version moves only when a dataset syncs, so a client can cache an
        # enumeration and revalidate against this rather than refetching it.
        response.headers["ETag"] = f'"{index.version}"'

        return IdentifierSearchResponse(
            identifiers=[IdentifierMatch(identifier=entry.identifier,
                                         name=entry.name,
                                         datasets=list(entry.datasets),
                                         exchange=entry.exchange,
                                         currency=entry.currency)
                         for entry in found.entries],
            total=found.total,
            truncated=found.truncated,
            version=index.version)

    @router.get("/prices/{identifier}", response_model=PricesResponse)
    def prices(request: Request,
               identifier: str,
               start: StartQuery = None,
               end: EndQuery = None,
               interval: IntervalQuery = NATIVE_INTERVAL,
               columns: ColumnsQuery = None) -> PricesResponse:
        # No `adjusted` parameter: the library holds no price-adjustment
        # logic, so accepting one and returning the stored series regardless
        # would misrepresent the response rather than merely limit it.
        if interval not in SUPPORTED_INTERVALS:
            raise DataNotFoundError(
                f"interval '{interval}'",
                source=f"supported intervals are {', '.join(SUPPORTED_INTERVALS)}")

        frame = _data_fetcher(request).fetch_market_data(identifier, start, end, columns)

        if frame.empty:
            raise DataNotFoundError(f"market data for '{identifier}'",
                                    source="MarketData")

        return PricesResponse.from_frame(identifier, interval, _resample(frame, interval))

    # Declared before the single-name route. Both paths are distinct so the
    # order does not decide matching, but reading them in this order is what
    # makes the batch form the obvious default rather than an afterthought.
    @router.get("/reference", response_model=BatchReferenceResponse)
    def reference_batch(request: Request,
                        identifiers: IdentifiersQuery = None,
                        date: AsOfQuery = None,
                        fields: FieldsQuery = None) -> BatchReferenceResponse:
        # No 404 when nothing matches: a batch that found none of its
        # identifiers is a successful answer to a question about names this
        # dataset does not carry, and the per-entry `found` flag already says
        # so for each one.
        names = parse_identifiers(identifiers)
        entries = build_entries(_data_fetcher(request), names, date, fields)

        return BatchReferenceResponse(entries=entries, as_of=date)

    @router.get("/reference/{identifier}", response_model=ReferenceResponse)
    def reference(request: Request,
                  identifier: str,
                  date: AsOfQuery = None) -> ReferenceResponse:
        # GICS and profile remain in this endpoint's brief and out of the
        # data model, so nothing here invents them: whatever columns the
        # loaded reference data carries are returned as-is.
        #
        # Universe memberships *are* answerable now. They were not when this
        # endpoint was written -- no universe could be created through the
        # API and none was seeded, so the answer was always "none" and saying
        # so would have been noise. BN-132 changed that.
        frame = _data_fetcher(request).fetch_reference_data(identifier, date)

        if frame.empty:
            raise DataNotFoundError(f"reference data for '{identifier}'",
                                    source="ReferenceData")

        return ReferenceResponse.from_row(identifier, frame.iloc[0],
                                          _memberships(request, identifier))

    @router.get("/corporate-actions/{identifier}",
                response_model=CorporateActionsResponse)
    def corporate_actions(request: Request,
                          identifier: str,
                          start: StartQuery = None,
                          end: EndQuery = None,
                          types: TypesQuery = None) -> CorporateActionsResponse:
        # An instrument with no actions is not an error: plenty of companies
        # pay nothing, and a 404 would make "no dividends" indistinguishable
        # from "no such instrument". The identifier is checked separately.
        fetcher = _data_fetcher(request)

        if identifier not in fetcher.identifiers:
            raise DataNotFoundError(f"instrument '{identifier}'",
                                    source="MarketData")

        frame = fetcher.fetch_corporate_actions(identifier, start, end, types)
        as_of = pd.Timestamp(end) if end else fetcher.date_range[1]

        return CorporateActionsResponse.from_frame(
            identifier=identifier,
            frame=frame,
            trailing_dividend=fetcher.fetch_trailing_dividend(identifier, as_of),
            trailing_dividend_yield=fetcher.fetch_trailing_dividend_yield(
                identifier, as_of),
            cumulative_split_ratio=fetcher.corporate_actions.cumulative_ratio(
                identifier, start, end))

    return router
