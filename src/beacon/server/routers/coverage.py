# src/beacon/server/routers/coverage.py
"""
Data-coverage reporting and the sync job.

Coverage reports whether each dataset is loaded, how many identifiers it holds,
the span of dates it covers, and when it was last refreshed. A null age means
the dataset is not loaded at all, which is a different statement from "loaded
and never refreshed"; see `decisions/0002-caching-and-data-freshness.md`.

`POST /{dataset}/sync` is deprecated. It refreshes the active store from its
own source, the same as `POST /data/stores/{store_id}/refresh`.
"""
# Freshness joined the report in BN-99. BN-240 deprecated `/sync`: it used to
# download from Yahoo Finance into memory, and now delegates to the store
# refresh.
import logging
from typing import Any

from ..._optional import require
from ...data import store
from ...data.fetcher import (
    ACTIONS_DATASET,
    FEATURES_DATASET,
    FREQUENCY_FOR_DATASET,
    FX_DATASET,
    STALE_AFTER_SECONDS,
    DataFetcher,
)
from ..active_data import active_data, current_data, require_data
from ..data_stores import StoreRegistry
from ..schemas import (
    CoverageResponse,
    DatasetCoverage,
    RefreshJobStatus,
    SyncRequest,
)
from .refresh import submit_refresh

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Request, status  # noqa: E402

logger = logging.getLogger(__name__)

MARKET = "market"
REFERENCE = "reference"
ACTIONS = ACTIONS_DATASET
FEATURES = FEATURES_DATASET
FX = FX_DATASET

# The datasets the deprecated sync route accepts in its path. Any of them
# refreshes the whole store.
DATASETS = (MARKET, REFERENCE)


def _freshness(fetcher: DataFetcher,
               dataset: str) -> dict[str, Any]:
    """Age and timestamp for one dataset.

    Both age and timestamp, not just the age: an age is only true at the
    instant it was read, and a client holding a response for a minute needs the
    timestamp to work out what it now has.

    The frequency and its threshold travel with them so a client renders
    "stale" without holding its own 24h/7d numbers. A threshold in a UI is a
    guess at a property of the data, and it diverges from the engine the moment
    either changes.
    """
    stamped = fetcher.last_refreshed(dataset)
    frequency = FREQUENCY_FOR_DATASET[dataset]

    return {"cache_age": fetcher.age_seconds(dataset),
            "last_refreshed": stamped.isoformat() if stamped else None,
            "frequency": frequency,
            "stale_after_seconds": STALE_AFTER_SECONDS[frequency]}


def _provenance(fetcher: DataFetcher,
                dataset: str) -> dict[str, Any]:
    """Where a dataset came from, and what it costs on disk.

    The size is this dataset's file, not the whole store: three rows each
    showing the store total would display the same number three times and make
    any sum of them wrong. The store total is reported once, at the top.
    """
    path = fetcher.store_path

    return {"source": fetcher.source,
            "cache_size_bytes": (store.dataset_size_on_disk(path, dataset)
                                 if path else None)}


def _market_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the market dataset."""
    if fetcher is None:
        return DatasetCoverage(dataset=MARKET, configured=False, identifiers=0)

    common = {**_freshness(fetcher, MARKET), **_provenance(fetcher, MARKET),
              "field_count": len(fetcher.market_columns)}

    identifiers = fetcher.identifiers
    if not identifiers:
        return DatasetCoverage(dataset=MARKET, configured=True, identifiers=0,
                               **common)

    start, end = fetcher.date_range

    return DatasetCoverage(dataset=MARKET,
                           configured=True,
                           identifiers=len(identifiers),
                           start=start.isoformat(),
                           end=end.isoformat(),
                           **common)


def _reference_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the reference dataset.

    Reference data has validity windows rather than a single date axis, so no
    start/end is reported for it.
    """
    if fetcher is None or fetcher.reference_identifiers is None:
        return DatasetCoverage(dataset=REFERENCE, configured=False, identifiers=0)

    # The validity columns are keys, not fields: counting DATE_FROM and DATE_TO
    # would make "three fields held" mean one real attribute.
    columns = [name for name in (fetcher.reference_columns or [])
               if name not in ("DATE_FROM", "DATE_TO")]

    return DatasetCoverage(dataset=REFERENCE,
                           configured=True,
                           identifiers=len(fetcher.reference_identifiers),
                           field_count=len(columns),
                           **_freshness(fetcher, REFERENCE),
                           **_provenance(fetcher, REFERENCE))


def _actions_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the corporate-action history.

    Reported even when empty, because "we hold no actions" is a fact the pane
    should state rather than a dataset it should omit.
    """
    if fetcher is None or fetcher.corporate_actions.is_empty:
        return DatasetCoverage(dataset=ACTIONS, configured=False, identifiers=0)

    actions = fetcher.corporate_actions
    dates = actions.data["EX_DATE"]
    fields = [name for name in actions.data.columns
              if name not in ("IDENTIFIER", "EX_DATE")]

    return DatasetCoverage(dataset=ACTIONS,
                           configured=True,
                           identifiers=len(actions.identifiers),
                           start=dates.min().isoformat(),
                           end=dates.max().isoformat(),
                           field_count=len(fields),
                           **_freshness(fetcher, ACTIONS),
                           **_provenance(fetcher, ACTIONS))


def _features_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the feature table.

    Reported even when empty, like the actions above: "we hold no features" is
    a fact the pane should state rather than a dataset it should omit, and a
    client deciding whether to offer a fundamentals screen needs the answer
    either way.
    """
    if fetcher is None or fetcher.features.is_empty:
        return DatasetCoverage(dataset=FEATURES, configured=False,
                               identifiers=0)

    features = fetcher.features
    dates = features.data.index.get_level_values("DATE")

    return DatasetCoverage(dataset=FEATURES,
                           configured=True,
                           identifiers=len(features.identifiers),
                           start=dates.min().isoformat(),
                           end=dates.max().isoformat(),
                           # Distinct FIELD values, not columns. A feature
                           # table has five columns however many datapoints it
                           # carries, so a column count would report the same
                           # number for every store ever loaded.
                           field_count=len(features.fields()),
                           **_freshness(fetcher, FEATURES),
                           **_provenance(fetcher, FEATURES))


def _fx_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the currency pairs.

    A dataset of its own in the report, though the rows live in the market
    frame. "Do we hold exchange rates" has its own answer, and a client
    deciding whether to offer an unhedged-versus-hedged comparison needs it,
    and a market row count cannot say.

    The pairs stay counted inside `market` as well, because they *are* market
    rows and a client summing the identifier counts would otherwise be told
    the store holds more instruments than it does. `identifiers_union` is
    unaffected for the same reason: it is distinct names across datasets, and
    these were already among them.
    """
    if fetcher is None or not fetcher.fx_pairs:
        return DatasetCoverage(dataset=FX, configured=False, identifiers=0)

    pairs = fetcher.fx_pairs
    start, end = fetcher.date_range

    return DatasetCoverage(dataset=FX,
                           configured=True,
                           identifiers=len(pairs),
                           start=start.isoformat(),
                           end=end.isoformat(),
                           # One field: the rate. The pairs share the market
                           # frame's columns, but the others are null on a
                           # pair, and reporting eight would claim we hold a
                           # volume and a share count for EURUSD.
                           field_count=1,
                           **_freshness(fetcher, FX),
                           # No file of its own: the pairs are rows in the
                           # market file, which `market` already reports.
                           # Repeating that size here would count it twice and
                           # make the sum of the parts exceed the store total,
                           # which is the invariant `TestCacheSize` pins.
                           source=fetcher.source,
                           cache_size_bytes=None)


def _identifiers_union(fetcher: DataFetcher | None) -> int:
    """Distinct identifiers across every dataset.

    Not the sum of the per-dataset counts. A name held in both market and
    reference data would be counted twice, and "assets covered" would come out
    above the size of the universe it is drawn from.
    """
    if fetcher is None:
        return 0

    names = set(fetcher.identifiers)
    names |= set(fetcher.reference_identifiers or [])
    names |= set(fetcher.corporate_actions.identifiers)
    names |= set(fetcher.features.identifiers)

    return len(names)


def build_coverage_router() -> APIRouter:
    """Build the /data/coverage router.

    Returns:
        APIRouter: Router carrying coverage reporting and the sync endpoint.
    """
    router = APIRouter(prefix="/data/coverage", tags=["coverage"])

    @router.get("", response_model=CoverageResponse)
    def coverage(request: Request) -> CoverageResponse:
        fetcher = current_data(request)

        path = fetcher.store_path if fetcher else None

        return CoverageResponse(
            datasets=[_market_coverage(fetcher),
                      _reference_coverage(fetcher),
                      _actions_coverage(fetcher),
                      _features_coverage(fetcher),
                      _fx_coverage(fetcher)],
            identifiers_union=_identifiers_union(fetcher),
            cache_size_bytes=store.size_on_disk(path) if path else None)

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/{dataset}/sync",
                 response_model=RefreshJobStatus,
                 status_code=status.HTTP_202_ACCEPTED,
                 deprecated=True)
    async def sync(request: Request,
                   dataset: str,
                   body: SyncRequest | None = None) -> RefreshJobStatus:
        """Refresh the store being served. Deprecated: use
        `POST /data/stores/{store_id}/refresh`.

        Kept so existing clients work. It refreshes the whole active store
        from its own source, whichever dataset is named; the body's fields
        are ignored. It downloads from Yahoo Finance only when the store is
        set to refresh from it.
        """
        # Until BN-240 this downloaded from Yahoo Finance into memory.
        if dataset not in DATASETS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown dataset '{dataset}'. Known: {', '.join(DATASETS)}.")

        require_data(request, "there is nothing to sync")
        store_id = active_data(request).store_id
        registry: StoreRegistry = request.app.state.data_stores
        record = registry.get(store_id) if store_id is not None else None

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The data being served was named when the engine "
                       "started rather than registered as a store, so it has "
                       "no source to refresh from. Register it as a store to "
                       "refresh it.")

        return submit_refresh(request, record)

    return router
