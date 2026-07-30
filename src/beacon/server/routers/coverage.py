# src/beacon/server/routers/coverage.py
"""
Data-coverage reporting and the sync job.

Coverage reports what is genuinely knowable about each dataset: whether it is
loaded, how many identifiers it holds, and the span of dates it covers.

`POST /{dataset}/sync` returned 501 until BN-100 gave the library an ingestion
path. It is a job now: fetching several hundred identifiers over a network is
exactly the long-running work the job machinery exists for, and holding an HTTP
connection open for it would be the wrong shape.
"""
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...data.ingest import (
    Downloader,
    IngestResult,
    ingest_market_data,
    ingest_reference_data,
    yfinance_downloader,
    yfinance_reference_downloader,
)
from ..config import ServerConfig
from ..jobs import JobRegistry, ProgressReporter
from ..schemas import CoverageResponse, DatasetCoverage, JobStatus, SyncRequest

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Request, status  # noqa: E402

logger = logging.getLogger(__name__)

MARKET = "market"
REFERENCE = "reference"
DATASETS = (MARKET, REFERENCE)


def _market_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the market dataset."""
    if fetcher is None:
        return DatasetCoverage(dataset=MARKET, configured=False, identifiers=0)

    identifiers = fetcher.identifiers
    if not identifiers:
        return DatasetCoverage(dataset=MARKET, configured=True, identifiers=0)

    start, end = fetcher.date_range

    return DatasetCoverage(dataset=MARKET,
                           configured=True,
                           identifiers=len(identifiers),
                           start=start.isoformat(),
                           end=end.isoformat())


def _reference_coverage(fetcher: DataFetcher | None) -> DatasetCoverage:
    """Describe the reference dataset.

    Reference data has validity windows rather than a single date axis, so no
    start/end is reported for it.
    """
    if fetcher is None or fetcher.reference_identifiers is None:
        return DatasetCoverage(dataset=REFERENCE, configured=False, identifiers=0)

    return DatasetCoverage(dataset=REFERENCE,
                           configured=True,
                           identifiers=len(fetcher.reference_identifiers))


def build_coverage_router() -> APIRouter:
    """Build the /data/coverage router.

    Returns:
        APIRouter: Router carrying coverage reporting and the sync endpoint.
    """
    router = APIRouter(prefix="/data/coverage", tags=["coverage"])

    @router.get("", response_model=CoverageResponse)
    def coverage(request: Request) -> CoverageResponse:
        config: ServerConfig = request.app.state.config
        fetcher = config.data_fetcher

        return CoverageResponse(datasets=[_market_coverage(fetcher),
                                          _reference_coverage(fetcher)])

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/{dataset}/sync",
                 response_model=JobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def sync(request: Request,
                   dataset: str,
                   body: SyncRequest | None = None) -> JobStatus:
        # Dataset, data source and identifier list are all resolved before the
        # job is submitted, so a bad request fails immediately with a proper
        # error rather than becoming a job that fails a moment later for the
        # client to discover.
        if dataset not in DATASETS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown dataset '{dataset}'. Known: {', '.join(DATASETS)}.")

        config: ServerConfig = request.app.state.config
        fetcher = config.data_fetcher
        if fetcher is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="This server was started without a data source, so there "
                       "is nothing to sync into.")

        settings = body if body is not None else SyncRequest()
        identifiers = list(settings.identifiers) or fetcher.identifiers
        if not identifiers:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Nothing to sync: no identifiers were supplied and the "
                       "loaded dataset is empty.")

        registry: JobRegistry = request.app.state.jobs
        job = registry.submit(
            f"sync:{dataset}",
            build_sync_job(dataset, identifiers, settings, fetcher, registry,
                           config.market_downloader))

        return JobStatus(**job.snapshot())

    return router


def build_sync_job(dataset: str,
                   identifiers: list[str],
                   settings: SyncRequest,
                   fetcher: DataFetcher,
                   registry: JobRegistry,
                   downloader: Downloader | None
                   ) -> Callable[[ProgressReporter], Awaitable[dict[str, Any]]]:
    """Build the coroutine that runs a sync.

    Args:
        dataset: MARKET or REFERENCE.
        identifiers: What to fetch.
        settings: Window and options from the request.
        fetcher: The data source to merge into.
        registry: Used to publish the freshness event on completion.
        downloader: Injected source. None builds the yfinance-backed one, and
            that is where a missing `data` extra surfaces — inside the job, so
            it reaches the client as a failed job carrying the install message
            rather than as an import error at startup.

    Returns:
        A coroutine function suitable for JobRegistry.submit.
    """
    async def run(report: ProgressReporter) -> dict[str, Any]:
        await report(0.0, f"Syncing {len(identifiers)} identifier(s).")

        seen: list[tuple[int, int, str]] = []

        def note(done: int, total: int, identifier: str) -> None:
            # Recorded rather than awaited: the ingestion loop is synchronous
            # and cannot await, so progress is collected as it happens and
            # published once the fetch returns. Making the whole ingest path
            # async for the sake of a progress bar would be the tail wagging
            # the dog, and it would drag asyncio into a library function that
            # has no other reason to know about it.
            seen.append((done, total, identifier))

        result = _run_ingestion(dataset, identifiers, settings, downloader, note)

        for done, total, identifier in seen[-1:]:
            await report(done / total, f"Fetched {identifier} ({done}/{total}).")

        added = _merge(dataset, fetcher, result)

        await report(1.0, f"Synced {len(result.fetched)} of {len(identifiers)}.")

        # Announced only once the data is actually queryable, so a client that
        # refetches on the event cannot beat the merge to it.
        registry.publish_data_freshness(dataset,
                                        {"identifiers": len(result.fetched),
                                         "rows_added": added})

        return {**result.summary(), "dataset": dataset, "rows_added": added}

    return run


def _run_ingestion(dataset: str,
                   identifiers: list[str],
                   settings: SyncRequest,
                   downloader: Downloader | None,
                   note: Callable[[int, int, str], None]) -> IngestResult:
    """Fetch the requested dataset."""
    if dataset == REFERENCE:
        return ingest_reference_data(identifiers,
                                     yfinance_reference_downloader(),
                                     on_progress=note)

    return ingest_market_data(identifiers,
                              downloader if downloader is not None
                              else yfinance_downloader(),
                              start=settings.start,
                              end=settings.end,
                              on_progress=note)


def _merge(dataset: str,
           fetcher: DataFetcher,
           result: IngestResult) -> int:
    """Fold the fetched data into the live source."""
    if dataset == REFERENCE:
        return fetcher.merge_reference_data(result.reference)

    return fetcher.merge_market_data(result.market)
