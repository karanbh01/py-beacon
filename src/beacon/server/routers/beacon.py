# src/beacon/server/routers/beacon.py
"""
Beacon View: running and reading an index's backtest.

A backtest calculates the whole index and then simulates a tracking portfolio
day by day, which is far too slow to hold an HTTP connection open for. The
endpoint therefore submits a job and returns its id; the client polls
`GET /jobs/{id}` or listens on `/ws`.
"""
from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError
from ..backtests import build_backtest_job
from ..config import ServerConfig
from ..jobs import JobRegistry
from ..schemas import BacktestRequest, IndexDocument, JobStatus
from ..store import DocumentStore

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, status  # noqa: E402


def _index_document(request: Request,
                    index_id: str) -> IndexDocument:
    """Load a stored index definition, or fail with a mapped error."""
    store: DocumentStore = request.app.state.index_store
    document = store.read(index_id)
    if document is None:
        raise DataNotFoundError(f"index '{index_id}'", source="DocumentStore")

    return IndexDocument.model_validate(document)


def _data_fetcher(request: Request) -> DataFetcher:
    """Return the process's data source, or fail with a mapped error."""
    config: ServerConfig = request.app.state.config
    if config.data_fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so a backtest "
            "cannot be run. Restart it with one configured.")

    return config.data_fetcher


def build_beacon_router() -> APIRouter:
    """Build the /beacon router.

    Returns:
        APIRouter: Router carrying the backtest submission endpoint.
    """
    router = APIRouter(prefix="/beacon", tags=["beacon"])

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry's
    # asyncio.create_task to attach to.
    @router.post("/{index_id}/backtest",
                 response_model=JobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def submit_backtest(request: Request,
                              index_id: str,
                              body: BacktestRequest | None = None) -> JobStatus:
        # The definition and the data source are resolved here, before the job
        # is submitted, so an unknown index or an unconfigured server fails
        # immediately with a proper error instead of becoming a job that fails
        # a moment later for the client to discover.
        document = _index_document(request, index_id)
        fetcher = _data_fetcher(request)
        settings = body if body is not None else BacktestRequest()

        registry: JobRegistry = request.app.state.jobs
        store: DocumentStore = request.app.state.index_store
        job = registry.submit(
            f"backtest:{index_id}",
            build_backtest_job(document, fetcher, settings, store))

        return JobStatus(**job.snapshot())

    return router
