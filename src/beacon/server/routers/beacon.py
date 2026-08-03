# src/beacon/server/routers/beacon.py
"""
Beacon View: running an index's backtest, and reading the result.

A backtest calculates the whole index and then simulates a tracking portfolio
day by day, which is far too slow to hold an HTTP connection open for. The
submission endpoint therefore returns a job id; the client polls
`GET /jobs/{id}` or listens on `/ws`.

The read endpoints are the other half. They serve the panes of the view —
overview, weights, attribution, a single name, and a comparison across
indices — and each answers from the **most recent successful run** of that
index rather than recalculating anything. That is why BN-91 persisted job
results and why BN-71 extended the payload to carry composition: a client
switching tabs should not be waiting on a recalculation, and two panes read a
moment apart should describe the same run.

Every read is a 404 until a backtest has been run, which is the honest answer:
there is no view of an index nobody has calculated.
"""
from typing import Annotated, Any

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError
from ..backtests import build_backtest_job
from ..config import ServerConfig
from ..jobs import JobRegistry
from ..runs import snapshot_at, snapshots_from
from ..schemas import (
    AssetView,
    AttributionView,
    BacktestRequest,
    CompareView,
    IndexDocument,
    JobStatus,
    OverviewView,
    WeightsView,
)
from ..store import DocumentStore
from ..views import (
    build_asset_view,
    build_attribution,
    build_compare,
    build_overview,
)
from ..weights import build_weights

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request, status  # noqa: E402

BenchmarkQuery = Annotated[
    str | None,
    Query(description="Index id to measure tracking error against. "
                      "Requires risk=true.")]
RiskQuery = Annotated[
    bool,
    Query(description="Decompose the index's volatility across its "
                      "constituents. Off by default: estimating a "
                      "covariance over every name is the pane's whole "
                      "cost.")]
AsOfQuery = Annotated[
    str | None,
    Query(description="Date to report at, YYYY-MM-DD. Defaults to the latest "
                      "rebalance.")]
StartQuery = Annotated[str | None,
                       Query(description="Inclusive start date, YYYY-MM-DD.")]
EndQuery = Annotated[str | None,
                     Query(description="Inclusive end date, YYYY-MM-DD.")]
IdsQuery = Annotated[list[str],
                     Query(description="Index ids to compare, two or more.")]


def _latest_run(request: Request,
                index_id: str) -> dict[str, Any]:
    """The most recent successful backtest result for an index.

    Raises:
        DataNotFoundError: If the index has never been backtested successfully.
            Distinct from an unknown index, which fails earlier when the
            definition is loaded.
    """
    registry: JobRegistry = request.app.state.jobs
    run = registry.latest_result(f"backtest:{index_id}")

    if run is None:
        raise DataNotFoundError(
            f"a completed backtest for index '{index_id}'",
            source="run POST /beacon/{index_id}/backtest first")

    return run


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

    # Compare is declared before the parameterised routes so that a request for
    # /beacon/compare is not captured by /beacon/{index_id}/... — FastAPI
    # matches in declaration order, and "compare" would otherwise read as an
    # index id.
    @router.get("/compare", response_model=CompareView)
    def compare(request: Request,
                ids: IdsQuery) -> CompareView:
        if len(ids) < 2:
            raise DataNotFoundError(
                "at least two indices to compare",
                source=f"{len(ids)} id(s) were given")

        # Every definition is resolved first, so an unknown id fails as a 404
        # naming that id rather than as a comparison that quietly covers fewer
        # indices than were asked for.
        for index_id in ids:
            _index_document(request, index_id)

        return build_compare({index_id: _latest_run(request, index_id)
                              for index_id in ids})

    @router.get("/{index_id}/overview", response_model=OverviewView)
    def overview(request: Request,
                 index_id: str) -> OverviewView:
        document = _index_document(request, index_id)

        return build_overview(index_id, document.name, _latest_run(request, index_id))

    @router.get("/{index_id}/weights", response_model=WeightsView)
    def weights(request: Request,
                index_id: str,
                asof: AsOfQuery = None,
                risk: RiskQuery = False,
                benchmark: BenchmarkQuery = None) -> WeightsView:
        _index_document(request, index_id)

        # The benchmark's weights come from its own latest run, taken at the
        # same date. Reading them from a run rather than re-deriving the index
        # means the comparison is against what that benchmark actually was.
        benchmark_weights = None
        if benchmark and risk:
            _index_document(request, benchmark)
            reference = snapshot_at(
                snapshots_from(_latest_run(request, benchmark)), asof)
            benchmark_weights = dict(reference.weights)

        return build_weights(index_id,
                             _latest_run(request, index_id),
                             asof,
                             _data_fetcher(request),
                             with_risk=risk,
                             benchmark=benchmark_weights,
                             benchmark_id=benchmark if benchmark_weights else None)

    @router.get("/{index_id}/attribution", response_model=AttributionView)
    def attribution(request: Request,
                    index_id: str,
                    start: StartQuery = None,
                    end: EndQuery = None) -> AttributionView:
        _index_document(request, index_id)

        return build_attribution(index_id,
                                 _latest_run(request, index_id),
                                 _data_fetcher(request),
                                 start,
                                 end)

    @router.get("/{index_id}/assets/{identifier}", response_model=AssetView)
    def asset(request: Request,
              index_id: str,
              identifier: str) -> AssetView:
        _index_document(request, index_id)

        return build_asset_view(index_id,
                                identifier,
                                _latest_run(request, index_id),
                                _data_fetcher(request))

    return router
