# src/beacon/server/routers/risk.py
"""
Risk-model endpoints.

Estimation is a job: it means pulling a price history for every name in the
universe before any matrix arithmetic happens. Reading a finished model is
cheap and serves the stored result, the same arrangement as backtests and
optimisation runs.

The model id is chosen by the caller rather than generated, so a client can
re-estimate "the one I use for tech" over a new window and keep referring to it
by the same name. Each estimate supersedes the last under that id.
"""
from typing import Any

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError
from ..jobs import JobRegistry
from ..risk import build_estimation_job
from ..schemas import (
    JobStatus,
    RiskModelCollection,
    RiskModelRequest,
    RiskModelSummary,
    RiskModelView,
)

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, status  # noqa: E402

# Job kind prefix, so a stored result can be found by model id.
KIND = "risk"


def _data_fetcher(request: Request) -> DataFetcher:
    """The process's data source, or a mapped error."""
    fetcher = request.app.state.config.data_fetcher
    if fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so a risk model "
            "cannot be estimated. Restart it with one configured.")

    return fetcher  # type: ignore[no-any-return]


def _universe(request: Request,
              body: RiskModelRequest) -> list[str]:
    """The names to estimate over.

    An explicit list wins; otherwise the constituents of an index's latest run,
    which is how a client asks for "the risk model for this index" without
    restating its universe.
    """
    if body.identifiers:
        return list(body.identifiers)

    if body.index_id is None:
        raise DataNotFoundError(
            "identifiers to estimate over",
            source="supply `identifiers`, or an `index_id` to take them from")

    registry: JobRegistry = request.app.state.jobs
    backtest = registry.latest_result(f"backtest:{body.index_id}")
    if backtest is None:
        raise DataNotFoundError(
            f"a completed backtest for index '{body.index_id}'",
            source="run POST /beacon/{index_id}/backtest first")

    snapshots = backtest.get("rebalances") or []
    if not snapshots:
        raise DataNotFoundError(
            "rebalance snapshots on that run",
            source="the run predates composition being stored")

    return sorted(snapshots[-1]["weights"])


def _model(request: Request,
           model_id: str) -> dict[str, Any]:
    """A stored risk model, or a mapped error."""
    registry: JobRegistry = request.app.state.jobs
    result = registry.latest_result(f"{KIND}:{model_id}")

    if result is None:
        raise DataNotFoundError(
            f"an estimated risk model '{model_id}'",
            source="estimate it with POST /risk-models/{model_id}/estimate")

    return result


def build_risk_router() -> APIRouter:
    """Build the /risk-models router.

    Returns:
        APIRouter: Router carrying model reads and the estimation job.
    """
    router = APIRouter(prefix="/risk-models", tags=["risk"])

    @router.get("", response_model=RiskModelCollection)
    def list_models(request: Request) -> RiskModelCollection:
        registry: JobRegistry = request.app.state.jobs

        summaries = [
            RiskModelSummary(
                model_id=result["model_id"],
                assets=result["diagnostics"]["assets"],
                observations=result["diagnostics"]["observations"],
                average_correlation=result["diagnostics"]["average_correlation"],
                positive_semi_definite=result["diagnostics"][
                    "positive_semi_definite"])
            for result in registry.latest_results_by_kind(f"{KIND}:").values()
        ]
        summaries.sort(key=lambda entry: entry.model_id)

        return RiskModelCollection(risk_models=summaries)

    @router.get("/{model_id}", response_model=RiskModelView)
    def get_model(request: Request,
                  model_id: str) -> RiskModelView:
        return RiskModelView.model_validate(_model(request, model_id))

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/{model_id}/estimate",
                 response_model=JobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def estimate(request: Request,
                       model_id: str,
                       body: RiskModelRequest | None = None) -> JobStatus:
        # The universe and the data source are resolved before the job is
        # submitted, so a bad request fails immediately rather than becoming a
        # job that fails a moment later for the client to discover.
        settings = body if body is not None else RiskModelRequest()
        identifiers = _universe(request, settings)
        fetcher = _data_fetcher(request)

        registry: JobRegistry = request.app.state.jobs
        job = registry.submit(
            f"{KIND}:{model_id}",
            build_estimation_job(model_id, settings, identifiers, fetcher))

        return JobStatus(**job.snapshot())

    return router
