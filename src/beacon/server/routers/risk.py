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
import asyncio
from typing import Any

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import DataNotFoundError
from ..active_data import require_data
from ..documents import load_document, validated
from ..jobs import JobRegistry
from ..market_caps import DEFAULT_CURRENCY
from ..risk import build_estimation_job
from ..schemas import (
    Identifier,
    IndexDocument,
    RiskModelCollection,
    RiskModelJobStatus,
    RiskModelRequest,
    RiskModelSummary,
    RiskModelView,
)
from ..store import DocumentStore
from ..strategies import load_strategy
from ..strategy_runs import index_weights, require_run, strategy_weights

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, status  # noqa: E402

# Job kind prefix, so a stored result can be found by model id.
KIND = "risk"


def _data_fetcher(request: Request) -> DataFetcher:
    """The loaded data, or 409 NO_DATA_LOADED (see `active_data`)."""
    return require_data(request, "a risk model cannot be estimated")


def _source_index(request: Request,
                  body: RiskModelRequest) -> IndexDocument | None:
    """The index the names come from: the one named, or the one the named
    strategy holds; None when the names are listed."""
    index_id = body.index_id

    if body.strategy_id is not None:
        index_id = load_strategy(request.app.state.strategy_store, body.strategy_id).index_id

    if index_id is None:
        return None

    store: DocumentStore = request.app.state.index_store

    return load_document(store, index_id, validated(IndexDocument), f"index '{index_id}'")


def _currency(request: Request,
              body: RiskModelRequest) -> str:
    """The currency returns are measured in: the request's, else the index's,
    else USD."""
    if body.currency is not None:
        return body.currency.upper()

    document = _source_index(request, body)

    return document.currency.upper() if document is not None else DEFAULT_CURRENCY


def _universe(request: Request,
              body: RiskModelRequest) -> list[str]:
    """The names to estimate over.

    An explicit list wins. Otherwise an index's names at its latest rebalance,
    calculated, or what a strategy held at its latest run's last rebalance,
    which is how a client asks for "the risk model for this" without
    restating the names.
    """
    if body.identifiers:
        return list(body.identifiers)

    if body.strategy_id is not None:
        registry: JobRegistry = request.app.state.jobs
        _, held = strategy_weights(require_run(registry, body.strategy_id), None)

        return sorted(held)

    document = _source_index(request, body)

    if document is None:
        raise DataNotFoundError(
            "identifiers to estimate over",
            source="supply `identifiers`, or an `index_id` or `strategy_id` to take "
                   "them from")

    store: DocumentStore = request.app.state.index_store
    _, weights = index_weights(document, store, _data_fetcher(request), None)

    return sorted(weights)


def _model(request: Request,
           model_id: Identifier) -> dict[str, Any]:
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
                  model_id: Identifier) -> RiskModelView:
        return RiskModelView.model_validate(_model(request, model_id))

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/{model_id}/estimate",
                 response_model=RiskModelJobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def estimate(request: Request,
                       model_id: Identifier,
                       body: RiskModelRequest | None = None) -> RiskModelJobStatus:
        # The universe and the data source are resolved before the job is
        # submitted, so a bad request fails immediately rather than becoming a
        # job that fails a moment later for the client to discover.
        settings = body if body is not None else RiskModelRequest()
        # Off the loop (BN-333): an index's names are calculated here.
        identifiers = await asyncio.to_thread(_universe, request, settings)
        fetcher = _data_fetcher(request)
        currency = _currency(request, settings)

        registry: JobRegistry = request.app.state.jobs
        job = registry.submit(
            f"{KIND}:{model_id}",
            build_estimation_job(model_id, settings, identifiers, fetcher,
                                 currency))

        return RiskModelJobStatus(**job.snapshot())

    return router
