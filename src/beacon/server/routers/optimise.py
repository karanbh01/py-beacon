# src/beacon/server/routers/optimise.py
"""
Optimiser endpoints: constraint sets, runs, frontier and exposures.

Constraint sets are stored documents like indices and watchlists. A run is a
job, because the solve itself is fast but the risk model it needs is not — that
means a price history for every constituent and a covariance built from it.

The frontier and exposures panes read a completed run rather than re-solving,
which is the same arrangement as the Beacon View endpoints and for the same
reason: a client switching tabs should not wait on a recalculation.

Validation happens before the job. A malformed constraint set is a bad request
and the client should learn that from the submission, not from a job that fails
a moment later — and it reports every problem it finds, addressed to the row
that caused it, because someone fixing a constraint editor needs all the errors
rather than the first.
"""
import uuid
from typing import Annotated, Any

from ... import catalogue
from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError
from ..constraints import (
    build_constraints,
    constraint_types,
    has_errors,
    label_map,
    validate_constraint_set,
)
from ..jobs import JobRegistry
from ..optimisation import (
    build_exposures,
    build_frontier,
    build_optimisation_job,
    target_weights_from,
)
from ..schemas import (
    ConstraintSet,
    ConstraintSetCollection,
    ConstraintTypes,
    ExposuresView,
    FrontierView,
    JobStatus,
    OptimisationRunRequest,
    SavedConstraintSet,
    ValidationReport,
)
from ..store import DocumentStore
from ..types import specs_for

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Query, Request, status  # noqa: E402

RiskFreeQuery = Annotated[
    float, Query(description="Rate the tangency point is measured against.")]


def _store(request: Request) -> DocumentStore:
    """The constraint-set store."""
    store: DocumentStore = request.app.state.constraint_store
    return store


def _data_fetcher(request: Request) -> DataFetcher:
    """The process's data source, or a mapped error."""
    fetcher = request.app.state.config.data_fetcher
    if fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so an "
            "optimisation cannot be run. Restart it with one configured.")

    return fetcher  # type: ignore[no-any-return]


def _constraint_set(request: Request,
                    set_id: str) -> ConstraintSet:
    """Load a stored constraint set, or fail with a mapped error."""
    document = _store(request).read(set_id)
    if document is None:
        raise DataNotFoundError(f"constraint set '{set_id}'",
                                source="DocumentStore")

    return ConstraintSet.model_validate(document)


def _run(request: Request,
         run_id: str) -> dict[str, Any]:
    """Load a completed optimisation result by its run id."""
    registry: JobRegistry = request.app.state.jobs
    result = registry.latest_result(f"optimise:{run_id}")

    if result is None:
        raise DataNotFoundError(
            f"a completed optimisation run '{run_id}'",
            source="the run may still be in progress, or may have failed")

    return result


def build_optimise_router() -> APIRouter:
    """Build the /optimise router.

    Returns:
        APIRouter: Router carrying constraint-set CRUD, run submission, and the
        frontier and exposures reads.
    """
    router = APIRouter(prefix="/optimise", tags=["optimise"])

    @router.get("/constraint-types", response_model=ConstraintTypes)
    def types() -> ConstraintTypes:
        # `types` is kept as it was and `specs` added beside it: a client
        # written against the original shape keeps working, and one that wants
        # to render a form reads the richer field.
        return ConstraintTypes(types=constraint_types(),
                               specs=specs_for(catalogue.CONSTRAINT))

    @router.get("/constraint-sets", response_model=ConstraintSetCollection)
    def list_sets(request: Request) -> ConstraintSetCollection:
        return ConstraintSetCollection(
            constraint_sets=[ConstraintSet.model_validate(document)
                             for document in _store(request).read_all()])

    @router.post("/constraint-sets/validate", response_model=ValidationReport)
    def validate(body: ConstraintSet) -> ValidationReport:
        findings = validate_constraint_set(body)

        return ValidationReport(valid=not has_errors(findings), findings=findings)

    @router.get("/constraint-sets/{set_id}", response_model=ConstraintSet)
    def get_set(request: Request,
                set_id: str) -> ConstraintSet:
        return _constraint_set(request, set_id)

    @router.put("/constraint-sets/{set_id}",
                response_model=SavedConstraintSet,
                responses={422: {"model": ValidationReport}})
    def put_set(request: Request,
                set_id: str,
                body: ConstraintSet) -> SavedConstraintSet:
        # The path wins over the body, so a document cannot be saved under one
        # id while claiming another.
        document = body.model_copy(update={"id": set_id})
        findings = validate_constraint_set(document)

        if has_errors(findings):
            raise _rejected(findings)

        _store(request).write(set_id, document.model_dump())

        return SavedConstraintSet(constraint_set=document, findings=findings)

    @router.delete("/constraint-sets/{set_id}",
                   status_code=status.HTTP_204_NO_CONTENT)
    def delete_set(request: Request,
                   set_id: str) -> None:
        if not _store(request).exists(set_id):
            raise DataNotFoundError(f"constraint set '{set_id}'",
                                    source="DocumentStore")

        _store(request).delete(set_id)

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry to
    # attach a task to.
    @router.post("/runs",
                 response_model=JobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def submit_run(request: Request,
                         body: OptimisationRunRequest) -> JobStatus:
        # Everything resolvable is resolved before the job is submitted, so a
        # bad request fails immediately with a proper error rather than
        # becoming a job that fails a moment later for the client to discover.
        constraint_set = _constraint_set(request, body.constraint_set_id)
        findings = validate_constraint_set(constraint_set)
        if has_errors(findings):
            raise _rejected(findings)

        registry: JobRegistry = request.app.state.jobs
        backtest = registry.latest_result(f"backtest:{body.index_id}")
        if backtest is None:
            raise DataNotFoundError(
                f"a completed backtest for index '{body.index_id}'",
                source="run POST /beacon/{index_id}/backtest first")

        run_id = str(uuid.uuid4())
        job = registry.submit(
            f"optimise:{run_id}",
            build_optimisation_job(run_id,
                                   body,
                                   constraint_set,
                                   build_constraints(constraint_set),
                                   target_weights_from(backtest, body.as_of),
                                   label_map(constraint_set),
                                   _data_fetcher(request)))

        return JobStatus(**job.snapshot())

    @router.get("/runs/{run_id}/frontier", response_model=FrontierView)
    def frontier(request: Request,
                 run_id: str,
                 risk_free_rate: RiskFreeQuery = 0.0) -> FrontierView:
        run = _run(request, run_id)
        constraint_set = _constraint_set(request, run["constraint_set_id"])

        return build_frontier(run,
                              build_constraints(constraint_set),
                              _data_fetcher(request),
                              risk_free_rate)

    @router.get("/runs/{run_id}/exposures", response_model=ExposuresView)
    def exposures(request: Request,
                  run_id: str) -> ExposuresView:
        return build_exposures(_run(request, run_id), _data_fetcher(request))

    return router


def _rejected(findings: list[Any]) -> Any:
    """A 422 carrying the validation report.

    Raised rather than returned so the error handler renders it in the same
    envelope as every other failure, with the findings intact.
    """
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=ValidationReport(valid=False, findings=findings).model_dump())
