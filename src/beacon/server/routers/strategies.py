# src/beacon/server/routers/strategies.py
"""
Stored strategies: `/strategies`.

A strategy is what a backtest holds (an index in full, through a replication,
or an active strategy measured against it) and how it trades. It is stored
beside the index definitions, so it outlives the process and travels with
them; everything else a backtest needs stays on the backtest.
"""
# BN-312.
from ..._optional import require
from ...exceptions import DataNotFoundError, InvalidRuleError
from ..definitions import has_errors
from ..documents import load_document, read_collection, stored
from ..errors import FindingsError
from ..schemas import Identifier, TolerantCollection, ValidationReport
from ..store import DocumentStore
from ..strategies import read_strategy, validate_strategy, with_problem
from ..strategy_schemas import (
    SavedStrategy,
    StoredStrategy,
    StrategyCollection,
    StrategyDeletion,
    StrategyDocument,
)

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Request, status  # noqa: E402


def _strategies(request: Request) -> DocumentStore:
    """Return the process's strategy store."""
    store: DocumentStore = request.app.state.strategy_store

    return store


def _indices(request: Request) -> DocumentStore:
    """Return the process's index store, which strategies refer to."""
    store: DocumentStore = request.app.state.index_store

    return store


def build_strategies_router() -> APIRouter:
    """Build the /strategies router.

    Returns:
        APIRouter: Router carrying strategy list, read, validate, create,
        replace and delete.
    """
    router = APIRouter(prefix="/strategies", tags=["strategies"])

    @router.get("", response_model=StrategyCollection)
    def list_strategies(request: Request) -> StrategyCollection:
        """Every stored strategy, each with any problem it has, such as an
        index that has gone. A strategy the server cannot read is left out and
        counted, as the index list does."""
        documents, skipped = read_collection(_strategies(request),
                                             read_strategy,
                                             "strategy")
        indices = _indices(request)

        return StrategyCollection(strategies=[with_problem(document, indices)
                                              for document in documents],
                                  **TolerantCollection.skips(skipped))

    @router.post("/validate", response_model=ValidationReport)
    def validate(request: Request,
                 body: StrategyDocument) -> ValidationReport:
        """Check a strategy without saving it: the findings a save would
        refuse it with, each at its field's path."""
        findings = validate_strategy(body, _indices(request))

        return ValidationReport(valid=not has_errors(findings), findings=findings)

    @router.get("/{strategy_id}", response_model=StoredStrategy)
    def get_strategy(request: Request,
                     strategy_id: Identifier) -> StoredStrategy:
        document = load_document(_strategies(request),
                                 strategy_id,
                                 read_strategy,
                                 f"strategy '{strategy_id}'")

        return with_problem(document, _indices(request))

    @router.post("", response_model=SavedStrategy,
                 responses={409: {"description": "A strategy with this id already exists."}})
    def create_strategy(request: Request,
                        body: StrategyDocument) -> SavedStrategy:
        """Save a new strategy. An id already taken is refused with a 409:
        replacing a strategy is `PUT /strategies/{strategy_id}`."""
        taken = stored(_strategies(request), body.id, read_strategy)

        if taken.present:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                                detail=_taken(body.id, taken.readable))

        return _save(request, body)

    @router.put("/{strategy_id}", response_model=SavedStrategy)
    def put_strategy(request: Request,
                     strategy_id: Identifier,
                     body: StrategyDocument) -> SavedStrategy:
        """Save a strategy under its id, replacing any stored there."""
        # The URL owns the id, as it does for an index.
        if body.id != strategy_id:
            raise InvalidRuleError(
                f"strategy '{strategy_id}'",
                f"body id '{body.id}' does not match the URL id '{strategy_id}'")

        return _save(request, body)

    @router.delete("/{strategy_id}", response_model=StrategyDeletion)
    def delete_strategy(request: Request,
                        strategy_id: Identifier) -> StrategyDeletion:
        """Remove a stored strategy. One the server cannot read can still be
        deleted."""
        if not _strategies(request).delete(strategy_id):
            raise DataNotFoundError(f"strategy '{strategy_id}'", source="DocumentStore")

        return StrategyDeletion(strategy_id=strategy_id)

    def _save(request: Request,
              body: StrategyDocument) -> SavedStrategy:
        findings = validate_strategy(body, _indices(request))

        if has_errors(findings):
            raise FindingsError(f"strategy '{body.id}'", "the strategy has errors", findings)

        _strategies(request).write(body.id, body.model_dump())

        return SavedStrategy(strategy=body, findings=findings)

    return router


def _taken(strategy_id: str,
           readable: bool) -> str:
    """The 409 an occupied id earns, saying which way it is occupied."""
    if readable:
        return (f"A strategy with id '{strategy_id}' already exists. Choose "
                f"another id, or replace it with PUT /strategies/{strategy_id}.")

    return (f"A strategy with id '{strategy_id}' is stored but cannot be read, "
            f"which is why reading it answers not-found. Delete it "
            f"(DELETE /strategies/{strategy_id}) or replace it with PUT.")
