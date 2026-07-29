# src/beacon/server/routers/indices.py
"""
Index definition CRUD with structured validation.

A rejected save returns *findings*, not a bare 422. A user editing a pipeline
needs every problem at once, each addressable to the rule that caused it, so
the client can mark the offending row rather than showing one message for the
whole form.
"""
from ..._optional import require
from ...exceptions import DataNotFoundError, InvalidRuleError
from ..definitions import PipelineValidationError, has_errors, validate_document
from ..schemas import (
    IndexCollection,
    IndexDocument,
    SavedIndex,
    ValidationReport,
)
from ..store import DocumentStore
from .universes import load_universe

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request  # noqa: E402

COLLECTION = "indices"


def _store(request: Request) -> DocumentStore:
    """Return the process's index-definition store."""
    store: DocumentStore = request.app.state.index_store

    return store


def _resolve_universe(request: Request,
                      document: IndexDocument) -> IndexDocument:
    """Fill in identifiers from a referenced universe.

    A definition may reference a stored universe instead of listing members.
    Resolving on save means every stored definition carries its identifiers,
    so a consumer never has to chase the reference — and a reference to a
    universe that does not exist fails now rather than at calculation time.
    """
    if document.universe.universe_id is None:
        return document

    universe = load_universe(request, document.universe.universe_id)
    resolved = document.model_copy(deep=True)
    resolved.universe.identifiers = list(universe.identifiers)

    return resolved


def build_indices_router() -> APIRouter:
    """Build the /indices router.

    Returns:
        APIRouter: Router carrying index list, read, validate, create and
        update.
    """
    router = APIRouter(prefix="/indices", tags=["indices"])

    @router.get("", response_model=IndexCollection)
    def list_indices(request: Request) -> IndexCollection:
        return IndexCollection(
            indices=[IndexDocument.model_validate(doc)
                     for doc in _store(request).read_all()])

    @router.post("/validate", response_model=ValidationReport)
    def validate(request: Request,
                 body: IndexDocument) -> ValidationReport:
        # Validation without saving: what the UI's validation card calls as
        # the user edits, so problems surface before a save is attempted.
        resolved = _resolve_universe(request, body)
        findings = validate_document(resolved)

        return ValidationReport(valid=not has_errors(findings), findings=findings)

    @router.get("/{index_id}", response_model=IndexDocument)
    def get_index(request: Request,
                  index_id: str) -> IndexDocument:
        document = _store(request).read(index_id)
        if document is None:
            raise DataNotFoundError(f"index '{index_id}'", source="DocumentStore")

        return IndexDocument.model_validate(document)

    @router.post("", response_model=SavedIndex)
    def create_index(request: Request,
                     body: IndexDocument) -> SavedIndex:
        return _save(request, body.id, body)

    @router.put("/{index_id}", response_model=SavedIndex)
    def put_index(request: Request,
                  index_id: str,
                  body: IndexDocument) -> SavedIndex:
        # The URL owns the id. A body disagreeing with it is a mistake worth
        # reporting rather than silently resolving in either direction.
        if body.id != index_id:
            raise InvalidRuleError(
                f"index '{index_id}'",
                f"body id '{body.id}' does not match the URL id '{index_id}'")

        return _save(request, index_id, body)

    def _save(request: Request,
              index_id: str,
              body: IndexDocument) -> SavedIndex:
        resolved = _resolve_universe(request, body)
        findings = validate_document(resolved)

        if has_errors(findings):
            raise PipelineValidationError(f"index definition '{index_id}'",
                                          "the rule pipeline has errors",
                                          findings)

        _store(request).write(index_id, resolved.model_dump())

        return SavedIndex(index=resolved, findings=findings)

    return router
