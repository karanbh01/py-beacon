# src/beacon/server/routers/indices.py
"""
Index definition CRUD with structured validation.

A rejected save returns *findings*, not a bare 422. A user editing a pipeline
needs every problem at once, each addressable to the rule that caused it, so
the client can mark the offending row rather than showing one message for the
whole form.
"""
from ... import catalogue
from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError, InvalidRuleError
from ..config import ServerConfig
from ..definitions import PipelineValidationError, has_errors, validate_document
from ..preview import build_preview
from ..schemas import (
    IndexCollection,
    IndexDocument,
    PreviewRequest,
    PreviewResponse,
    RuleTypes,
    SavedIndex,
    ValidationReport,
)
from ..store import DocumentStore
from ..types import specs_for
from .universes import load_universe

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request  # noqa: E402

COLLECTION = "indices"


def _store(request: Request) -> DocumentStore:
    """Return the process's index-definition store."""
    store: DocumentStore = request.app.state.index_store

    return store


def _data_fetcher(request: Request) -> DataFetcher:
    """Return the process's data source, or fail with a mapped error.

    Preview evaluates real rules against real prices, so unlike the CRUD
    endpoints it cannot run without one.
    """
    config: ServerConfig = request.app.state.config
    if config.data_fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so a constituent "
            "preview cannot be derived. Restart it with one configured.")

    return config.data_fetcher


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

    # Declared before the "/{index_id}" routes so the literal path is not
    # swallowed as an index id. Static segments are matched first regardless,
    # but relying on that is a footgun the next person should not have to know
    # about.
    @router.get("/rule-types", response_model=RuleTypes)
    def rule_types() -> RuleTypes:
        # Needs no data source and no stored document: this describes what the
        # library can do, not what this server happens to hold, so it answers
        # on a process started with nothing configured.
        return RuleTypes(selection=specs_for(catalogue.SELECTION),
                         weighting=specs_for(catalogue.WEIGHTING))

    @router.post("/validate", response_model=ValidationReport)
    def validate(request: Request,
                 body: IndexDocument) -> ValidationReport:
        # Validation without saving: what the UI's validation card calls as
        # the user edits, so problems surface before a save is attempted.
        resolved = _resolve_universe(request, body)
        findings = validate_document(resolved)

        return ValidationReport(valid=not has_errors(findings), findings=findings)

    @router.post("/{index_id}/preview", response_model=PreviewResponse)
    def preview(request: Request,
                index_id: str,
                body: PreviewRequest | None = None) -> PreviewResponse:
        document = _store(request).read(index_id)
        if document is None:
            raise DataNotFoundError(f"index '{index_id}'", source="DocumentStore")

        as_of = body.as_of if body is not None else None

        return build_preview(IndexDocument.model_validate(document),
                             _data_fetcher(request),
                             as_of)

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
