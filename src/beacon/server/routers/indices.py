# src/beacon/server/routers/indices.py
"""
Index definition CRUD with structured validation.

A rejected save returns *findings*, not a bare 422. A user editing a pipeline
needs every problem at once, each addressable to the rule that caused it, so
the client can mark the offending row rather than showing one message for the
whole form.
"""
from typing import Annotated

import pandas as pd

from ... import catalogue
from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError, DataNotFoundError, InvalidRuleError
from ...index.schedule import FREQUENCY_MONTHS, next_rebalance, rebalance_dates
from ..config import ServerConfig
from ..definitions import (
    PipelineValidationError,
    build_definition,
    has_errors,
    validate_document,
)
from ..documents import load_document, read_collection, validated
from ..jobs import JobRegistry
from ..preview import build_preview
from ..schemas import (
    DeletedIndex,
    DerivationPayload,
    ErrorEnvelope,
    Identifier,
    IndexCollection,
    IndexDeletion,
    IndexDocument,
    OptimiseRequest,
    PreviewDocumentRequest,
    PreviewRequest,
    PreviewResponse,
    RuleTypes,
    SavedIndex,
    ScheduleView,
    ValidationReport,
)
from ..store import DocumentStore
from ..types import specs_for
from .universes import load_universe

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, HTTPException, Query, Request, status  # noqa: E402

COLLECTION = "indices"

AsOfQuery = Annotated[
    str | None,
    Query(description="Date to answer from, YYYY-MM-DD. Defaults to today.")]


def _store(request: Request) -> DocumentStore:
    """Return the process's index-definition store."""
    store: DocumentStore = request.app.state.index_store

    return store


def load_index(request: Request,
               index_id: Identifier) -> IndexDocument:
    """Read an index definition, or answer not-found.

    Not-found covers a definition that is absent, one that is not valid JSON,
    and one that no longer satisfies `IndexDocument` (BN-174) — the third being
    what every stored document hits the day a required field is added to the
    model. The listing skips exactly the same documents, so the two cannot
    disagree about what exists.

    Args:
        request: The incoming request.
        index_id: Identifier of the definition.

    Returns:
        IndexDocument: The stored definition.

    Raises:
        DataNotFoundError: If it cannot be served.
    """
    return load_document(_store(request),
                         index_id,
                         validated(IndexDocument),
                         f"index '{index_id}'")


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

    An optimiser-derived document has no universe of its own: it reallocates
    exactly the names its source published, so there is nothing to resolve.
    """
    if document.universe is None or document.universe.universe_id is None:
        return document

    universe = load_universe(request, document.universe.universe_id)
    resolved = document.model_copy(deep=True)

    assert resolved.universe is not None
    resolved.universe.identifiers = list(universe.identifiers)

    return resolved


def _children_by_source(store: DocumentStore) -> dict[str, list[str]]:
    """Source index id -> the ids of the optimised indices derived from it.

    Read off the stored documents rather than kept as an index of its own: the
    derivation *is* the link, so scanning cannot disagree with it. Raw dicts
    rather than validated documents, because one *invalid* document must not
    stop a delete from finding the children of a readable one — and through the
    tolerant reader, because `store.read_all()` raises on an unparseable file,
    which made that promise false for the one case most likely to occur.
    """
    children: dict[str, list[str]] = {}
    stored_documents, _ = read_collection(store,
                                          lambda _, document: document,
                                          "index definition")

    for stored in stored_documents:
        derivation = stored.get("derivation")
        document_id = stored.get("id")

        if not isinstance(derivation, dict) or not isinstance(document_id, str):
            continue

        source = derivation.get("source_index_id")

        if isinstance(source, str) and source:
            children.setdefault(source, []).append(document_id)

    return children


def _cascade(store: DocumentStore,
             index_id: str) -> list[tuple[str, str | None]]:
    """Everything a delete of *index_id* takes with it, in the order it goes.

    Breadth-first from the named index, each entry paired with the index it was
    derived from (None for the one the request named). The visited set is what
    makes a cycle harmless: a derivation chain that loops is refused at
    calculation time, but it can still be *stored*, and a delete that walked it
    without one would never return.
    """
    children = _children_by_source(store)
    order: list[tuple[str, str | None]] = [(index_id, None)]
    seen = {index_id}
    position = 0

    while position < len(order):
        parent = order[position][0]
        position += 1

        for child in children.get(parent, ()):
            if child not in seen:
                seen.add(child)
                order.append((child, parent))

    return order


def _delete_one(store: DocumentStore,
                registry: JobRegistry,
                records: DocumentStore,
                index_id: str,
                derived_from: str | None) -> DeletedIndex:
    """Put one index through the full cascade and report what went."""
    store.delete(index_id)

    return DeletedIndex(index_id=index_id,
                        derived_from=derived_from,
                        backtest_record_deleted=records.delete(index_id),
                        backtest_results_deleted=registry.forget(
                            f"backtest:{index_id}"))


# How much history and how far ahead the schedule view shows. Enough for a
# client to render "last rebalanced / next rebalance" and a short strip either
# side, without projecting a decade of dates nobody asked for.
SCHEDULE_HORIZON_PERIODS = 4


def build_schedule(document: IndexDocument,
                   as_of: str | None = None) -> ScheduleView:
    """Derive an index's rebalance schedule around a date.

    Derived rather than stored: the next rebalance is a function of the
    schedule, the calendar and today, so storing it would leave a date that
    silently expires.

    Args:
        document: The index definition.
        as_of: The date to answer from; defaults to today.

    Returns:
        ScheduleView: The next rebalance, days until, and a short strip of
        dates either side.
    """
    today = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()

    upcoming_date = next_rebalance(document.rebalancing_frequency,
                                   document.base_date,
                                   today,
                                   document.rebalance_day_rule,
                                   document.calendar)

    months = FREQUENCY_MONTHS[document.rebalancing_frequency]
    window = rebalance_dates(document.rebalancing_frequency,
                             document.base_date,
                             today + pd.DateOffset(
                                 months=months * SCHEDULE_HORIZON_PERIODS),
                             document.rebalance_day_rule,
                             document.calendar)

    return ScheduleView(
        index_id=document.id,
        rebalancing_frequency=document.rebalancing_frequency,
        rebalance_day_rule=document.rebalance_day_rule,
        calendar=document.calendar,
        as_of=str(today.date()),
        next_rebalance=str(upcoming_date.date()) if upcoming_date else None,
        # Calendar days, not sessions: it renders as "in 57 days" and a reader
        # counts those on a wall calendar.
        days_until=((upcoming_date - today).days if upcoming_date else None),
        recent=[str(date.date()) for date in window if date <= today][
            -SCHEDULE_HORIZON_PERIODS:],
        upcoming=[str(date.date()) for date in window if date > today][
            :SCHEDULE_HORIZON_PERIODS])


def build_indices_router() -> APIRouter:
    """Build the /indices router.

    Returns:
        APIRouter: Router carrying index list, read, validate, create and
        update.
    """
    router = APIRouter(prefix="/indices", tags=["indices"])

    @router.get("", response_model=IndexCollection)
    def list_indices(request: Request) -> IndexCollection:
        # A definition the server cannot read is skipped rather than failing
        # the listing (BN-174): one bad file used to take out every index, so
        # the picker went blank and nothing else was reachable through the UI.
        indices, skipped = read_collection(_store(request),
                                           validated(IndexDocument),
                                           "index definition")

        return IndexCollection(indices=indices, skipped=skipped)

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

    @router.post("/preview",
                 response_model=PreviewResponse,
                 responses={422: {"model": ValidationReport}})
    def preview_document(request: Request,
                         body: PreviewDocumentRequest) -> PreviewResponse:
        # The draft route. Its by-id sibling below reads what is *stored*, so
        # an editor holding unsaved changes shows figures for the old
        # definition with nothing to say they are stale.
        #
        # Universe resolution matters more here than there. A stored document
        # was resolved on save and carries its identifiers; a draft has not
        # been, so a definition referencing a universe by id would preview as
        # an empty index without this.
        resolved = _resolve_universe(request, body.document)

        # Validated first, unlike the by-id route, which can trust that
        # anything in the store passed on the way in. A draft naming a rule
        # that does not exist should come back as findings the editor can point
        # at, not as a 500 from the derivation.
        findings = validate_document(resolved)
        if has_errors(findings):
            raise PipelineValidationError(f"index definition '{resolved.id}'",
                                          "the rule pipeline has errors",
                                          findings)

        # The store goes with it because a *derived* draft names its source by
        # id and carries nothing else about it: the document being edited has
        # never been saved, but its parent has, and that is where it is found.
        return build_preview(resolved, _data_fetcher(request),
                             _store(request), body.as_of)

    @router.post("/{index_id}/preview", response_model=PreviewResponse)
    def preview(request: Request,
                index_id: Identifier,
                body: PreviewRequest | None = None) -> PreviewResponse:
        # Kept alongside the body route: a saved-index view has an id and no
        # document in hand, and making it send one back would mean fetching the
        # definition purely to post it again.
        document = load_index(request, index_id)
        as_of = body.as_of if body is not None else None

        return build_preview(document,
                             _data_fetcher(request),
                             _store(request),
                             as_of)

    @router.get("/{index_id}/schedule", response_model=ScheduleView)
    def schedule(request: Request,
                 index_id: Identifier,
                 asof: AsOfQuery = None) -> ScheduleView:
        return build_schedule(load_index(request, index_id), asof)

    @router.get("/{index_id}", response_model=IndexDocument)
    def get_index(request: Request,
                  index_id: Identifier) -> IndexDocument:
        return load_index(request, index_id)

    @router.delete("/{index_id}", response_model=IndexDeletion)
    def delete_index(request: Request,
                     index_id: Identifier) -> IndexDeletion:
        """Remove a stored index definition, its optimised children, and the
        backtest results of every one of them.

        The first cascade is deliberate (BN-157): results are keyed
        `backtest:{index_id}`, and orphaning them would leave records
        addressable by an id that no longer resolves -- the overview route
        404s on the definition load before it ever reaches them.

        The second is the owner's call for BN-168: an optimised index
        *references* its source rather than copying it, so a child left behind
        would be a methodology with no methodology — it could never be
        calculated again. Each child goes through the identical cascade, and
        the chain is followed recursively. The confirmation warning stays
        client-side: documents carry `source_index_id`, so the UI computes the
        blast radius from the catalogue before sending. This response then
        says what actually went, which is what the client reports.

        No refusal case: unlike universes, no index is seeded -- every stored
        definition was created by somebody, so every one may be deleted.
        """
        store = _store(request)

        if not store.exists(index_id):
            raise DataNotFoundError(f"index '{index_id}'",
                                    source="DocumentStore")

        registry: JobRegistry = request.app.state.jobs
        # The record store too (BN-158), or the delete leaves exactly the
        # orphan it exists to prevent: a nested record readable by an id
        # whose definition no longer resolves.
        records: DocumentStore = request.app.state.backtest_record_store

        removed = [_delete_one(store, registry, records, target, parent)
                   for target, parent in _cascade(store, index_id)]

        return IndexDeletion(index_id=index_id, deleted=removed)

    @router.post("", response_model=SavedIndex)
    def create_index(request: Request,
                     body: IndexDocument) -> SavedIndex:
        return _save(request, body.id, body)

    @router.post("/{index_id}/optimise",
                 response_model=SavedIndex,
                 responses={409: {"model": ErrorEnvelope,
                                  "description": "An index with the requested "
                                                 "id already exists."}})
    def optimise_index(request: Request,
                       index_id: Identifier,
                       body: OptimiseRequest) -> SavedIndex:
        """Derive a new, optimised index from a stored one.

        The UI's "Optimise" action on any index. Provenance is server-truth:
        the source is the index in the URL, and the body has no way to assert
        a parentage the server did not create.

        The derived document inherits the parent's identity — base date, base
        value, currency, calendar, and the rebalancing cadence in particular,
        because the child solves exactly at the parent's published snapshots
        and a cadence of its own would have no parent weights at the extra
        dates (design record, default 2).
        """
        parent = load_index(request, index_id)

        if _store(request).exists(body.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"An index with id '{body.id}' already exists. Choose "
                       f"another id, or delete that index first.")

        return _save(request, body.id, _derived_document(parent, body))

    @router.put("/{index_id}", response_model=SavedIndex)
    def put_index(request: Request,
                  index_id: Identifier,
                  body: IndexDocument) -> SavedIndex:
        # The URL owns the id. A body disagreeing with it is a mistake worth
        # reporting rather than silently resolving in either direction.
        if body.id != index_id:
            raise InvalidRuleError(
                f"index '{index_id}'",
                f"body id '{body.id}' does not match the URL id '{index_id}'")

        _refuse_repointed_derivation(request, index_id, body)

        return _save(request, index_id, body)

    def _refuse_repointed_derivation(request: Request,
                                     index_id: Identifier,
                                     body: IndexDocument) -> None:
        """Hold `derivation.source_index_id` still across an update.

        Re-pointing a derivation is a new index, not an edit: every calculated
        level, every stored backtest record and every comparison against the
        old parent would silently become a statement about a different one.
        The objective and the constraints are ordinary rule edits — they
        re-fingerprint and recalculate exactly as a pipeline change does — and
        stay editable.
        """
        stored = _store(request).read(index_id)

        if stored is None:
            return

        current = IndexDocument.model_validate(stored).derivation

        if current is None:
            return

        if body.derivation is None:
            raise InvalidRuleError(
                f"index '{index_id}'",
                f"it is derived from '{current.source_index_id}', and a "
                f"derivation cannot be dropped by an update. Create a new "
                f"index for the rule pipeline")

        if body.derivation.source_index_id != current.source_index_id:
            raise InvalidRuleError(
                f"index '{index_id}'",
                f"its derivation source is '{current.source_index_id}' and "
                f"cannot be changed to '{body.derivation.source_index_id}': "
                f"re-pointing a derivation is a new index, not an edit. The "
                f"objective and the constraints are editable")

    def _save(request: Request,
              index_id: Identifier,
              body: IndexDocument) -> SavedIndex:
        resolved = _resolve_universe(request, body)
        findings = validate_document(resolved)

        if has_errors(findings):
            raise PipelineValidationError(f"index definition '{index_id}'",
                                          "the rule pipeline has errors",
                                          findings)

        if resolved.derivation is not None:
            # Resolved on save for the same reason a referenced universe is: a
            # derivation naming a source that does not exist is a document
            # that can never be calculated, and the editor should hear that
            # now rather than from a backtest a minute later.
            build_definition(resolved, _store(request))

        _store(request).write(index_id, resolved.model_dump())

        return SavedIndex(index=resolved, findings=findings)

    return router


def _derived_document(parent: IndexDocument,
                      body: OptimiseRequest) -> IndexDocument:
    """The document `POST /indices/{id}/optimise` stores.

    A normal index document with all the usual identity attributes, whose
    methodology is the derivation rather than a pipeline. No weights are
    carried — not the parent's, not the solved ones — because no Beacon
    document stores weights.
    """
    return IndexDocument(
        id=body.id,
        name=body.name,
        base_date=parent.base_date,
        base_value=parent.base_value,
        currency=parent.currency,
        rebalancing_frequency=parent.rebalancing_frequency,
        derivation=DerivationPayload(source_index_id=parent.id,
                                     objective=body.objective,
                                     constraints=list(body.constraints)),
        description=body.description,
        return_type=parent.return_type,
        withholding_tax_rate=parent.withholding_tax_rate,
        calendar=parent.calendar,
        rebalance_day_rule=parent.rebalance_day_rule,
        publication_time=parent.publication_time,
        effective_lag_sessions=parent.effective_lag_sessions)
