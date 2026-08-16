# src/beacon/server/routers/universes.py
"""
Universes: named sets of instrument identifiers.

A universe is a server-side concept. The library has no universe object — an
`IndexDefinition` carries a plain list of identifiers — so these documents
exist to let several definitions share one curated list rather than each
repeating it.

## Members are checked against the loaded data

A universe naming an instrument the server has no data for is a universe that
produces an empty index and no explanation. Both `POST` and `PUT` resolve
every member against the loaded data and refuse the ones that are not there —
as *findings*, naming each missing identifier, in the shape the index editor
already renders. A bare 422 would tell somebody a list of five hundred tickers
was wrong without saying which one.

## Seeded universes are read-only

The synthetic generator writes a `GLOBAL` universe covering everything it
produced, so a fresh workspace has something to select. It is marked
`source: "seeded"` and refuses edits: it derives from the dataset, so
regenerating would discard whatever had been changed. Refusing now beats
losing it later.
"""
import re
from typing import Any

from ..._optional import require
from ...exceptions import DataNotFoundError, InvalidRuleError
from ..schemas import (
    SOURCE_SEEDED,
    SOURCE_USER,
    Finding,
    Identifier,
    Universe,
    UniverseCollection,
    UniverseCreate,
    UniverseMembers,
    UniverseUpsert,
)
from ..store import DocumentStore

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, Response, status  # noqa: E402

COLLECTION = "universes"

# How many missing identifiers a rejection names before it stops listing them.
# A paste of a thousand tickers with nine hundred typos should not answer with
# nine hundred findings; the summary finding still reports the true count.
MAX_REPORTED_MISSING = 20

# The universe every loaded dataset gets for free.
#
# Without it a fresh workspace answers `GET /universes` with an empty list, so
# the index editor has nothing to select and the universe pane nothing to
# show -- the data is all there and none of it is reachable.
#
# Seeded by the *server* rather than by the synthetic generator, which is a
# deliberate departure from the issue. Universe documents live under
# `storage_root`, not in the data store the generator writes, so a generator
# that seeded them would have to know where the server keeps its documents and
# would still leave every other source (a local store, a yfinance sync)
# without one. Doing it where the data is loaded covers all of them.
GLOBAL_ID = "GLOBAL"
GLOBAL_NAME = "Global"
GLOBAL_DESCRIPTION = ("Every instrument in the loaded dataset. Written by the "
                      "server when the data was loaded, and refreshed when "
                      "the dataset changes.")


def _store(request: Request) -> DocumentStore:
    """Return the process's universe store."""
    store: DocumentStore = request.app.state.universe_store

    return store


def _to_universe(document: dict[str, Any]) -> Universe:
    """Build the response model from a stored document.

    `source` defaults to "user" for documents written before it existed:
    absent means somebody created it, because nothing else could have.
    """
    return Universe(id=document["id"],
                    name=document["name"],
                    identifiers=document.get("identifiers", []),
                    description=document.get("description"),
                    source=document.get("source", SOURCE_USER))


def slug(name: str) -> str:
    """Derive an identifier from a display name.

    Lower-cased, runs of non-alphanumerics collapsed to one dash, trimmed:
    "My Tech Names!" becomes "my-tech-names", which is what appears in the URL.

    Returns an empty string when nothing survives. A name of pure punctuation
    has no identifier, and the caller refuses it rather than inventing one.
    """
    reduced = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())

    return reduced.strip("-")[:64]


class UniverseValidationError(InvalidRuleError):
    """An invalid universe, carrying every finding.

    The same shape `PipelineValidationError` uses for index definitions, and
    for the same reason: `InvalidRuleError` already maps to 422 with the
    INVALID_RULE code, and the error envelope reads `findings` off the
    exception to build its structured detail. A client receives every bad
    identifier at once rather than one message for the whole list.
    """
    def __init__(self,
                 reason: str,
                 findings: list[Finding]):
        super().__init__("universe", reason)
        self.findings = [finding.model_dump() for finding in findings]


def _rejected(findings: list[Finding]) -> UniverseValidationError:
    """A rejection carrying findings, in the shape the index editor renders."""
    return UniverseValidationError("its members are not valid", findings)


def _resolve_members(request: Request,
                     identifiers: list[str],
                     path: str = "identifiers",
                     allow_empty: bool = False) -> list[str]:
    """De-duplicate members and refuse any the loaded data does not know.

    Args:
        request: The incoming request, for the data source.
        identifiers: What the caller sent.
        path: Where the findings should point.
        allow_empty: Whether an empty list is acceptable. False on create --
            a universe with no members is not a thing anybody meant to make.
            True on update, where clearing the list is a step somebody takes
            part-way through editing one, and where refusing it would change
            behaviour this endpoint has always had.

    Returns:
        list: The members, de-duplicated, in the order first given. Order is
        kept rather than sorted because a curated list has an order somebody
        chose.

    Raises:
        InvalidRuleError: If the list is empty or names anything unknown,
            carrying one finding per missing identifier.
    """
    seen: dict[str, None] = {}
    for identifier in identifiers:
        seen.setdefault(identifier.strip(), None)

    members = [identifier for identifier in seen if identifier]

    if not members:
        if allow_empty:
            return members

        raise _rejected([Finding(
            path=path, severity="error", code="EMPTY_UNIVERSE",
            message="A universe must name at least one identifier.")])

    fetcher = getattr(request.app.state.config, "data_fetcher", None)

    if fetcher is None:
        # Nothing to check against. A server started without a data source can
        # still hold universes; it simply cannot say whether members exist.
        return members

    known = set(fetcher.identifiers)
    reference = fetcher.reference_identifiers

    if reference:
        known |= set(reference)

    missing = [identifier for identifier in members if identifier not in known]

    if not missing:
        return members

    shown = missing[:MAX_REPORTED_MISSING]
    findings = [
        Finding(path=f"{path}[{members.index(identifier)}]",
                severity="error", code="UNKNOWN_IDENTIFIER",
                message=f"'{identifier}' is not in the loaded data.")
        for identifier in shown]

    if len(missing) > len(shown):
        findings.append(Finding(
            path=path, severity="error", code="UNKNOWN_IDENTIFIER",
            message=f"{len(missing) - len(shown)} further identifier(s) are "
                    f"also not in the loaded data."))

    raise _rejected(findings)


def _refuse_if_seeded(document: dict[str, Any],
                      universe_id: str) -> None:
    """Stop an edit to a generator-written universe."""
    if document.get("source") == SOURCE_SEEDED:
        raise InvalidRuleError(
            f"universe '{universe_id}'",
            "it was written by the data generator and is read-only. Copy it "
            "with POST /universes to make an editable version.")


def load_universe(request: Request,
                  universe_id: Identifier) -> Universe:
    """Read a universe or raise the mapped not-found error.

    Shared with the indices router, which resolves a universe reference when
    saving a definition.

    Args:
        request: The incoming request.
        universe_id: Identifier of the universe.

    Returns:
        Universe: The stored universe.

    Raises:
        DataNotFoundError: If no such universe exists.
    """
    document = _store(request).read(universe_id)
    if document is None:
        raise DataNotFoundError(f"universe '{universe_id}'", source="DocumentStore")

    return _to_universe(document)


def seed_global_universe(store: DocumentStore,
                         fetcher: Any) -> bool:
    """Write the GLOBAL universe for a loaded dataset.

    Idempotent, and deterministic: the members are the dataset's identifiers
    in sorted order, so regenerating with the same seed reproduces the same
    document byte for byte.

    Rewritten when the dataset's membership changes -- a store swapped for a
    larger one should not leave GLOBAL describing the old one -- but left
    alone otherwise, so the file's mtime does not churn on every boot.

    Args:
        store: The universe document store.
        fetcher: The loaded data source.

    Returns:
        bool: Whether anything was written.
    """
    identifiers = sorted(set(fetcher.reference_identifiers
                             or fetcher.identifiers))

    if not identifiers:
        return False

    document = Universe(id=GLOBAL_ID,
                        name=GLOBAL_NAME,
                        identifiers=identifiers,
                        description=GLOBAL_DESCRIPTION,
                        source=SOURCE_SEEDED).model_dump()

    existing = store.read(GLOBAL_ID)

    if existing is not None and existing.get("identifiers") == identifiers:
        return False

    store.write(GLOBAL_ID, document)

    return True


def build_universes_router() -> APIRouter:
    """Build the /universes router.

    Returns:
        APIRouter: Router carrying universe list, read, members, upsert and
        delete.
    """
    router = APIRouter(prefix="/universes", tags=["universes"])

    @router.get("", response_model=UniverseCollection)
    def list_universes(request: Request) -> UniverseCollection:
        return UniverseCollection(
            universes=[_to_universe(doc) for doc in _store(request).read_all()])

    @router.get("/{universe_id}", response_model=Universe)
    def get_universe(request: Request,
                     universe_id: Identifier) -> Universe:
        return load_universe(request, universe_id)

    @router.get("/{universe_id}/members", response_model=UniverseMembers)
    def get_members(request: Request,
                    universe_id: Identifier) -> UniverseMembers:
        universe = load_universe(request, universe_id)

        return UniverseMembers(universe_id=universe.id,
                               identifiers=universe.identifiers)

    @router.post("", response_model=Universe,
                 status_code=status.HTTP_201_CREATED)
    def create_universe(request: Request,
                        body: UniverseCreate) -> Universe:
        universe_id = slug(body.name)

        if not universe_id:
            raise _rejected([Finding(
                path="name", severity="error", code="UNUSABLE_NAME",
                message=f"'{body.name}' has no letters or digits, so no "
                        f"identifier can be derived from it.")])

        store = _store(request)

        if store.exists(universe_id):
            raise InvalidRuleError(
                f"universe '{universe_id}'",
                f"a universe named '{body.name}' already exists. Choose "
                f"another name, or PUT to replace it.")

        universe = Universe(id=universe_id,
                            name=body.name,
                            identifiers=_resolve_members(request,
                                                         body.identifiers),
                            description=body.description,
                            source=SOURCE_USER)

        store.write(universe_id, universe.model_dump())

        return universe

    @router.put("/{universe_id}", response_model=Universe)
    def put_universe(request: Request,
                     universe_id: Identifier,
                     body: UniverseUpsert) -> Universe:
        store = _store(request)
        existing = store.read(universe_id)

        if existing is not None:
            _refuse_if_seeded(existing, universe_id)

        # Validated on update as well as create. PUT predates the loaded data,
        # so it accepted any list at all -- meaning a universe could be edited
        # into naming instruments the server has no prices for, and the index
        # built from it would come back empty with nothing to point at.
        universe = Universe(id=universe_id,
                            name=body.name,
                            identifiers=_resolve_members(request,
                                                         body.identifiers,
                                                         allow_empty=True),
                            description=body.description,
                            source=SOURCE_USER)
        store.write(universe_id, universe.model_dump())

        return universe

    @router.delete("/{universe_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_universe(request: Request,
                        universe_id: Identifier) -> Response:
        store = _store(request)
        existing = store.read(universe_id)

        if existing is not None:
            _refuse_if_seeded(existing, universe_id)

        if not store.delete(universe_id):
            raise DataNotFoundError(f"universe '{universe_id}'", source="DocumentStore")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
