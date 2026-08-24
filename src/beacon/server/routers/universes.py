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
from typing import Annotated, Any

import pandas as pd

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import (
    ConfigurationError,
    DataNotFoundError,
    ExpressionError,
    InvalidRuleError,
)
from ...expressions.core import from_dict
from ...universe import where
from ..config import ServerConfig
from ..schemas import (
    MODE_FROZEN,
    MODE_LIVE,
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

from fastapi import APIRouter, Query, Request, Response, status  # noqa: E402

COLLECTION = "universes"

AsOfQuery = Annotated[
    str | None,
    Query(description="Date to resolve a live filter at, YYYY-MM-DD. "
                      "Defaults to the end of the loaded data.")]

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
                    source=document.get("source", SOURCE_USER),
                    filter=document.get("filter"),
                    mode=document.get("mode", MODE_FROZEN),
                    as_of=document.get("as_of"))


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


def _data_fetcher(request: Request) -> DataFetcher:
    """The process's data source, or a mapped error.

    Defined here rather than imported from the data router, matching what the
    other routers do: the CRUD endpoints work without a data source, but a
    filter cannot be resolved without one.
    """
    config: ServerConfig = request.app.state.config

    if config.data_fetcher is None:
        raise ConfigurationError(
            "data_source",
            "This server was started without a data source, so a universe "
            "filter cannot be resolved. Restart it with one configured.")

    return config.data_fetcher


def _standing(request: Request) -> pd.Timestamp:
    """The date a filter resolves at by default.

    The end of the loaded data rather than today: a store loaded from a file
    has a last date, and resolving against a calendar the data does not reach
    would select nothing and look like an empty filter.
    """
    return _data_fetcher(request).date_range[1]


def _evaluate(request: Request,
              stored: dict[str, Any],
              date: str | None = None) -> list[str]:
    """Resolve a stored filter against the loaded data."""
    fetcher = _data_fetcher(request)

    try:
        expression = from_dict(stored)
    except ExpressionError as error:
        raise _rejected([Finding(
            path="filter", severity="error", code="MALFORMED_FILTER",
            message=str(error))]) from error

    return where(expression, fetcher, date)


def _members_for(request: Request,
                 body: UniverseCreate) -> list[str]:
    """The membership a create request asks for, however it asked.

    A filter and an explicit list are mutually exclusive rather than merged:
    a request carrying both is ambiguous about which one is the definition,
    and guessing would make the answer depend on an implementation detail.
    """
    if body.filter is None:
        return _resolve_members(request, body.identifiers)

    if body.identifiers:
        raise _rejected([Finding(
            path="filter", severity="error", code="AMBIGUOUS_MEMBERSHIP",
            message="A universe is defined by a filter or by a list of "
                    "identifiers, not both. Send one.")])

    if body.mode not in (MODE_FROZEN, MODE_LIVE):
        raise _rejected([Finding(
            path="mode", severity="error", code="UNKNOWN_MODE",
            message=f"'{body.mode}' is not a mode. Expected "
                    f"'{MODE_FROZEN}' or '{MODE_LIVE}'.")])

    members = _evaluate(request, body.filter)

    if not members:
        raise _rejected([Finding(
            path="filter", severity="error", code="EMPTY_FILTER",
            message="The filter matches no instruments in the loaded data, "
                    "so it would create an empty universe.")])

    return members


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
                    universe_id: Identifier,
                    date: AsOfQuery = None) -> UniverseMembers:
        """The members, re-evaluating the filter when the universe is live.

        This is where the frozen/live distinction becomes observable: a frozen
        universe answers with what it stored, a live one answers with what its
        filter selects now. Both are legitimate; a universe that looked like
        one and behaved like the other would not be.
        """
        universe = load_universe(request, universe_id)

        if universe.mode != MODE_LIVE or universe.filter is None:
            return UniverseMembers(universe_id=universe.id,
                                   identifiers=universe.identifiers)

        return UniverseMembers(
            universe_id=universe.id,
            identifiers=_evaluate(request, universe.filter, date))

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
                            identifiers=_members_for(request, body),
                            description=body.description,
                            source=SOURCE_USER,
                            filter=body.filter,
                            mode=body.mode if body.filter else MODE_FROZEN,
                            as_of=(_standing(request).strftime("%Y-%m-%d")
                                   if body.filter else None))

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
