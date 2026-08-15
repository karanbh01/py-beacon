# src/beacon/server/routers/universes.py
"""
Universes: named sets of instrument identifiers.

A universe is a server-side concept. The library has no universe object — an
`IndexDefinition` carries a plain list of identifiers — so these documents
exist to let several definitions share one curated list rather than each
repeating it.

The issue asks only for the two read endpoints. PUT and DELETE are here
because without a way to create one, the reads could only ever return an empty
collection.
"""
from typing import Any

from ..._optional import require
from ...exceptions import DataNotFoundError
from ..schemas import Identifier, Universe, UniverseCollection, UniverseMembers, UniverseUpsert
from ..store import DocumentStore

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, Response, status  # noqa: E402

COLLECTION = "universes"


def _store(request: Request) -> DocumentStore:
    """Return the process's universe store."""
    store: DocumentStore = request.app.state.universe_store

    return store


def _to_universe(document: dict[str, Any]) -> Universe:
    """Build the response model from a stored document."""
    return Universe(id=document["id"],
                    name=document["name"],
                    identifiers=document.get("identifiers", []))


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

    @router.put("/{universe_id}", response_model=Universe)
    def put_universe(request: Request,
                     universe_id: Identifier,
                     body: UniverseUpsert) -> Universe:
        universe = Universe(id=universe_id,
                            name=body.name,
                            identifiers=body.identifiers)
        _store(request).write(universe_id, universe.model_dump())

        return universe

    @router.delete("/{universe_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_universe(request: Request,
                        universe_id: Identifier) -> Response:
        if not _store(request).delete(universe_id):
            raise DataNotFoundError(f"universe '{universe_id}'", source="DocumentStore")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
