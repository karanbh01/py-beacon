# src/beacon/server/routers/watchlists.py
"""
Watchlist CRUD, persisted through the DocumentStore.

Watchlists are user-authored and must outlive the process, so they go to disk
rather than to application state.
"""
from ..._optional import require
from ...exceptions import DataNotFoundError
from ..schemas import Watchlist, WatchlistCollection, WatchlistUpsert
from ..store import DocumentStore

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Request, Response, status  # noqa: E402

COLLECTION = "watchlists"


def _store(request: Request) -> DocumentStore:
    """Return the process's watchlist store."""
    store: DocumentStore = request.app.state.watchlist_store

    return store


def build_watchlists_router() -> APIRouter:
    """Build the /data/watchlists router.

    Returns:
        APIRouter: Router carrying watchlist list, read, upsert and delete.
    """
    router = APIRouter(prefix="/data/watchlists", tags=["watchlists"])

    @router.get("", response_model=WatchlistCollection)
    def list_watchlists(request: Request) -> WatchlistCollection:
        documents = _store(request).read_all()

        return WatchlistCollection(
            watchlists=[Watchlist(id=doc["id"],
                                  name=doc["name"],
                                  identifiers=doc.get("identifiers", []))
                        for doc in documents])

    @router.get("/{watchlist_id}", response_model=Watchlist)
    def get_watchlist(request: Request,
                      watchlist_id: str) -> Watchlist:
        document = _store(request).read(watchlist_id)
        if document is None:
            raise DataNotFoundError(f"watchlist '{watchlist_id}'", source="DocumentStore")

        return Watchlist(id=document["id"],
                         name=document["name"],
                         identifiers=document.get("identifiers", []))

    @router.put("/{watchlist_id}", response_model=Watchlist)
    def put_watchlist(request: Request,
                      watchlist_id: str,
                      body: WatchlistUpsert) -> Watchlist:
        # Upsert rather than separate create/update: the client owns the id,
        # so there is no server-assigned identity to protect.
        watchlist = Watchlist(id=watchlist_id,
                              name=body.name,
                              identifiers=body.identifiers)
        _store(request).write(watchlist_id, watchlist.model_dump())

        return watchlist

    @router.delete("/{watchlist_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_watchlist(request: Request,
                         watchlist_id: str) -> Response:
        if not _store(request).delete(watchlist_id):
            raise DataNotFoundError(f"watchlist '{watchlist_id}'", source="DocumentStore")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
