# src/beacon/server/routers/watchlists.py
"""
Watchlist CRUD, persisted through the DocumentStore.

Watchlists are user-authored and must outlive the process, so they go to disk
rather than to application state.
"""
from ..._optional import require
from ...exceptions import DataNotFoundError
from ..documents import load_document, read_collection, validated
from ..schemas import (
    Identifier,
    TolerantCollection,
    Watchlist,
    WatchlistCollection,
    WatchlistUpsert,
)
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
        # Through the tolerant reader (BN-178): one unparseable file used to
        # 500 the whole listing, so every other watchlist went with it.
        #
        # `validated(Watchlist)` rather than the field-by-field build this
        # carried, and that is the substantive half of the change: reading
        # `doc["name"]` off a raw dict raises KeyError, which `UNREADABLE`
        # deliberately does not catch, so a document missing a field could not
        # have been skipped at all. The model produces exactly the same row for
        # every document that has the fields -- id, name and identifiers, with
        # the same defaults -- and a ValidationError for the ones that do not.
        watchlists, skipped = read_collection(_store(request),
                                              validated(Watchlist),
                                              "watchlist")

        return WatchlistCollection(watchlists=watchlists, **TolerantCollection.skips(skipped))

    @router.get("/{watchlist_id}", response_model=Watchlist)
    def get_watchlist(request: Request,
                      watchlist_id: Identifier) -> Watchlist:
        # The detail half of the same pattern: a document the server cannot
        # read answers exactly as one that was never written, so the listing
        # and this route cannot disagree about what exists.
        return load_document(_store(request),
                             watchlist_id,
                             validated(Watchlist),
                             f"watchlist '{watchlist_id}'")

    @router.put("/{watchlist_id}", response_model=Watchlist)
    def put_watchlist(request: Request,
                      watchlist_id: Identifier,
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
                         watchlist_id: Identifier) -> Response:
        if not _store(request).delete(watchlist_id):
            raise DataNotFoundError(f"watchlist '{watchlist_id}'", source="DocumentStore")

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return router
