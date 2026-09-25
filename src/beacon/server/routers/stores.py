# src/beacon/server/routers/stores.py
"""Named data stores: list, register, rename, forget, and load one (BN-236).

Loading runs as a job, like a backtest: reading a large store takes seconds,
and the client watches its progress on the event socket. When it finishes,
the engine serves the new store's data, remembers it for the next start, and
publishes a `data.loaded` event.

A request already running when a load finishes keeps the data it started
with: the swap replaces the engine's data whole, it never changes it in place.
"""
import asyncio
from pathlib import Path
from typing import Any

from ..._optional import require
from ...data import store as data_store
from ...exceptions import DataNotFoundError
from ..active_data import ActiveData, active_data
from ..data_stores import StoreRegistry, describe, load
from ..errors import FindingsError
from ..jobs import JobRegistry, ProgressReporter
from ..schemas import (
    ErrorEnvelope,
    Finding,
    Identifier,
    LoadJobStatus,
    LoadResult,
    TolerantCollection,
)
from ..store_schemas import (
    DataStore,
    DataStoreCollection,
    DataStoreCreate,
    DataStoreUpdate,
)
from .universes import seed_global_universe

require("fastapi", "The Beacon API server")

from fastapi import (  # noqa: E402
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)

CONFLICT_RESPONSE: dict[int | str, dict[str, Any]] = {
    409: {"model": ErrorEnvelope,
          "description": "The engine's state refuses the request: a store is "
                         "already loading, the store is the one being "
                         "served, or the folder is already registered."}}


def _registry(request: Request) -> StoreRegistry:
    registry: StoreRegistry = request.app.state.data_stores

    return registry


def _record(request: Request,
            store_id: str) -> dict[str, Any]:
    """A store's record, or 404 naming it."""
    record = _registry(request).get(store_id)

    if record is None:
        raise DataNotFoundError(f"data store '{store_id}'",
                                source="the store registry")

    return record


def _view(request: Request,
          record: dict[str, Any]) -> DataStore:
    return describe(record, active_data(request).store_id)


def _refuse_unless_a_store(path: Path) -> None:
    """Refuse a folder that is not a py-beacon data store, saying why."""
    problem = None

    if not path.is_absolute():
        problem = "Give the folder as an absolute path."
    elif not path.is_dir():
        problem = f"There is no folder at {path}."
    elif not data_store.exists(path):
        problem = (f"{path} is not a py-beacon data store: it needs "
                   f"{data_store.MANIFEST_NAME} and {data_store.MARKET_FILE}.")

    if problem is not None:
        raise FindingsError("data store", "its folder cannot be used", [
            Finding(path="path", rule_id=None, severity="error",
                    code="NOT_A_DATA_STORE", message=problem)])


def build_stores_router() -> APIRouter:
    """Build the /data/stores router."""
    router = APIRouter(prefix="/data/stores", tags=["data stores"])

    @router.get("", response_model=DataStoreCollection)
    def list_stores(request: Request) -> DataStoreCollection:
        records, skipped = _registry(request).listing()

        return DataStoreCollection(stores=[_view(request, record)
                                           for record in records],
                                   active=active_data(request).store_id,
                                   **TolerantCollection.skips(skipped))

    @router.post("",
                 response_model=DataStore,
                 status_code=status.HTTP_201_CREATED,
                 responses=CONFLICT_RESPONSE)
    def register_store(request: Request,
                       body: DataStoreCreate) -> DataStore:
        path = Path(body.path)
        _refuse_unless_a_store(path)

        existing = _registry(request).find_by_path(path)

        if existing is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"{path} is already registered as "
                       f"'{existing['name']}' ({existing['id']}).")

        return _view(request, _registry(request).create(body.name, path,
                                                        body.kind))

    @router.get("/{store_id}", response_model=DataStore)
    def get_store(request: Request,
                  store_id: Identifier) -> DataStore:
        return _view(request, _record(request, store_id))

    @router.patch("/{store_id}", response_model=DataStore)
    def rename_store(request: Request,
                     store_id: Identifier,
                     body: DataStoreUpdate) -> DataStore:
        _record(request, store_id)
        renamed = _registry(request).rename(store_id, body.name)
        holder = active_data(request)

        if holder.store_id == store_id and renamed is not None:
            holder.store_name = renamed["name"]

        return _view(request, _record(request, store_id))

    @router.delete("/{store_id}",
                   status_code=status.HTTP_204_NO_CONTENT,
                   responses=CONFLICT_RESPONSE)
    def forget_store(request: Request,
                     store_id: Identifier) -> Response:
        """Forget a store. Its folder and files are left untouched."""
        _record(request, store_id)

        if active_data(request).store_id == store_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This is the store the engine is serving. Load "
                       "another store first.")

        _registry(request).delete(store_id)

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # async, as every job-submitting endpoint must be: the registry attaches
    # the job to the running event loop.
    @router.post("/{store_id}/activate",
                 response_model=LoadJobStatus,
                 status_code=status.HTTP_202_ACCEPTED,
                 responses=CONFLICT_RESPONSE)
    async def activate_store(request: Request,
                             store_id: Identifier) -> LoadJobStatus:
        record = _record(request, store_id)
        holder = active_data(request)

        if holder.loading:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A data store is already loading. Wait for it to "
                       "finish.")

        if not data_store.exists(Path(record["path"])):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"The store '{record['name']}' cannot be read: "
                       f"nothing usable at {record['path']}.")

        # Set before the job is scheduled, so a second request arriving
        # before the job starts is refused rather than racing it.
        holder.loading = True
        jobs: JobRegistry = request.app.state.jobs
        job = jobs.submit(f"load:{store_id}",
                          _load_job(request.app, record))

        return LoadJobStatus(**job.snapshot())

    return router


def _load_job(app: FastAPI,
              record: dict[str, Any]) -> Any:
    """The coroutine that loads a store and starts serving it."""

    async def run(report: ProgressReporter) -> dict[str, Any]:
        holder: ActiveData = app.state.active_data

        try:
            await report(0.05, f"Reading '{record['name']}'")
            fetcher = await asyncio.to_thread(load, record)

            await report(0.9, "Serving the new data")
            registry: StoreRegistry = app.state.data_stores
            app.state.active_data = ActiveData(fetcher=fetcher,
                                               store_id=record["id"],
                                               store_name=record["name"])
            registry.set_active(record["id"])
            registry.mark_loaded(record["id"])

            # GLOBAL describes the loaded dataset, so it follows the switch.
            seed_global_universe(app.state.universe_store, fetcher)

            first, last = fetcher.date_range
            result = LoadResult(store_id=record["id"],
                                name=record["name"],
                                identifiers=len(fetcher.identifiers),
                                start=f"{first:%Y-%m-%d}",
                                end=f"{last:%Y-%m-%d}")

            jobs: JobRegistry = app.state.jobs
            jobs.publish_data_loaded(record["id"], record["name"])

            return result.model_dump()
        finally:
            # A failed load leaves the previous data in place. Either way the
            # next load may start.
            app.state.active_data.loading = False
            holder.loading = False

    return run
