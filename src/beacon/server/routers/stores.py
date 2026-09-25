# src/beacon/server/routers/stores.py
"""Named data stores: list, register, rename, forget, and load one.

Loading runs as a job, like a backtest: reading a large store takes seconds,
and the client watches its progress on the event socket. When it finishes,
the engine serves the new store's data, remembers it for the next start, and
publishes a `data.loaded` event.

A request already running when a load finishes keeps the data it started
with: the swap replaces the engine's data whole, it never changes it in place.
"""
# Named stores arrived in BN-236.
import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from ..._optional import require
from ...data import postgres
from ...data import store as data_store
from ...exceptions import DataNotFoundError
from ..active_data import ActiveData, active_data
from ..data_stores import StoreRegistry, available, describe, load
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

logger = logging.getLogger(__name__)

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


def record_or_404(request: Request,
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


def managed_root(app: FastAPI) -> Path:
    """The folder the engine creates its own stores in."""
    root: Path = app.state.managed_store_root

    return root


def remove_managed_folder(app: FastAPI,
                          path: Path) -> None:
    """Delete a folder the engine created, and nothing outside its own root.

    The check is the point: a registry record could have been edited by hand,
    and deleting whatever path it names would make one bad file a way to
    remove anything on the disk.
    """
    root = managed_root(app).resolve()
    target = path.resolve()

    if target.parent != root:
        logger.warning("Not deleting %s: it is outside the engine's store "
                       "folder %s.", target, root)
        return

    shutil.rmtree(target, ignore_errors=True)


def refuse_if_refreshing(request: Request,
                         record: dict[str, Any]) -> None:
    """Refuse to load or forget a store while a refresh is rewriting it (409)."""
    refreshing: set[str] = request.app.state.refreshing

    if record["id"] in refreshing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"'{record['name']}' is refreshing. Wait for it to "
                   f"finish.")


def _refuse_if_registered(registry: StoreRegistry,
                          location: Path | str,
                          kind: str) -> None:
    """Refuse a folder or database that already has a store (409)."""
    existing = registry.find_by_path(location, kind)

    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"{location} is already registered as "
                   f"'{existing['name']}' ({existing['id']}).")


def _yahoo_finding(message: str) -> Finding:
    return Finding(path="refresh_from", rule_id=None, severity="error",
                   code="REFRESH_SOURCE_UNSUITABLE", message=message)


def _refuse_yahoo_for_synthetic(path: Path,
                                refresh_from: str) -> None:
    """Refuse Yahoo Finance as the refresh source of synthetic data (422).

    Its tickers are invented, so Yahoo has nothing for them, and mixing
    downloaded prices into a generated market would make it neither.
    """
    if refresh_from != "yfinance" or not data_store.exists(path):
        return

    if data_store.read_manifest(path).source == data_store.SOURCE_SYNTHETIC:
        raise FindingsError("data store", "it cannot refresh from Yahoo "
                            "Finance", [_yahoo_finding(
                                "Synthetic data has invented tickers that "
                                "Yahoo Finance does not know. It refreshes "
                                "by extending instead.")])


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
        """Register a folder or a database, checking it can be read first.

        A database is connected to and read in full before it is registered,
        so a store that is registered is one that loaded at least once. Its
        problems are refused with one finding each, as an import's are.
        """
        registry = _registry(request)

        if body.kind == "postgres":
            assert body.connection is not None
            connection = body.connection.model_dump(by_alias=True)
            source = postgres.PostgresSource(**connection)
            location = source.describe()
            _refuse_if_registered(registry, location, "postgres")
            postgres.load(source)

            return _view(request, registry.create(body.name, location,
                                                  "postgres",
                                                  connection=connection))

        assert body.path is not None
        path = Path(body.path)
        _refuse_unless_a_store(path)
        _refuse_if_registered(registry, path, "folder")
        _refuse_yahoo_for_synthetic(path, body.refresh_from)

        return _view(request, registry.create(body.name, path, "folder",
                                              refresh_from=body.refresh_from))

    @router.get("/{store_id}", response_model=DataStore)
    def get_store(request: Request,
                  store_id: Identifier) -> DataStore:
        return _view(request, record_or_404(request, store_id))

    @router.patch("/{store_id}", response_model=DataStore)
    def update_store(request: Request,
                     store_id: Identifier,
                     body: DataStoreUpdate) -> DataStore:
        """Rename a store, or change where a refresh takes its data from."""
        record = record_or_404(request, store_id)

        if body.refresh_from == "yfinance":
            if record["kind"] == "postgres":
                raise FindingsError("data store", "it cannot refresh from "
                                    "Yahoo Finance", [_yahoo_finding(
                                        "A database is read-only, so it "
                                        "cannot take downloaded data.")])

            _refuse_yahoo_for_synthetic(Path(record["path"]), "yfinance")

        updated = _registry(request).update(store_id, name=body.name,
                                            refresh_from=body.refresh_from)
        holder = active_data(request)

        if holder.store_id == store_id and updated is not None:
            holder.store_name = updated["name"]

        return _view(request, record_or_404(request, store_id))

    @router.delete("/{store_id}",
                   status_code=status.HTTP_204_NO_CONTENT,
                   responses=CONFLICT_RESPONSE)
    def forget_store(request: Request,
                     store_id: Identifier) -> Response:
        """Forget a store.

        A folder the user registered is left exactly as it is. A store the
        engine created (`managed`) has its folder deleted too, since nothing
        else knows it is there.
        """
        record = record_or_404(request, store_id)
        refuse_if_refreshing(request, record)

        if active_data(request).store_id == store_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This is the store the engine is serving. Load "
                       "another store first.")

        _registry(request).delete(store_id)

        if record.get("managed"):
            remove_managed_folder(request.app, Path(record["path"]))

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # async, as every job-submitting endpoint must be: the registry attaches
    # the job to the running event loop.
    @router.post("/{store_id}/activate",
                 response_model=LoadJobStatus,
                 status_code=status.HTTP_202_ACCEPTED,
                 responses=CONFLICT_RESPONSE)
    async def activate_store(request: Request,
                             store_id: Identifier) -> LoadJobStatus:
        record = record_or_404(request, store_id)
        refuse_if_refreshing(request, record)

        if not available(record):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"The store '{record['name']}' cannot be read: "
                       + ("its password variable is not set."
                          if record["kind"] == "postgres"
                          else f"nothing usable at {record['path']}."))

        claim_loading(request)
        jobs: JobRegistry = request.app.state.jobs
        job = jobs.submit(f"load:{store_id}",
                          load_store_job(request.app, record))

        return LoadJobStatus(**job.snapshot())

    return router


async def serve_store(app: FastAPI,
                      record: dict[str, Any],
                      report: ProgressReporter) -> LoadResult:
    """Load a store and start serving it: the one path for every load.

    Used by activation and by generation, so a store the engine just wrote is
    served exactly as one the user picked. The caller has set
    `active_data.loading`; this clears it however it ends. A failed load
    leaves the data already served in place.
    """
    holder: ActiveData = app.state.active_data

    try:
        await report(0.92, f"Reading '{record['name']}'")
        fetcher = await asyncio.to_thread(load, record)

        await report(0.98, "Serving the new data")
        registry: StoreRegistry = app.state.data_stores
        served = ActiveData(fetcher=fetcher,
                            store_id=record["id"],
                            store_name=record["name"])
        app.state.active_data = served
        registry.set_active(record["id"])
        registry.mark_loaded(record["id"])

        # GLOBAL describes the loaded dataset, so it follows the switch.
        seed_global_universe(app.state.universe_store, fetcher)

        jobs: JobRegistry = app.state.jobs
        jobs.publish_data_loaded(record["id"], record["name"],
                                 served.data_version)

        first, last = fetcher.date_range

        return LoadResult(store_id=record["id"],
                          name=record["name"],
                          identifiers=len(fetcher.identifiers),
                          start=f"{first:%Y-%m-%d}",
                          end=f"{last:%Y-%m-%d}")
    finally:
        app.state.active_data.loading = False
        holder.loading = False


def claim_loading(request: Request) -> None:
    """Mark a load as started, or refuse when one already is (409).

    Set before the job is scheduled, so a second request arriving before the
    job starts is refused rather than racing it.
    """
    holder = active_data(request)

    if holder.loading:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A data store is already loading. Wait for it to finish.")

    holder.loading = True


def load_store_job(app: FastAPI,
              record: dict[str, Any]) -> Any:
    """The coroutine that loads a store and starts serving it."""

    async def run(report: ProgressReporter) -> dict[str, Any]:
        return (await serve_store(app, record, report)).model_dump()

    return run
