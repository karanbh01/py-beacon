# src/beacon/server/data_stores.py
"""Named data stores: which exist, which one the engine serves, and loading it.

BN-236. A data store holds one dataset. Users name their stores ("Synthetic
data", "My data"), one is active at a time, and the engine remembers which,
so the next start serves the same data. A store is a py-beacon data folder
today; Postgres joins in BN-241.

The registry is saved with the engine's other documents (indices, universes),
so it lives wherever `--documents` points.

## What the engine serves at startup

In order:

1. ``--data <path>``: a folder named on the command line.
2. ``$BEACON_DATA_PATH``.
3. The active registered store.
4. If nothing is registered yet but a store exists in the default app-data
   folder, it is registered as "Synthetic data" (or by its source) and made
   active. This is how an install from before named stores keeps working.
5. Nothing: the engine starts empty, and a store can be loaded later.

The first two fail loudly, because naming data that cannot be read is a
mistake worth stopping for. The active store only warns and starts empty, so
a store that was moved or damaged never stops the engine from starting and
offering another.
"""
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..data import store as data_store
from ..data.fetcher import DataFetcher
from ..exceptions import ConfigurationError
from .config import resolve_data_source
from .documents import UNREADABLE, SkipCounts, read_collection, slug
from .store import DocumentStore
from .store_schemas import DataStore

logger = logging.getLogger(__name__)

COLLECTION = "data_stores"

# The name an existing default store is registered under when nothing is
# registered yet, by the source its manifest records.
NAME_FOR_SOURCE = {"synthetic": "Synthetic data"}
DEFAULT_NAME = "My data"


class StoreRecord(BaseModel):
    """A registered store, as saved. Which store is active is saved here too,
    as a flag on its record, so the registry is one collection with nothing
    beside it to fall out of step."""
    name: str
    kind: str = "folder"
    path: str
    created_at: str
    last_loaded_at: str | None = None
    active: bool = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _as_record(document_id: str,
               document: dict[str, Any]) -> dict[str, Any]:
    """A saved store as a plain dict with its id, validated on the way."""
    return {**StoreRecord.model_validate(document).model_dump(),
            "id": document_id}


class StoreRegistry:
    """The registered data stores and which one is active.

    Args:
        root: Where the engine keeps its documents. None uses the platform
            app-data location, as every other collection does.
    """

    def __init__(self,
                 root: Path | None = None):
        self._stores = DocumentStore(COLLECTION, root=root)

    def listing(self) -> tuple[list[dict[str, Any]], SkipCounts]:
        """Every readable store in name order, and what was skipped and why."""
        found, skipped = read_collection(self._stores, _as_record, "data store")

        return sorted(found, key=lambda record: record["name"].lower()), skipped

    def records(self) -> list[dict[str, Any]]:
        """Every readable store, in name order."""
        return self.listing()[0]

    def get(self,
            store_id: str) -> dict[str, Any] | None:
        """One store's record, or None when there is none or it can't be read."""
        try:
            document = self._stores.read(store_id)

            return _as_record(store_id, document) if document is not None else None
        except UNREADABLE as error:
            logger.warning("Skipping data store %s: %s", store_id, error)

            return None

    def find_by_path(self,
                     path: Path) -> dict[str, Any] | None:
        """The store registered for this folder, if any."""
        wanted = path.resolve()

        for record in self.records():
            if Path(record["path"]).resolve() == wanted:
                return record

        return None

    def create(self,
               name: str,
               path: Path,
               kind: str = "folder") -> dict[str, Any]:
        """Register a store, with an id derived from its name.

        A second store with the same name gets a numbered id ("my-data-2"),
        so names never have to be unique, only ids.
        """
        base = slug(name) or "store"
        store_id = base
        suffix = 2

        while self._stores.exists(store_id):
            store_id = f"{base}-{suffix}"
            suffix += 1

        record = {**StoreRecord(name=name.strip(),
                                kind=kind,
                                path=str(path.resolve()),
                                created_at=_now()).model_dump(),
                  "id": store_id}
        self._write(record)

        return record

    def rename(self,
               store_id: str,
               name: str) -> dict[str, Any] | None:
        """Change a store's display name. Its id stays as it was."""
        record = self.get(store_id)

        if record is None:
            return None

        record["name"] = name.strip()
        self._write(record)

        return record

    def delete(self,
               store_id: str) -> None:
        """Forget a store. Its folder is left exactly as it is."""
        self._stores.delete(store_id)

    def active_id(self) -> str | None:
        """The store the engine last served, or None.

        If more than one record is flagged, as a crash between two writes in
        `set_active` could leave it, the most recently loaded one wins.
        """
        active = [record for record in self.records() if record["active"]]

        if not active:
            return None

        latest = max(active, key=lambda record: record["last_loaded_at"] or "")

        return str(latest["id"])

    def set_active(self,
                   store_id: str | None) -> None:
        """Remember which store to serve, including across restarts.

        The new store is flagged before the old one is cleared, so a crash in
        between leaves two flagged rather than none, and `active_id` resolves
        that.
        """
        records = self.records()

        for record in records:
            if record["id"] == store_id and not record["active"]:
                record["active"] = True
                self._write(record)

        for record in records:
            if record["id"] != store_id and record["active"]:
                record["active"] = False
                self._write(record)

    def mark_loaded(self,
                    store_id: str) -> None:
        """Record that a store was just loaded."""
        record = self.get(store_id)

        if record is not None:
            record["last_loaded_at"] = _now()
            self._write(record)

    def _write(self,
               record: dict[str, Any]) -> None:
        body = StoreRecord.model_validate(record).model_dump()
        self._stores.write(record["id"], body)


def load(record: dict[str, Any],
         **settings: Any) -> DataFetcher:
    """Read a registered store into a fetcher.

    Args:
        record: The store's registry record.
        **settings: Passed to the loader: `fx_policy`,
            `max_price_staleness_days`, `free_float_backfill_days`.

    Raises:
        ConfigurationError: If the store cannot be read.
    """
    return data_store.load(Path(record["path"]), **settings)


def describe(record: dict[str, Any],
             active_id: str | None) -> DataStore:
    """A store's listing row, read from its folder without loading it."""
    path = Path(record["path"])
    readable = data_store.exists(path)
    source: str | None = None
    size: int | None = None

    if readable:
        try:
            source = data_store.read_manifest(path).source
        except ConfigurationError:
            readable = False

    if readable:
        size = sum(item.stat().st_size for item in path.iterdir()
                   if item.is_file())

    return DataStore(id=record["id"],
                     name=record["name"],
                     kind=record.get("kind", "folder"),
                     path=str(path),
                     source=source,
                     size_bytes=size,
                     readable=readable,
                     active=record["id"] == active_id,
                     created_at=record["created_at"],
                     last_loaded_at=record.get("last_loaded_at"))


@dataclass(frozen=True)
class StartupData:
    """What the engine serves when it starts, and a line saying why."""
    fetcher: DataFetcher | None
    store_id: str | None
    store_name: str | None
    origin: str


def resolve_startup(explicit: Path | None,
                    root: Path | None) -> StartupData:
    """Find the data to serve at startup. See the module docstring for the order.

    Args:
        explicit: ``--data`` from the command line, or None.
        root: Where the engine keeps its documents (``--documents``).

    Raises:
        ConfigurationError: If ``--data`` or ``$BEACON_DATA_PATH`` names data
            that cannot be read.
    """
    named = os.environ.get(data_store.DATA_PATH_ENV_VAR, "").strip()

    if explicit is not None or named:
        fetcher, origin = resolve_data_source(explicit)

        return StartupData(fetcher, None, origin, origin)

    registry = StoreRegistry(root)
    active = registry.active_id()

    if active is None:
        active = _adopt_default_store(registry)

    if active is None:
        return StartupData(None, None, None, "no data loaded")

    record = registry.get(active)

    if record is None:
        logger.warning("The active data store %s is no longer registered; "
                       "starting with no data.", active)

        return StartupData(None, None, None, "no data loaded")

    try:
        fetcher = load(record)
    except ConfigurationError as error:
        logger.warning("The data store '%s' could not be loaded (%s). "
                       "Starting with no data, so another can be loaded.",
                       record["name"], error)

        return StartupData(None, None, None,
                           f"no data loaded: '{record['name']}' is unreadable")

    registry.mark_loaded(active)

    return StartupData(fetcher, active, record["name"],
                       f"the data store '{record['name']}' ({record['path']})")


def _adopt_default_store(registry: StoreRegistry) -> str | None:
    """Register the default app-data store, for installs from before BN-236.

    Only when nothing is registered, so a user who removed every store is
    not surprised by one reappearing.
    """
    if registry.records():
        return None

    path = data_store.default_path()

    if not data_store.exists(path):
        return None

    try:
        source = data_store.read_manifest(path).source
    except ConfigurationError:
        source = ""

    record = registry.create(NAME_FOR_SOURCE.get(source, DEFAULT_NAME), path)
    registry.set_active(record["id"])
    logger.info("Registered the existing data store at %s as '%s'.",
                path, record["name"])

    return str(record["id"])
