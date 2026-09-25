# src/beacon/server/store_schemas.py
"""Request and response models for named data stores (BN-236).

Kept apart from `schemas.py`, which is far past the size this project keeps a
module to, and which these models do not need to live beside.
"""
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import Identifier, TolerantCollection

# What a store is, physically. Postgres joins in BN-241.
StoreKind = Literal["folder"]

# Where a store's rows came from, from the store's own manifest: the
# synthetic generator, a download, a file import, or a folder written some
# other way.
STORE_SOURCE_DESCRIPTION = (
    "Where the rows came from, as the store records it: 'synthetic' for "
    "generated data, 'yfinance' for downloaded prices, 'local' for a folder "
    "written another way. Null when the store cannot be read.")


class DataStoreCreate(BaseModel):
    """Body of `POST /data/stores`: register an existing store."""
    name: str = Field(min_length=1, max_length=80,
                      description="Display name, e.g. 'My data'. Also the "
                                  "basis of the store's id.")
    kind: StoreKind = Field(default="folder",
                            description="'folder' for a py-beacon data "
                                        "folder.")
    path: str = Field(min_length=1,
                      description="The folder, as an absolute path on the "
                                  "machine the engine runs on.")


class DataStoreUpdate(BaseModel):
    """Body of `PATCH /data/stores/{store_id}`."""
    name: str = Field(min_length=1, max_length=80,
                      description="The new display name. The id does not "
                                  "change.")


class DataStore(BaseModel):
    """One registered data store."""
    id: Identifier = Field(description="Stable id, from the name it was "
                                       "created with.")
    name: str = Field(description="Display name.")
    kind: StoreKind = Field(description="What the store is physically.")
    path: str = Field(description="Where the store is.")
    source: str | None = Field(default=None,
                               description=STORE_SOURCE_DESCRIPTION)
    size_bytes: int | None = Field(
        default=None,
        description="Size of the store's files. Null when it cannot be read.")
    readable: bool = Field(
        description="Whether the store can be read now. A folder that has "
                    "been moved or deleted stays registered but is not "
                    "readable.")
    active: bool = Field(description="Whether this is the store the engine "
                                     "is serving.")
    created_at: str = Field(description="When it was registered, ISO 8601 "
                                        "UTC.")
    last_loaded_at: str | None = Field(
        default=None,
        description="When the engine last loaded it, ISO 8601 UTC. Null if "
                    "never.")


class DataStoreCollection(TolerantCollection):
    """Response of `GET /data/stores`."""
    stores: list[DataStore] = Field(description="Every readable registered "
                                                "store, in name order.")
    active: Identifier | None = Field(
        default=None,
        description="The id of the store being served, or null when none is, "
                    "or when the data came from `--data` or "
                    "`BEACON_DATA_PATH` rather than a registered store.")
