# src/beacon/server/store_schemas.py
"""Request and response models for named data stores (BN-236).

Kept apart from `schemas.py`, which is far past the size this project keeps a
module to, and which these models do not need to live beside.
"""
from typing import Literal

from pydantic import BaseModel, Field

from .schemas import Identifier, IsoDate, LoadJobStatus, TolerantCollection

# What a store is, physically. Postgres joins in BN-241.
StoreKind = Literal["folder"]

# Where a store's rows came from, from the store's own manifest: the
# synthetic generator, a download, a file import, or a folder written some
# other way.
STORE_SOURCE_DESCRIPTION = (
    "Where the rows came from, as the store records it: 'synthetic' for "
    "generated data, 'imported' for data loaded from files, 'yfinance' for "
    "downloaded prices, 'local' for a folder written another way. Null when "
    "the store cannot be read.")


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
    managed: bool = Field(
        default=False,
        description="True for a store the engine created, such as generated "
                    "synthetic data, in its own folder. Deleting a managed "
                    "store deletes its files; deleting any other store only "
                    "forgets it and leaves the folder alone.")


class DataStoreCollection(TolerantCollection):
    """Response of `GET /data/stores`."""
    stores: list[DataStore] = Field(description="Every readable registered "
                                                "store, in name order.")
    active: Identifier | None = Field(
        default=None,
        description="The id of the store being served, or null when none is, "
                    "or when the data came from `--data` or "
                    "`BEACON_DATA_PATH` rather than a registered store.")


class GenerateSyntheticRequest(BaseModel):
    """Body of `POST /data/synthetic`: generate a synthetic data store.

    Every field is optional. Anything left out takes the same default as
    `python -m beacon.synthetic`, because the engine runs that command: the
    same settings always give the same data, whichever way it was made.
    """
    name: str = Field(default="Synthetic data", min_length=1, max_length=80,
                      description="Display name for the new store.")
    assets: int | None = Field(
        default=None, ge=1, le=50_000,
        description="How many names. Default 6,000, or 10,000 with "
                    "`extended_universe`.")
    extended_universe: bool = Field(
        default=False,
        description="10,000 names instead of 6,000. Ignored when `assets` is "
                    "given.")
    start: IsoDate | None = Field(
        default=None,
        description="First date. Default ten years before `end`, or the "
                    "start of the crises the generator models with "
                    "`long_history`.")
    end: IsoDate | None = Field(default=None,
                                description="Last date. Default today.")
    long_history: bool = Field(
        default=False,
        description="Reach back to cover every crisis the generator models. "
                    "Ignored when `start` is given.")
    seed: int | None = Field(
        default=None, ge=0,
        description="Everything random is drawn from this, so the same seed "
                    "and settings always give the same data.")
    risk_free_rate: float | None = Field(
        default=None, description="Annualised, as a decimal.")
    equity_premium: float | None = Field(
        default=None,
        description="Annualised excess return on a beta-one name, as a "
                    "decimal.")
    calendar: str | None = Field(
        default=None,
        description="Exchange code whose trading days the data has prices "
                    "on. Default XNYS.")
    features: bool = Field(
        default=True,
        description="Include fundamental ratios and alternative data.")
    activate: bool = Field(
        default=True,
        description="Serve the new store as soon as it is written.")


class ImportRequest(BaseModel):
    """Body of `POST /data/import`: load CSV files or an Excel workbook."""
    name: str = Field(default="Imported data", min_length=1, max_length=80,
                      description="Display name for the new store.")
    paths: list[str] = Field(
        min_length=1,
        description="Absolute paths on the machine the engine runs on: CSV "
                    "files named after their sheet (market.csv, "
                    "reference.csv, ...), or one Excel workbook with those "
                    "sheets. `GET /data/import/template` has the layout.")
    activate: bool = Field(default=True,
                           description="Load the new store as soon as it is "
                                       "saved.")


class ImportResult(BaseModel):
    """Response of `POST /data/import`."""
    store: DataStore = Field(description="The new store, saved and "
                                         "registered.")
    load_job: LoadJobStatus | None = Field(
        default=None,
        description="The job loading it, when `activate` was set and no "
                    "other load was running. Null otherwise: the store is "
                    "saved and can be activated later.")

