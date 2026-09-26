# src/beacon/server/store_schemas.py
"""Request and response models for named data stores."""
# Added with BN-236. Kept apart from `schemas.py`, which is far past the size
# this project keeps a module to, and which these models do not need to live
# beside.
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import Identifier, IsoDate, LoadJobStatus, TolerantCollection

# What a store is, physically: a py-beacon data folder, or a Postgres
# database read through tables or views in the import layout.
StoreKind = Literal["folder", "postgres"]

# Where a refresh takes a store's new data from: its own source, or, for a
# folder that chooses it, Yahoo Finance.
RefreshSource = Literal["source", "yfinance"]

# What refreshing a store does, by what the store is.
RefreshAction = Literal["extend", "reread", "download"]

REFRESH_FROM_DESCRIPTION = (
    "Where a refresh takes new data from. 'source' (the default) uses the "
    "store's own source: synthetic data is extended to today, a folder or "
    "database is read again. 'yfinance' downloads new prices for the store's "
    "instruments from Yahoo Finance and saves them into its folder; it needs "
    "the `data` extra, and only a folder store that is not synthetic data "
    "can choose it.")


class PostgresConnection(BaseModel):
    """Where a Postgres store's tables or views are. Read-only."""
    host: str = Field(min_length=1, description="The database server.")
    port: int = Field(default=5432, ge=1, le=65535,
                      description="The server's port.")
    database: str = Field(min_length=1, description="The database name.")
    schema_name: str = Field(
        default="public", alias="schema", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description="The schema holding tables or views named market, "
                    "reference, and optionally fx, corporate_actions and "
                    "features, with the import template's columns.")
    user: str = Field(min_length=1,
                      description="The user to connect as. Read access is "
                                  "all it needs.")
    password_env: str | None = Field(
        default=None,
        description="The environment variable holding the password, read "
                    "each time the engine connects. The password itself is "
                    "never sent to the engine or saved. Null when the server "
                    "needs none.")

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)

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
                                        "folder, 'postgres' for a database.")
    path: str | None = Field(
        default=None, min_length=1,
        description="For a folder: its absolute path on the machine the "
                    "engine runs on.")
    connection: PostgresConnection | None = Field(
        default=None,
        description="For a Postgres database: where to connect.")
    refresh_from: RefreshSource = Field(default="source",
                                        description=REFRESH_FROM_DESCRIPTION)

    @model_validator(mode="after")
    def _one_location(self) -> "DataStoreCreate":
        """A folder needs a path, a database a connection."""
        if self.kind == "folder" and self.path is None:
            raise ValueError("a folder store needs a path")

        if self.kind == "postgres" and self.connection is None:
            raise ValueError("a postgres store needs a connection")

        if self.kind == "postgres" and self.refresh_from == "yfinance":
            raise ValueError("a database is read-only, so it cannot refresh "
                             "from Yahoo Finance")

        return self


class DataStoreUpdate(BaseModel):
    """Body of `PATCH /data/stores/{store_id}`. Omitted fields are unchanged."""
    name: str | None = Field(default=None, min_length=1, max_length=80,
                             description="The new display name. The id does "
                                         "not change.")
    refresh_from: RefreshSource | None = Field(
        default=None, description=REFRESH_FROM_DESCRIPTION)


class RefreshRequest(BaseModel):
    """Body of `POST /data/stores/{store_id}/refresh`. Optional."""
    end: str | None = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="For synthetic data, the date to extend to, YYYY-MM-DD. "
                    "Defaults to today. Ignored by other stores.")


class DataStore(BaseModel):
    """One registered data store."""
    id: Identifier = Field(description="Stable id, from the name it was "
                                       "created with.")
    name: str = Field(description="Display name.")
    kind: StoreKind = Field(description="What the store is physically.")
    path: str = Field(description="Where the store is: a folder, or a "
                                  "database described without its "
                                  "password.")
    connection: PostgresConnection | None = Field(
        default=None,
        description="For a Postgres store, its connection details.")
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
    refresh_from: RefreshSource = Field(default="source",
                                        description=REFRESH_FROM_DESCRIPTION)
    refresh: RefreshAction | None = Field(
        default=None,
        description="What `POST /data/stores/{id}/refresh` would do: 'extend' "
                    "synthetic data to today, 'reread' a folder or database, "
                    "or 'download' from Yahoo Finance. Null when there is "
                    "nothing to refresh: imported files (import again "
                    "instead), synthetic data generated before py-beacon "
                    "0.2.0, or a store that cannot be read.")


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

