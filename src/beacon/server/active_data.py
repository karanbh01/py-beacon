# src/beacon/server/active_data.py
"""The data the engine is serving right now, and the one rule for needing it.

A data store can be loaded, or switched, while the engine runs, so the current
data lives here, on app state, and every reader asks this module.

A route that needs data and finds none answers 409: no data being loaded is a
state the caller can change by loading a store, not a server fault.
"""
# BN-236. The engine used to hold its data in `ServerConfig.data_fetcher`, fixed
# when the process started: a server started without data stayed without it.
# Seven routers each had a copy of the same check ("started without a data
# source"), each answering 500. That is one rule, so it is one function here.
import uuid
from dataclasses import dataclass, field

from .._optional import require
from ..data.fetcher import DataFetcher
from ..exceptions import NoDataLoadedError

require("fastapi", "The Beacon API server")

from fastapi import Request  # noqa: E402


def new_data_version() -> str:
    """A token no earlier data, in this process or any before it, can have."""
    return uuid.uuid4().hex


@dataclass
class ActiveData:
    """What the engine serves: the loaded data and the store it came from.

    Replaced whole when a store is loaded, never edited in place, so a request
    that has already read `fetcher` keeps a consistent dataset for its whole
    run even if another store is loaded meanwhile.

    Attributes:
        fetcher: The loaded data, or None when nothing is loaded.
        store_id: The registered store it came from. None for data given on
            the command line or by `BEACON_DATA_PATH`, which is not a
            registered store, and when nothing is loaded.
        store_name: A name to show for it: the store's name, or where the
            unregistered data came from.
        loading: Whether a store is being loaded right now. One load at a
            time: a second would race the first to replace the data.
        data_version: An opaque token that changes whenever the data being
            served changes: at startup, and on every load (the same store
            loaded again included, since its files may have changed, and a
            refresh of the served store, which reloads it). A client
            compares it for equality only, to know whether what it cached
            is still current. Random rather than a counter, so an engine
            restart can never bring an old value back.
    """
    fetcher: DataFetcher | None = None
    store_id: str | None = None
    store_name: str | None = None
    loading: bool = False
    data_version: str = field(default_factory=new_data_version)

    def data_changed(self) -> str:
        """Mint a new `data_version` after the data changed in place."""
        self.data_version = new_data_version()

        return self.data_version



def active_data(request: Request) -> ActiveData:
    """The engine's current data holder."""
    holder: ActiveData = request.app.state.active_data

    return holder


def current_data(request: Request) -> DataFetcher | None:
    """The loaded data, or None, for a reader that works either way."""
    return active_data(request).fetcher


def require_data(request: Request,
                 purpose: str) -> DataFetcher:
    """The loaded data, or a refusal saying what could not be done.

    Args:
        request: The incoming request.
        purpose: What needs the data, completing "No data is loaded, so ...",
            e.g. "a backtest cannot be run".

    Raises:
        NoDataLoadedError: When nothing is loaded. Maps to 409 NO_DATA_LOADED.
    """
    fetcher = current_data(request)

    if fetcher is None:
        raise NoDataLoadedError(purpose)

    return fetcher
