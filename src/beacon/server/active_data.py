# src/beacon/server/active_data.py
"""The data the engine is serving right now, and the one rule for needing it.

BN-236. The engine used to hold its data in `ServerConfig.data_fetcher`,
fixed when the process started: a server started without data stayed without
it. Now a data store can be loaded, or switched, while the engine runs, so the
current data lives here, on app state, and every reader asks this module.

Seven routers each had a copy of the same check ("started without a data
source"), each answering 500. That is one rule, so it is one function here,
and it answers 409: no data being loaded is a state the caller can change by
loading a store, not a server fault.
"""
from dataclasses import dataclass

from .._optional import require
from ..data.fetcher import DataFetcher
from ..exceptions import NoDataLoadedError

require("fastapi", "The Beacon API server")

from fastapi import Request  # noqa: E402


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
    """
    fetcher: DataFetcher | None = None
    store_id: str | None = None
    store_name: str | None = None
    loading: bool = False


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
