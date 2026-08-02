# src/beacon/server/config.py
"""
Configuration for the Beacon API server.

The server is spawned and owned by a desktop client, so its configuration
arrives from the command line and the environment rather than from a file.
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..data import store
from ..data.fetcher import DataFetcher
from ..data.ingest import Downloader
from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Environment variable consulted when no token is passed on the command line.
TOKEN_ENV_VAR = "BEACON_API_TOKEN"

# The desktop client is served from a custom scheme; the dev build runs on an
# arbitrary localhost port, hence the regex rather than a fixed list.
DEFAULT_CORS_ORIGINS = ("app://",)
LOCALHOST_ORIGIN_PATTERN = r"^http://localhost(:\d+)?$"


@dataclass(frozen=True)
class ServerConfig:
    """Settings for a single server process.

    Attributes:
        auth_token: Bearer token required on every request. Never empty — a
            server with no token would be open to any process on the machine.
        host: Interface to bind. Defaults to loopback and should stay there:
            the server has no transport security and trusts its bearer token
            alone.
        port: Port to bind. 0 asks the OS for a free one, which the launcher
            then reads back and prints.
        data_fetcher: The data source to serve, or None to run without one.
        cors_origins: Exact origins allowed, in addition to the localhost
            pattern.
        market_downloader: Where a sync fetches market data from. None builds
            the yfinance-backed downloader on first use, which is the real
            deployment; tests and offline runs inject their own so the sync
            path is exercisable without a network.
        storage_root: Base directory for persisted documents. None uses the
            platform app-data location; tests point it at a temporary path.
    """
    auth_token: str
    host: str = "127.0.0.1"
    port: int = 0
    data_fetcher: DataFetcher | None = None
    market_downloader: Downloader | None = None
    cors_origins: tuple[str, ...] = field(default=DEFAULT_CORS_ORIGINS)
    storage_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.auth_token:
            raise ValueError(
                "auth_token cannot be empty: every route requires a bearer token.")
        if not 0 <= self.port <= 65535:
            raise ValueError(f"port must be between 0 and 65535, got {self.port}.")

    @classmethod
    def from_environment(cls,
                         token: str | None = None,
                         host: str = "127.0.0.1",
                         port: int = 0,
                         data_fetcher: DataFetcher | None = None) -> "ServerConfig":
        """Build a config, taking the token from the environment if not given.

        Args:
            token: Explicit token, typically from the command line. When None,
                the value of ``BEACON_API_TOKEN`` is used.
            host: Interface to bind.
            port: Port to bind; 0 asks the OS for a free one.
            data_fetcher: Data source to serve, or None.

        Returns:
            ServerConfig: The assembled configuration.

        Raises:
            ValueError: If no token is available from either source.
        """
        resolved = token or os.environ.get(TOKEN_ENV_VAR, "")
        if not resolved:
            raise ValueError(
                f"No auth token supplied. Pass --token or set {TOKEN_ENV_VAR}.")

        return cls(auth_token=resolved,
                   host=host,
                   port=port,
                   data_fetcher=data_fetcher)


def resolve_data_source(explicit: Path | None = None) -> tuple[DataFetcher | None, str]:
    """Find the data source a spawned server should serve.

    In order:

    1. ``--data <path>``, passed here as ``explicit``
    2. ``$BEACON_DATA_PATH``
    3. the app-data store, if one has been written there
    4. nothing — the server starts data-less, as it always did

    The two explicit branches fail loudly: asking for a store that cannot be
    read is a mistake worth stopping for, and starting data-less instead would
    turn it into a puzzle about why every endpoint returns
    ``CONFIGURATION_ERROR``. The auto-load branch does the opposite and only
    warns, because a corrupt app-data store must not leave the client unable to
    start the server that would let it write a new one.

    Args:
        explicit: Path from the command line, or None.

    Returns:
        tuple: The fetcher (or None), and a sentence naming the branch that
        ran, for the caller to log. Which branch ran is the first thing anyone
        debugging an empty client will want to know.
    """
    if explicit is not None:
        return store.load(explicit), f"--data {explicit}"

    from_environment = os.environ.get(store.DATA_PATH_ENV_VAR, "").strip()
    if from_environment:
        path = Path(from_environment)

        return store.load(path), f"${store.DATA_PATH_ENV_VAR} ({path})"

    auto = store.default_path()
    if store.exists(auto):
        try:
            return store.load(auto), f"the app-data store ({auto})"
        except ConfigurationError as exc:
            logger.warning(
                "The app-data store at %s could not be read (%s). Starting "
                "without a data source so a sync can replace it.", auto, exc)

            return None, f"no data source: the store at {auto} is unreadable"

    return None, f"no data source (no store at {auto})"
