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
#
# `beacon://app` is the packaged renderer's actual origin. The default used to
# be `app://` alone — a scheme with no host, which no renderer ever sends, so
# every direct call from the packaged app failed preflight. `app://` is kept
# beside it rather than deleted: it costs nothing, and silently removing an
# origin from a default is how a working build stops working with no message
# that says why. Drop it once nothing is confirmed to send it.
PACKAGED_APP_ORIGIN = "beacon://app"
LEGACY_APP_ORIGIN = "app://"
DEFAULT_CORS_ORIGINS = (PACKAGED_APP_ORIGIN, LEGACY_APP_ORIGIN)
LOCALHOST_ORIGIN_PATTERN = r"^http://localhost(:\d+)?$"

# Comma-separated origins, consulted when none are passed on the command line.
CORS_ORIGINS_ENV_VAR = "BEACON_CORS_ORIGINS"


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
                         data_fetcher: DataFetcher | None = None,
                         cors_origins: tuple[str, ...] | None = None
                         ) -> "ServerConfig":
        """Build a config, taking the token from the environment if not given.

        Args:
            token: Explicit token, typically from the command line. When None,
                the value of ``BEACON_API_TOKEN`` is used.
            host: Interface to bind.
            port: Port to bind; 0 asks the OS for a free one.
            data_fetcher: Data source to serve, or None.
            cors_origins: Exact origins to allow. None resolves them from the
                environment and the defaults.

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
                   data_fetcher=data_fetcher,
                   cors_origins=(cors_origins if cors_origins is not None
                                 else resolve_cors_origins()))


def resolve_cors_origins(explicit: list[str] | None = None) -> tuple[str, ...]:
    """Find the exact origins this server should allow.

    In order: ``--cors-origin`` (repeatable), then ``$BEACON_CORS_ORIGINS``
    as a comma-separated list, then the defaults.

    Explicit origins **replace** the defaults rather than adding to them. An
    operator narrowing what may call the server should not find two extra
    origins still permitted — that is the opposite of what configuring it
    means. The localhost pattern is applied separately by the middleware and
    is unaffected either way.

    Args:
        explicit: Origins from the command line, or None.

    Returns:
        tuple: Origins, in the order given, with duplicates removed.
    """
    supplied = list(explicit or [])

    if not supplied:
        raw = os.environ.get(CORS_ORIGINS_ENV_VAR, "")
        supplied = [part.strip() for part in raw.split(",") if part.strip()]

    if not supplied:
        return DEFAULT_CORS_ORIGINS

    return tuple(dict.fromkeys(supplied))


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
