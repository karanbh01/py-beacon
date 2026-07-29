# src/beacon/server/config.py
"""
Configuration for the Beacon API server.

The server is spawned and owned by a desktop client, so its configuration
arrives from the command line and the environment rather than from a file.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..data.fetcher import DataFetcher

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
        storage_root: Base directory for persisted documents. None uses the
            platform app-data location; tests point it at a temporary path.
    """
    auth_token: str
    host: str = "127.0.0.1"
    port: int = 0
    data_fetcher: DataFetcher | None = None
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
