# src/beacon/server/__init__.py
"""
The local Beacon API server.

Requires the ``server`` extra:

    pip install "py-beacon[server]"

Importing this subpackage pulls in FastAPI, so the guard fires here and names
the extra rather than letting a bare ImportError surface. The rest of Beacon
stays importable without it.
"""
from .app import create_app
from .config import TOKEN_ENV_VAR, ServerConfig
from .errors import classify, register_exception_handlers
from .schemas import (
    BacktestResultSummary,
    ErrorEnvelope,
    HealthResponse,
    IndexResultSummary,
    Money,
    SeriesPayload,
    TableFrame,
)
from .serialisation import dataframe_to_payload, series_to_payload

__all__ = [
    "TOKEN_ENV_VAR",
    "BacktestResultSummary",
    "ErrorEnvelope",
    "HealthResponse",
    "IndexResultSummary",
    "Money",
    "SeriesPayload",
    "ServerConfig",
    "TableFrame",
    "classify",
    "create_app",
    "dataframe_to_payload",
    "register_exception_handlers",
    "series_to_payload",
]
