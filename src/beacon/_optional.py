# src/beacon/_optional.py
"""
Import guards for py-beacon's optional dependencies.

A plain install does everything but serve the API and download market data:
indices, backtests, funds and derivatives, charts and interactive charts, PDF
and Excel reports, the optimiser and Postgres stores. The API server and Yahoo
Finance downloads live behind the `server` and `data` extras. Modules needing
one import it through :func:`require`, so a missing package reports the extra
to install instead of surfacing a bare ImportError.
"""
# `exchange_calendars` moved from the `calendars` extra into the core in
# BN-180; charts, reports, Excel, the optimiser, Postgres stores and
# platformdirs followed in BN-310.
import importlib
from types import ModuleType

from .exceptions import MissingDependencyError

# Optional module -> the pyproject.toml extra that provides it. Every module
# passed to require() must appear here; a missing entry is a packaging bug.
EXTRA_FOR_MODULE = {
    "fastapi": "server",
    "orjson": "server",
    "uvicorn": "server",
    "websockets": "server",
    "yfinance": "data",
}


def require(module_name: str,
            feature: str) -> ModuleType:
    """Import an optional dependency, or raise an actionable error.

    Args:
        module_name: Module to import, e.g. ``"yfinance"``.
        feature: Human-readable name of the py-beacon feature that needs it,
            used to open the error message, e.g. ``"Market-data ingestion"``.

    Returns:
        The imported module.

    Raises:
        MissingDependencyError: If the module is not installed. The message
            names the extra to install.
        KeyError: If module_name is not a registered optional dependency.
    """
    extra = EXTRA_FOR_MODULE[module_name]

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(module_name, feature, extra) from exc
