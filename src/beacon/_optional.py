# src/beacon/_optional.py
"""
Import guards for py-beacon's optional dependencies.

The core pipeline (index, backtest, portfolio, fund, derivatives) runs on
pandas, numpy, pydantic and `exchange_calendars` alone. `exchange_calendars`
is core rather than an extra because every index schedules against a real
trading calendar, and a required input cannot sit behind an optional install.
Everything beyond that (Excel reporting, plotting, optimisation, market-data
downloads, the API server) lives behind an extra.
Modules needing one import it through :func:`require`, so a missing package
reports the extra to install instead of surfacing a bare ImportError.
"""
# `exchange_calendars` moved from the `calendars` extra into the core in
# BN-180.
import importlib
from types import ModuleType

from .exceptions import MissingDependencyError

# Optional module -> the pyproject.toml extra that provides it. Every module
# passed to require() must appear here; a missing entry is a packaging bug.
EXTRA_FOR_MODULE = {
    "fastapi": "server",
    "matplotlib": "plot",
    "openpyxl": "excel",
    "orjson": "server",
    "platformdirs": "server",
    "plotly": "plot-interactive",
    "psycopg": "postgres",
    "reportlab": "pdf",
    "scipy": "optimise",
    "uvicorn": "server",
    "websockets": "server",
    "yfinance": "data",
}


def require(module_name: str,
            feature: str) -> ModuleType:
    """Import an optional dependency, or raise an actionable error.

    Args:
        module_name: Module to import, e.g. ``"openpyxl"``.
        feature: Human-readable name of the py-beacon feature that needs it,
            used to open the error message, e.g. ``"Excel reporting"``.

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
