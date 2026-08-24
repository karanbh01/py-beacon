# src/beacon/data/__init__.py
"""
The __init__.py for the 'data' module.

This module handles fetching, parsing, and providing financial data.
"""
from typing import Any

from .base import MarketData, ReferenceData
from .fetcher import DataFetcher
from .loader import load_data

__all__ = [
    "DataFetcher",
    "MarketData",
    "ReferenceData",
    "load_data",
]

# The expression root is spelled `data` too, and `from beacon import data`
# reaches *this package* rather than that object -- importing any submodule
# rebinds the name on the parent, so which one a caller got would depend on
# import order.
#
# The expression root therefore lives in `beacon.expressions`, and this points
# anybody who guessed the other import at the right one. Without it the error
# is "module 'beacon.data' has no attribute 'market'", which is true and tells
# nobody what to do about it.
_EXPRESSION_ROOTS = ("market", "reference", "actions")


def __getattr__(name: str) -> Any:
    if name in _EXPRESSION_ROOTS:
        raise AttributeError(
            f"'beacon.data' is the data package, not the expression root, so "
            f"it has no '{name}'. For `data.{name}...` in a screen, import it "
            "as `from beacon.expressions import data`.")

    raise AttributeError(f"module 'beacon.data' has no attribute '{name}'")
