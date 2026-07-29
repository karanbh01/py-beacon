# src/beacon/__init__.py
"""
Beacon — an end-to-end toolkit for index, ETF, and Delta-1 derivatives
development.
"""
__version__ = "0.0.2"

from . import derivatives
from .derivatives import (
    DerivativeBase,
    IndexFuture,
    ETFFuture,
    TotalReturnSwap,
)

__all__ = [
    "__version__",
    "derivatives",
    "DerivativeBase",
    "IndexFuture",
    "ETFFuture",
    "TotalReturnSwap",
]
