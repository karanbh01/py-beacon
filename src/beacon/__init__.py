# src/beacon/__init__.py
"""
py-beacon, an end-to-end toolkit for index, ETF, and Delta-1 derivatives
development.
"""
__version__ = "0.1.1"

from . import derivatives
from .derivatives import (
    DerivativeBase,
    ETFFuture,
    IndexFuture,
    TotalReturnSwap,
)
from .sources import use

__all__ = [
    "DerivativeBase",
    "ETFFuture",
    "IndexFuture",
    "TotalReturnSwap",
    "__version__",
    "derivatives",
    "use",
]
