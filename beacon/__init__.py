# beacon/__init__.py
"""
Beacon — an end-to-end toolkit for index, ETF, and Delta-1 derivatives
development.
"""
from . import derivatives
from .derivatives import (
    DerivativeBase,
    IndexFuture,
    ETFFuture,
    TotalReturnSwap,
)

__all__ = [
    "derivatives",
    "DerivativeBase",
    "IndexFuture",
    "ETFFuture",
    "TotalReturnSwap",
]
