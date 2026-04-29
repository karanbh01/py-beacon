"""Derivative instruments."""

from .base import DerivativeBase
from .swaps import TotalReturnSwap

__all__ = ["DerivativeBase", "TotalReturnSwap"]
