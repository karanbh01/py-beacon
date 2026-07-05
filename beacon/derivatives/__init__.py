# beacon/derivatives/__init__.py
"""
The 'derivatives' package models exchange-traded and OTC Delta-1 derivatives
that reference beacon indices, ETFs, and equities.
"""
from .base import DerivativeBase
from .futures import IndexFuture, ETFFuture
from .swaps import TotalReturnSwap
from .pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    implied_repo_rate,
    futures_roll_return,
    trs_breakeven_spread,
)

__all__ = [
    "DerivativeBase",
    "IndexFuture",
    "ETFFuture",
    "TotalReturnSwap",
    "cost_of_carry_fair_value",
    "discrete_dividend_fair_value",
    "implied_repo_rate",
    "futures_roll_return",
    "trs_breakeven_spread",
]
