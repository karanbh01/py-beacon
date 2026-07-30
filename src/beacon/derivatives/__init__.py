# src/beacon/derivatives/__init__.py
"""
The 'derivatives' package models exchange-traded and OTC Delta-1 derivatives
that reference beacon indices, ETFs, and equities.
"""
from .base import DerivativeBase
from .curves import BASIS_POINT, RateCurve
from .futures import ETFFuture, IndexFuture
from .pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    futures_roll_return,
    implied_repo_rate,
    trs_breakeven_spread,
)
from .swaps import TotalReturnSwap
from .term_structure import FuturesQuote, TermStructure, sensitivity_grid

__all__ = [
    "BASIS_POINT",
    "DerivativeBase",
    "ETFFuture",
    "FuturesQuote",
    "IndexFuture",
    "RateCurve",
    "TermStructure",
    "TotalReturnSwap",
    "cost_of_carry_fair_value",
    "discrete_dividend_fair_value",
    "futures_roll_return",
    "implied_repo_rate",
    "sensitivity_grid",
    "trs_breakeven_spread",
]
