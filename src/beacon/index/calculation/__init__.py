# src/beacon/index/calculation/__init__.py
"""
The index calculator.

Re-exports `IndexCalculator`, which composes constituent selection, weighting,
market values, corporate actions, deletions and total-return reinvestment, and
the selection result objects that record how a universe narrowed.
"""
from .calculator import IndexCalculator
from .selection import (
    UNIVERSE_POSITION,
    SelectionResult,
    SelectionStep,
    select_with_provenance,
)

__all__ = [
    "UNIVERSE_POSITION",
    "IndexCalculator",
    "SelectionResult",
    "SelectionStep",
    "select_with_provenance",
]
