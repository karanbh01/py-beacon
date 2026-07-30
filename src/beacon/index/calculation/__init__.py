# src/beacon/index/calculation/__init__.py
"""
The __init__.py for the 'index.calculation' package.

Re-exports IndexCalculator, which composes constituent-selection,
weighting, market-value, and corporate-action logic, and the selection result
objects that carry the record of how a universe narrowed.
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
