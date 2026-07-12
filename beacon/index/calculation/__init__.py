# beacon/index/calculation/__init__.py
"""
The __init__.py for the 'index.calculation' package.

Re-exports IndexCalculator, which composes constituent-selection,
weighting, market-value, and corporate-action logic.
"""
from .calculator import IndexCalculator

__all__ = ["IndexCalculator"]
