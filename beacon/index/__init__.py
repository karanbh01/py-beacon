# beacon/index/__init__.py
"""
The __init__.py for the 'index' module.

This module is core for defining index methodologies, selecting constituents,
calculating weights, and computing index levels.
"""
from .methodology import (
    EligibilityRuleBase,
    MarketCapRule, # Example implementation
    LiquidityRule, # Example implementation
    WeightingSchemeBase,
    MarketCapWeighted, # Example implementation
    EqualWeighted # Example implementation
)
from .constructor import IndexDefinition # If IndexDefinition is in constructor.py
from .calculation import IndexCalculationAgent
from .result import IndexResult

__all__ = [
    "EligibilityRuleBase",
    "MarketCapRule",
    "LiquidityRule",
    "WeightingSchemeBase",
    "MarketCapWeighted",
    "EqualWeighted",
    "IndexDefinition",
    "IndexCalculationAgent",
    "IndexResult",
]