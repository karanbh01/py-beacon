# src/beacon/index/__init__.py
"""
Index methodologies and calculation.

Defining an index (`IndexDefinition`, eligibility rules and weighting
schemes), selecting its constituents, weighting them, and computing its levels
(`IndexCalculator`, which returns an `IndexResult`).
"""
from .asset_view import IndexAssetView
from .calculation import IndexCalculator
from .constructor import IndexDefinition
from .derived import OptimisedIndexDefinition, calculate_derived_index
from .expression_rules import ExpressionRule
from .feature_rules import FeatureRule
from .methodology import (
    EligibilityRuleBase,
    EqualWeighted,
    LiquidityRule,
    MarketCapRule,
    MarketCapWeighted,
    WeightingSchemeBase,
)
from .result import IndexResult

__all__ = [
    "EligibilityRuleBase",
    "EqualWeighted",
    "ExpressionRule",
    "FeatureRule",
    "IndexAssetView",
    "IndexCalculator",
    "IndexDefinition",
    "IndexResult",
    "LiquidityRule",
    "MarketCapRule",
    "MarketCapWeighted",
    "OptimisedIndexDefinition",
    "WeightingSchemeBase",
    "calculate_derived_index",
]
