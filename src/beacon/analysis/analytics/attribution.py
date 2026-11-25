# src/beacon/analysis/analytics/attribution.py
"""
A run's attribution analytics: what the return and the active return came from.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class AttributionCategory(Category):
    """A run's attribution analytics: what the return and the active return came from."""

    name = "attribution"
