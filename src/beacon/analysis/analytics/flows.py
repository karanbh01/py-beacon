# src/beacon/analysis/analytics/flows.py
"""
A run's flows analytics: money arriving and leaving, and what dealing in it cost.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class FlowsCategory(Category):
    """A run's flows analytics: money arriving and leaving, and what dealing in it cost."""

    name = "flows"
