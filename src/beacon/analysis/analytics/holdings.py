# src/beacon/analysis/analytics/holdings.py
"""
A run's holdings analytics: the names held, their groups and their style.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class HoldingsCategory(Category):
    """A run's holdings analytics: the names held, their groups and their style."""

    name = "holdings"
