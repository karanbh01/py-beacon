# src/beacon/analysis/analytics/trading.py
"""
A run's trading analytics: turnover, trades and their costs.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class TradingCategory(Category):
    """A run's trading analytics: turnover, trades and their costs."""

    name = "trading"
