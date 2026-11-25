# src/beacon/analysis/analytics/performance.py
"""
A run's performance analytics: returns, risk-adjusted ratios, rolling
outcomes, and crises and regimes.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class PerformanceCategory(Category):
    """A run's performance analytics: returns, risk-adjusted ratios, rolling
    outcomes, and crises and regimes."""

    name = "performance"
