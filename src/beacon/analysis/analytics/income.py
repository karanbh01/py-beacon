# src/beacon/analysis/analytics/income.py
"""
A run's income analytics: the dividends the portfolio earned.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class IncomeCategory(Category):
    """A run's income analytics: the dividends the portfolio earned."""

    name = "income"
