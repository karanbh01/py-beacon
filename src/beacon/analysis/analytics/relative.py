# src/beacon/analysis/analytics/relative.py
"""
A run's relative analytics: the fund against its index or benchmark: tracking,
regression and capture.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class RelativeCategory(Category):
    """A run's relative analytics: the fund against its index or benchmark:
    tracking, regression and capture."""

    name = "relative"
