# src/beacon/analysis/analytics/__init__.py
"""
A run's analytics, the one way in to every calculation on it, as
`result.analytics`:

    result.analytics.portfolio.performance   # a whole portfolio's figures
    result.analytics.request                 # the analysis the run was asked for
    result.analytics.to_dict()               # its analytics block

A portfolio's analytics come in nine categories: `performance`, `risk`,
`relative`, `holdings`, `trading`, `flows`, `income`, `vehicle` and
`attribution`.
"""
# BN-433.
from .category import Category
from .entry import Analytics
from .portfolio import PortfolioLevel

__all__ = ["Analytics", "Category", "PortfolioLevel"]
