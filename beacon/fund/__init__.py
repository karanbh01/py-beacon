# beacon/fund/__init__.py
"""
The __init__.py for the 'fund' module.

This module models financial funds, particularly ETFs and index funds.
"""
from .base import IndexFund
from .etf import ETF

__all__ = [
    "IndexFund",
    "ETF",
]