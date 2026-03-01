# beacon/data/__init__.py
"""
The __init__.py for the 'data' module.

This module handles fetching, parsing, and providing financial data.
"""
from .base import MarketData, ReferenceData
from .fetcher import DataFetcher
from .loader import load_data

__all__ = [
    "MarketData",
    "ReferenceData",
    "DataFetcher",
    "load_data",
]
