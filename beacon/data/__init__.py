# beacon/data/__init__.py
"""
The __init__.py for the 'data' module.

This module handles fetching, parsing, and providing financial data.
"""
from .fetcher import DataFetcher, CSVSecurityDataProvider # , APIClientDataFetcher (example for future)
# from .parsers import DataParser # If implemented

__all__ = [
    "DataFetcher",
    "CSVSecurityDataProvider",
    # "DataParser",
]