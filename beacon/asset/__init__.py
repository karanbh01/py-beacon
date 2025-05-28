# beacon/asset/__init__.py
"""
The __init__.py for the 'asset' module.

This module defines and manages financial assets.
"""
from .base import Asset
from .equity import Equity

__all__ = [
    "Asset",
    "Equity",
]