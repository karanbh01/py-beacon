# beacon/asset/__init__.py
"""
The __init__.py for the 'asset' module.

This module defines and manages financial assets.
"""
from .base import Asset
from .equity import Equity
from .bond import Bond
from .commodity import Commodity
from .view import AssetView

__all__ = [
    "Asset",
    "Equity",
    "Bond",
    "Commodity",
    "AssetView",
]