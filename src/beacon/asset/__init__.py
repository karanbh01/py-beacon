# src/beacon/asset/__init__.py
"""
Financial assets and the per-asset view.

`Asset` is the immutable base class holding a name, currency, identifier and
asset type; `Equity`, `Bond` and `Commodity` extend it. The index pipeline
accepts only `Equity`. `AssetView` pairs an asset identifier with a
`DataFetcher` so you can ask one asset for its prices, returns, reference data
and corporate actions.
"""
from .base import Asset
from .bond import Bond
from .commodity import Commodity
from .equity import Equity
from .view import AssetView

__all__ = [
    "Asset",
    "AssetView",
    "Bond",
    "Commodity",
    "Equity",
]
