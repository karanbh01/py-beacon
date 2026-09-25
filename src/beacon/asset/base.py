# src/beacon/asset/base.py
"""
Module defining the base class for financial assets.
"""
from dataclasses import dataclass


# BN-185 made the index pipeline equity-only.
@dataclass(frozen=True)
class Asset:
    """Base class for a financial asset. An immutable metadata container.

    Every field must be non-empty, or construction raises `ValueError`.

    **The index pipeline accepts only :class:`~beacon.asset.equity.Equity`.**
    Subclasses such as :class:`~beacon.asset.bond.Bond` and
    :class:`~beacon.asset.commodity.Commodity` are usable as metadata, but a
    universe containing one is refused by selection, weighting, market values
    and corporate-action handling alike; see
    :func:`~beacon.asset.equity.require_equity`.
    """
    name: str
    currency: str
    asset_id: str = ""
    asset_type: str = ""

    def __post_init__(self) -> None:
        for field_name in ('asset_id', 'asset_type', 'name', 'currency'):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")
