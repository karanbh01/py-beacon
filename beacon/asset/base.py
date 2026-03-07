# beacon/asset/base.py
"""
Module defining the base class for financial assets.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Asset:
    """
    Base class for a financial asset. Serves as an immutable metadata container.
    """
    asset_id: str
    asset_type: str
    name: str
    currency: str

    def __post_init__(self):
        for field_name in ('asset_id', 'asset_type', 'name', 'currency'):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty.")
