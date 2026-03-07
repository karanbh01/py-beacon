# beacon/asset/commodity.py
"""
Module defining the Commodity asset class.
"""
from dataclasses import dataclass
from .base import Asset


@dataclass(frozen=True)
class Commodity(Asset):
    """
    Represents a commodity asset.
    """
    commodity_type: str = ""
    contract_unit: str = ""

    def __post_init__(self):
        if not self.commodity_type:
            raise ValueError("commodity_type cannot be empty.")
        if not self.contract_unit:
            raise ValueError("contract_unit cannot be empty.")
        if not self.asset_type:
            object.__setattr__(self, 'asset_type', 'COMMODITY')
        super().__post_init__()
