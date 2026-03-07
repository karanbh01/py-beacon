# beacon/asset/bond.py
"""
Module defining the Bond asset class.
"""
from dataclasses import dataclass
from typing import Optional
from .base import Asset


@dataclass(frozen=True)
class Bond(Asset):
    """
    Represents a bond security.
    """
    coupon: float = 0.0
    maturity_date: str = ""
    issuer: str = ""
    credit_rating: Optional[str] = None
    face_value: float = 1000.0

    def __post_init__(self):
        if not self.maturity_date:
            raise ValueError("maturity_date cannot be empty.")
        if not self.issuer:
            raise ValueError("issuer cannot be empty.")
        if not self.asset_type:
            object.__setattr__(self, 'asset_type', 'BOND')
        super().__post_init__()
