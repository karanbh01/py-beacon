# beacon/asset/equity.py
"""
Module defining the Equity asset class.
"""
from dataclasses import dataclass
from typing import Optional
from .base import Asset


@dataclass(frozen=True)
class Equity(Asset):
    """
    Represents an equity security.
    """
    ticker: str = ""
    exchange: str = ""
    isin: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None

    def __post_init__(self):
        if not self.ticker:
            raise ValueError("ticker cannot be empty.")
        if not self.exchange:
            raise ValueError("exchange cannot be empty.")
        if not self.asset_id:
            object.__setattr__(self, 'asset_id', self.ticker)
        if not self.asset_type:
            object.__setattr__(self, 'asset_type', 'EQUITY')
        super().__post_init__()
