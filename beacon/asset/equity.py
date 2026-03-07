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

    def __post_init__(self):
        super().__post_init__()
        if not self.ticker:
            raise ValueError("ticker cannot be empty.")
        if not self.exchange:
            raise ValueError("exchange cannot be empty.")
