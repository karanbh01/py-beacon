"""Base abstractions for derivative instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class DerivativeBase(ABC):
    """Abstract base for Delta-1 derivative instruments."""

    def __init__(
        self,
        derivative_id: str,
        underlying_id: str,
        underlying_type: str,
        currency: str,
        expiry_date: str,
        notional: float,
    ):
        if not derivative_id:
            raise ValueError("derivative_id cannot be empty.")
        if not underlying_id:
            raise ValueError("underlying_id cannot be empty.")
        if not underlying_type:
            raise ValueError("underlying_type cannot be empty.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if notional <= 0:
            raise ValueError("notional must be positive.")

        self.derivative_id = derivative_id
        self.underlying_id = underlying_id
        self.underlying_type = underlying_type
        self.currency = currency
        self.expiry_date = pd.Timestamp(expiry_date)
        self.notional = float(notional)

    @abstractmethod
    def fair_value(
        self,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> float:
        """Return the instrument fair value at the valuation date."""

    @abstractmethod
    def mark_to_market(
        self,
        market_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> Dict[str, float]:
        """Return mark-to-market valuation details."""

    def time_to_expiry(self, valuation_date: pd.Timestamp) -> float:
        """Time to expiry in years using ACT/365."""
        valuation_ts = pd.Timestamp(valuation_date)
        days = (self.expiry_date - valuation_ts).days
        return max(days, 0) / 365.0
