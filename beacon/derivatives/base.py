"""Base abstractions for derivative instruments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class DerivativeBase(ABC):
    """Abstract base for derivative instruments."""

    def __init__(self, derivative_id: str, underlying_id: str, currency: str, expiry_date: str, notional: float):
        if not derivative_id:
            raise ValueError("derivative_id cannot be empty.")
        if not underlying_id:
            raise ValueError("underlying_id cannot be empty.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if notional <= 0:
            raise ValueError("notional must be positive.")
        self.derivative_id = derivative_id
        self.underlying_id = underlying_id
        self.currency = currency
        self.expiry_date = pd.Timestamp(expiry_date)
        self.notional = float(notional)

    @abstractmethod
    def fair_value(self, spot_price: float, valuation_date: pd.Timestamp, market_data: Dict[str, float]) -> float:
        """Return fair value at the valuation date."""

    @abstractmethod
    def mark_to_market(
        self,
        market_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> Dict[str, float]:
        """Return mark-to-market details."""

    def accrued_days(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
        """Return non-negative ACT day count between two dates."""
        return max((pd.Timestamp(end_date) - pd.Timestamp(start_date)).days, 0)
