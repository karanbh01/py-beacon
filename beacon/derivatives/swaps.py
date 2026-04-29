"""Swap derivative models."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .base import DerivativeBase


class TotalReturnSwap(DerivativeBase):
    """Total return swap on an index or equity basket."""

    def __init__(
        self,
        derivative_id: str,
        underlying_id: str,
        currency: str,
        start_date: str,
        end_date: str,
        notional: float,
        spread_bps: float,
        reference_rate: float,
        payment_frequency: str,
        reset_type: str = "UNFUNDED",
    ):
        super().__init__(derivative_id, underlying_id, currency, end_date, notional)
        if not payment_frequency:
            raise ValueError("payment_frequency cannot be empty.")
        if reset_type not in {"UNFUNDED", "FUNDED"}:
            raise ValueError("reset_type must be 'UNFUNDED' or 'FUNDED'.")
        self.start_date = pd.Timestamp(start_date)
        if self.expiry_date <= self.start_date:
            raise ValueError("end_date must be after start_date.")
        self.spread_bps = float(spread_bps)
        self.reference_rate = float(reference_rate)
        self.payment_frequency = payment_frequency
        self.reset_type = reset_type

    @property
    def spread_rate(self) -> float:
        """Return contractual spread as a decimal rate."""
        return self.spread_bps / 10000.0

    def financing_cost(self, valuation_date: pd.Timestamp, last_reset_date: pd.Timestamp, reference_rate: float) -> float:
        """Return accrued financing cost since last reset using ACT/365."""
        accrued_days = self.accrued_days(last_reset_date, valuation_date)
        return self.notional * (reference_rate + self.spread_rate) * accrued_days / 365.0

    def total_return_leg(self, spot_price: float, initial_spot_price: float) -> float:
        """Return receiver total-return leg P&L."""
        if initial_spot_price <= 0:
            raise ValueError("initial_spot_price must be positive.")
        return self.notional * (spot_price / initial_spot_price - 1.0)

    def fair_value(self, spot_price: float, valuation_date: pd.Timestamp, market_data: Dict[str, float]) -> float:
        """Return receiver P&L net of accrued financing."""
        initial_spot_price = market_data["initial_spot_price"]
        last_reset_date = market_data.get("last_reset_date", self.start_date)
        reference_rate = market_data.get("reference_rate", self.reference_rate)
        total_return_leg = self.total_return_leg(spot_price, initial_spot_price)
        accrued_financing = self.financing_cost(valuation_date, last_reset_date, reference_rate)
        return total_return_leg - accrued_financing

    def mark_to_market(
        self,
        market_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> Dict[str, float]:
        """Return total return leg, financing leg, net MTM, and accrued days."""
        initial_spot_price = market_data["initial_spot_price"]
        last_reset_date = market_data.get("last_reset_date", self.start_date)
        reference_rate = market_data.get("reference_rate", self.reference_rate)
        total_return_leg = self.total_return_leg(spot_price, initial_spot_price)
        financing_leg = self.financing_cost(valuation_date, last_reset_date, reference_rate)
        return {
            "total_return_leg": total_return_leg,
            "financing_leg": financing_leg,
            "net_mtm": total_return_leg - financing_leg,
            "accrued_days": self.accrued_days(last_reset_date, valuation_date),
        }
