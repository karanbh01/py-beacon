"""Futures contract models."""
from __future__ import annotations

import math
from typing import Dict

import pandas as pd

from .pricing import cost_of_carry_fair_value


class IndexFuture:
    """Index futures contract with cost-of-carry pricing and basis analytics."""

    def __init__(
        self,
        derivative_id: str,
        underlying_id: str,
        currency: str,
        expiry_date: str,
        contract_multiplier: float,
        tick_size: float,
        tick_value: float,
    ):
        if not derivative_id:
            raise ValueError("derivative_id cannot be empty.")
        if not underlying_id:
            raise ValueError("underlying_id cannot be empty.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be positive.")
        if tick_size <= 0:
            raise ValueError("tick_size must be positive.")
        if tick_value <= 0:
            raise ValueError("tick_value must be positive.")

        self.derivative_id = derivative_id
        self.underlying_id = underlying_id
        self.currency = currency
        self.expiry_date = pd.Timestamp(expiry_date)
        self.contract_multiplier = float(contract_multiplier)
        self.tick_size = float(tick_size)
        self.tick_value = float(tick_value)

    def time_to_expiry(self, valuation_date: pd.Timestamp) -> float:
        """Return ACT/365 years to expiry, floored at zero after expiry."""
        return max((self.expiry_date - pd.Timestamp(valuation_date)).days, 0) / 365.0

    def fair_value(
        self,
        spot_price: float,
        valuation_date: pd.Timestamp,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        borrow_cost: float = 0.0,
    ) -> float:
        """Return cost-of-carry fair value for the contract."""
        return cost_of_carry_fair_value(
            spot=spot_price,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            borrow_cost=borrow_cost,
            time_to_expiry_years=self.time_to_expiry(valuation_date),
        )

    def basis(self, futures_price: float, spot_price: float) -> float:
        """Return futures basis: futures price minus spot price."""
        return futures_price - spot_price

    def annualised_basis(self, futures_price: float, spot_price: float, valuation_date: pd.Timestamp) -> float:
        """Return implied financing rate ln(F/S) / T."""
        if futures_price <= 0:
            raise ValueError("futures_price must be positive.")
        if spot_price <= 0:
            raise ValueError("spot_price must be positive.")
        time_to_expiry = self.time_to_expiry(valuation_date)
        if time_to_expiry == 0:
            return 0.0
        return math.log(futures_price / spot_price) / time_to_expiry

    def mark_to_market(
        self,
        futures_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        risk_free_rate: float,
        dividend_yield: float = 0.0,
        borrow_cost: float = 0.0,
    ) -> Dict[str, float]:
        """Return fair value, basis, theoretical edge, and time to expiry."""
        fair_value = self.fair_value(spot_price, valuation_date, risk_free_rate, dividend_yield, borrow_cost)
        return {
            "fair_value": fair_value,
            "basis": self.basis(futures_price, spot_price),
            "theoretical_edge": futures_price - fair_value,
            "time_to_expiry": self.time_to_expiry(valuation_date),
        }
