"""Pricing helpers for Delta-1 derivative instruments."""
from __future__ import annotations

import math


def cost_of_carry_fair_value(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    time_to_expiry_years: float,
    borrow_cost: float = 0.0,
) -> float:
    """Return futures fair value using F = S * exp((r - q + c) * T)."""
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if time_to_expiry_years < 0:
        raise ValueError("time_to_expiry_years cannot be negative.")
    return spot * math.exp((risk_free_rate - dividend_yield + borrow_cost) * time_to_expiry_years)
