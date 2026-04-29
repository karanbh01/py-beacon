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


def implied_repo_rate(
    futures_price: float,
    spot: float,
    dividend_yield: float,
    time_to_expiry_years: float,
) -> float:
    """Return implied repo/risk-free rate from a futures price."""
    if futures_price <= 0:
        raise ValueError("futures_price must be positive.")
    if spot <= 0:
        raise ValueError("spot must be positive.")
    if time_to_expiry_years <= 0:
        raise ValueError("time_to_expiry_years must be positive.")
    return (math.log(futures_price / spot) + dividend_yield * time_to_expiry_years) / time_to_expiry_years
