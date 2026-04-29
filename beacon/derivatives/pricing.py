"""Stateless pricing utilities for Delta-1 derivatives."""
from __future__ import annotations

import math
from typing import List, Tuple

import pandas as pd


def _validate_spot(spot: float) -> None:
    if spot <= 0:
        raise ValueError("spot must be positive.")


def _validate_time(time_to_expiry_years: float) -> None:
    if time_to_expiry_years < 0:
        raise ValueError("time_to_expiry_years cannot be negative.")


def cost_of_carry_fair_value(
    spot: float,
    risk_free_rate: float,
    dividend_yield: float,
    time_to_expiry_years: float,
    borrow_cost: float = 0.0,
) -> float:
    """Return fair value from the cost-of-carry model.

    F = S * exp((r - q + c) * T)
    """
    _validate_spot(spot)
    _validate_time(time_to_expiry_years)
    return spot * math.exp((risk_free_rate - dividend_yield + borrow_cost) * time_to_expiry_years)


def discrete_dividend_fair_value(
    spot: float,
    risk_free_rate: float,
    time_to_expiry_years: float,
    dividends: List[Tuple[float, float]],
) -> float:
    """Return fair value with known discrete dividends.

    Dividends are ``(time_to_ex_years, amount)`` pairs. Dividends outside the
    futures tenor are ignored.

    F = (S - PV(divs)) * exp(r * T)
    """
    _validate_spot(spot)
    _validate_time(time_to_expiry_years)
    pv_dividends = 0.0
    for time_to_ex_years, amount in dividends:
        if time_to_ex_years < 0 or time_to_ex_years > time_to_expiry_years:
            continue
        pv_dividends += amount * math.exp(-risk_free_rate * time_to_ex_years)
    return (spot - pv_dividends) * math.exp(risk_free_rate * time_to_expiry_years)


def implied_repo_rate(
    futures_price: float,
    spot: float,
    dividend_yield: float,
    time_to_expiry_years: float,
) -> float:
    """Return implied repo/risk-free rate from a futures price.

    r_implied = (ln(F/S) + q*T) / T
    """
    if futures_price <= 0:
        raise ValueError("futures_price must be positive.")
    _validate_spot(spot)
    if time_to_expiry_years <= 0:
        raise ValueError("time_to_expiry_years must be positive.")
    return (math.log(futures_price / spot) + dividend_yield * time_to_expiry_years) / time_to_expiry_years


def futures_roll_return(
    front_price: float,
    back_price: float,
    front_expiry: pd.Timestamp,
    back_expiry: pd.Timestamp,
) -> float:
    """Return annualised roll return from front to back futures contracts."""
    if front_price <= 0:
        raise ValueError("front_price must be positive.")
    if back_price <= 0:
        raise ValueError("back_price must be positive.")
    front_ts = pd.Timestamp(front_expiry)
    back_ts = pd.Timestamp(back_expiry)
    days_between = (back_ts - front_ts).days
    if days_between <= 0:
        raise ValueError("back_expiry must be after front_expiry.")
    return (front_price / back_price - 1.0) * (365.0 / days_between)


def trs_breakeven_spread(
    futures_price: float,
    spot: float,
    risk_free_rate: float,
    time_to_expiry_years: float,
    dividend_yield: float,
) -> float:
    """Return spread at which TRS economics equal futures economics.

    Solves for ``s`` in ``F = S * exp((r + s - q) * T)``.
    """
    if futures_price <= 0:
        raise ValueError("futures_price must be positive.")
    _validate_spot(spot)
    if time_to_expiry_years <= 0:
        raise ValueError("time_to_expiry_years must be positive.")
    implied_rate = implied_repo_rate(futures_price, spot, dividend_yield, time_to_expiry_years)
    return implied_rate - risk_free_rate
