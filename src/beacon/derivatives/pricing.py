# src/beacon/derivatives/pricing.py
"""
Pure, stateless pricing functions for Delta-1 derivatives.

These functions implement standard textbook cost-of-carry relationships and
have **zero dependencies on the rest of Beacon** (only the standard library and
pandas Timestamps for date arithmetic). All rates are continuously compounded
and expressed as annual decimals (e.g. ``0.05`` for 5%); times are in years.
"""
import math

import pandas as pd

__all__ = [
    "cost_of_carry_fair_value",
    "discrete_dividend_fair_value",
    "futures_roll_return",
    "implied_repo_rate",
    "trs_breakeven_spread",
]

# Seconds in an average year (365.25 days), used for date-difference year fractions.
_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def cost_of_carry_fair_value(spot: float,
                             risk_free_rate: float,
                             dividend_yield: float,
                             time_to_expiry_years: float,
                             borrow_cost: float = 0.0) -> float:
    """Fair forward/futures value under continuous cost of carry.

    ``F = S * exp((r - q + c) * T)``

    Args:
        spot: Current spot price ``S`` (must be non-negative).
        risk_free_rate: Continuously compounded risk-free rate ``r``.
        dividend_yield: Continuous dividend yield ``q``.
        time_to_expiry_years: Time to expiry ``T`` in years (must be non-negative).
        borrow_cost: Continuous borrow/financing spread ``c`` (default 0).

    Returns:
        The fair value ``F``. Equals *spot* when ``T == 0``.

    Raises:
        ValueError: If *spot* or *time_to_expiry_years* is negative.
    """
    if spot < 0:
        raise ValueError(f"spot must be non-negative, got {spot}")
    if time_to_expiry_years < 0:
        raise ValueError(
            f"time_to_expiry_years must be non-negative, got {time_to_expiry_years}"
        )

    carry = risk_free_rate - dividend_yield + borrow_cost
    return spot * math.exp(carry * time_to_expiry_years)


def discrete_dividend_fair_value(spot: float,
                                 risk_free_rate: float,
                                 time_to_expiry_years: float,
                                 dividends: list[tuple[float, float]]) -> float:
    """Fair forward/futures value with discrete cash dividends.

    ``F = (S - PV(divs)) * exp(r * T)`` where each dividend is discounted at the
    risk-free rate to today: ``PV = amount * exp(-r * t_ex)``.

    Args:
        spot: Current spot price ``S`` (must be non-negative).
        risk_free_rate: Continuously compounded risk-free rate ``r``.
        time_to_expiry_years: Time to expiry ``T`` in years (must be non-negative).
        dividends: List of ``(time_to_ex_years, amount)`` tuples. Only dividends
            with ex-dates on or before expiry (``0 <= t_ex <= T``) are included.

    Returns:
        The fair value ``F``.

    Raises:
        ValueError: If *spot* or *time_to_expiry_years* is negative.
    """
    if spot < 0:
        raise ValueError(f"spot must be non-negative, got {spot}")
    if time_to_expiry_years < 0:
        raise ValueError(
            f"time_to_expiry_years must be non-negative, got {time_to_expiry_years}"
        )

    pv_dividends = 0.0
    for t_ex, amount in dividends:
        if 0.0 <= t_ex <= time_to_expiry_years:
            pv_dividends += amount * math.exp(-risk_free_rate * t_ex)

    return (spot - pv_dividends) * math.exp(risk_free_rate * time_to_expiry_years)


def implied_repo_rate(futures_price: float,
                      spot: float,
                      dividend_yield: float,
                      time_to_expiry_years: float) -> float:
    """Continuously compounded financing rate implied by a futures price.

    Inverts the cost-of-carry relationship:
    ``r_implied = (ln(F / S) + q * T) / T``

    Args:
        futures_price: Observed futures price ``F`` (must be positive).
        spot: Current spot price ``S`` (must be positive).
        dividend_yield: Continuous dividend yield ``q``.
        time_to_expiry_years: Time to expiry ``T`` in years (must be positive).

    Returns:
        The implied repo (financing) rate.

    Raises:
        ValueError: If *time_to_expiry_years*, *spot* or *futures_price* is
            non-positive.
    """
    if time_to_expiry_years <= 0:
        raise ValueError(
            f"time_to_expiry_years must be positive, got {time_to_expiry_years}"
        )
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if futures_price <= 0:
        raise ValueError(f"futures_price must be positive, got {futures_price}")

    return (math.log(futures_price / spot) + dividend_yield * time_to_expiry_years) \
        / time_to_expiry_years


def futures_roll_return(front_price: float,
                        back_price: float,
                        front_expiry: pd.Timestamp,
                        back_expiry: pd.Timestamp) -> float:
    """Annualised simple roll return from rolling a front contract to a back one.

    ``roll = (front / back - 1) / dt`` where ``dt`` is the year fraction between
    the two expiries. Positive in backwardation (front above back), negative in
    contango.

    Args:
        front_price: Price of the near (front) contract (must be positive).
        back_price: Price of the far (back) contract (must be positive).
        front_expiry: Expiry of the front contract.
        back_expiry: Expiry of the back contract (must be after *front_expiry*).

    Returns:
        The annualised roll return as a decimal.

    Raises:
        ValueError: If either price is non-positive, or *back_expiry* is not
            strictly after *front_expiry*.
    """
    if front_price <= 0:
        raise ValueError(f"front_price must be positive, got {front_price}")
    if back_price <= 0:
        raise ValueError(f"back_price must be positive, got {back_price}")

    dt_years = float((back_expiry - front_expiry).total_seconds()) / _SECONDS_PER_YEAR
    if dt_years <= 0:
        raise ValueError("back_expiry must be strictly after front_expiry.")

    return (front_price / back_price - 1.0) / dt_years


def trs_breakeven_spread(futures_price: float,
                         spot: float,
                         risk_free_rate: float,
                         time_to_expiry_years: float,
                         dividend_yield: float) -> float:
    """Financing spread at which a total return swap matches futures economics.

    The futures price embeds an implied financing rate (:func:`implied_repo_rate`).
    A TRS financed at ``r + spread`` reproduces those economics when the spread
    equals the gap between the implied financing rate and the risk-free rate:

    ``spread = implied_repo_rate(F, S, q, T) - r``

    A fairly priced future (financed exactly at ``r``) gives a breakeven spread
    of zero.

    Args:
        futures_price: Observed futures price ``F`` (must be positive).
        spot: Current spot price ``S`` (must be positive).
        risk_free_rate: Continuously compounded risk-free rate ``r``.
        time_to_expiry_years: Time to expiry ``T`` in years (must be positive).
        dividend_yield: Continuous dividend yield ``q``.

    Returns:
        The breakeven financing spread as a decimal.

    Raises:
        ValueError: If *time_to_expiry_years*, *spot* or *futures_price* is
            non-positive.
    """
    implied = implied_repo_rate(
        futures_price, spot, dividend_yield, time_to_expiry_years
    )
    return implied - risk_free_rate
