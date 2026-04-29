"""Pricing helpers for derivative instruments."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def continuous_carry_forward_price(
    spot_price: float,
    risk_free_rate: float,
    time_to_maturity: float,
    dividend_yield: float = 0.0,
) -> float:
    """Price a forward/future using continuous carry.

    F = S * exp((r - q) * T)
    """
    return spot_price * math.exp((risk_free_rate - dividend_yield) * time_to_maturity)


def present_value_discrete_dividends(
    dividends: Iterable[Tuple[float, float]],
    risk_free_rate: float,
    time_to_maturity: float,
) -> float:
    """Return the present value of known dividends within the contract tenor.

    Each dividend is represented as ``(time_to_payment, amount)`` where time is
    in years from valuation date. Dividends outside ``time_to_maturity`` are
    ignored because they do not affect the current futures tenor.
    """
    pv = 0.0
    for payment_time, amount in dividends:
        if payment_time < 0 or payment_time > time_to_maturity:
            continue
        pv += amount * math.exp(-risk_free_rate * payment_time)
    return pv


def discrete_dividend_forward_price(
    spot_price: float,
    risk_free_rate: float,
    time_to_maturity: float,
    dividends: Iterable[Tuple[float, float]],
) -> float:
    """Price a forward/future by subtracting PV(discrete dividends).

    F = (S - PV(divs)) * exp(r * T)
    """
    dividend_pv = present_value_discrete_dividends(
        dividends,
        risk_free_rate,
        time_to_maturity,
    )
    return (spot_price - dividend_pv) * math.exp(risk_free_rate * time_to_maturity)
