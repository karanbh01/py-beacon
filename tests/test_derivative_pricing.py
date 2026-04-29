"""Tests for stateless derivative pricing utilities."""

import math

import pandas as pd
import pytest

from beacon.derivatives.pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    futures_roll_return,
    implied_repo_rate,
    trs_breakeven_spread,
)


def test_cost_of_carry_fair_value_matches_textbook_formula():
    assert cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.5, borrow_cost=0.01) == pytest.approx(
        100.0 * math.exp((0.05 - 0.02 + 0.01) * 1.5)
    )


def test_cost_of_carry_handles_zero_time_and_negative_rates():
    assert cost_of_carry_fair_value(100.0, -0.01, 0.0, 0.0) == pytest.approx(100.0)
    assert cost_of_carry_fair_value(100.0, -0.01, 0.0, 1.0) < 100.0


def test_discrete_dividend_fair_value_matches_present_value_formula():
    dividends = [(0.25, 1.0), (0.75, 1.5)]
    pv_divs = 1.0 * math.exp(-0.04 * 0.25) + 1.5 * math.exp(-0.04 * 0.75)

    assert discrete_dividend_fair_value(100.0, 0.04, 1.0, dividends) == pytest.approx(
        (100.0 - pv_divs) * math.exp(0.04)
    )


def test_discrete_dividend_fair_value_ignores_dividends_outside_tenor():
    assert discrete_dividend_fair_value(100.0, 0.05, 1.0, [(-0.1, 100.0), (0.5, 1.0), (2.0, 100.0)]) == pytest.approx(
        (100.0 - math.exp(-0.05 * 0.5)) * math.exp(0.05)
    )


def test_implied_repo_rate_inverts_cost_of_carry_formula():
    futures = cost_of_carry_fair_value(100.0, 0.045, 0.015, 2.0)

    assert implied_repo_rate(futures, 100.0, 0.015, 2.0) == pytest.approx(0.045)


def test_futures_roll_return_annualizes_price_spread_between_expiries():
    assert futures_roll_return(99.0, 101.0, pd.Timestamp("2025-03-31"), pd.Timestamp("2025-06-30")) == pytest.approx(
        (99.0 / 101.0 - 1.0) * (365.0 / 91.0)
    )


def test_trs_breakeven_spread_matches_implied_repo_minus_risk_free_rate():
    futures = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0, borrow_cost=0.0075)

    assert trs_breakeven_spread(futures, 100.0, 0.05, 1.0, 0.02) == pytest.approx(0.0075)


@pytest.mark.parametrize(
    "func,args",
    [
        (cost_of_carry_fair_value, (0.0, 0.05, 0.02, 1.0)),
        (discrete_dividend_fair_value, (0.0, 0.05, 1.0, [])),
        (implied_repo_rate, (100.0, 0.0, 0.02, 1.0)),
        (trs_breakeven_spread, (100.0, 0.0, 0.05, 1.0, 0.02)),
    ],
)
def test_zero_spot_rejected(func, args):
    with pytest.raises(ValueError):
        func(*args)


def test_time_and_expiry_edge_cases_rejected():
    with pytest.raises(ValueError):
        cost_of_carry_fair_value(100.0, 0.05, 0.02, -1.0)
    with pytest.raises(ValueError):
        implied_repo_rate(100.0, 100.0, 0.02, 0.0)
    with pytest.raises(ValueError):
        futures_roll_return(100.0, 101.0, pd.Timestamp("2025-06-30"), pd.Timestamp("2025-03-31"))
