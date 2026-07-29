# tests/test_derivatives_pricing.py
"""Unit tests for beacon.derivatives.pricing — pure Delta-1 pricing functions.

Uses textbook / hand-calculated examples and covers the required edge cases:
zero time to expiry, zero spot, and negative rates.
"""
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

# ---------------------------------------------------------------------------
# cost_of_carry_fair_value
# ---------------------------------------------------------------------------

class TestCostOfCarry:

    def test_textbook_value(self):
        # S=100, r=5%, q=2%, T=1 -> F = 100 * exp(0.03) = 103.045453...
        f = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0)
        assert f == pytest.approx(100.0 * math.exp(0.03))
        assert f == pytest.approx(103.045453, abs=1e-5)

    def test_borrow_cost_added_to_carry(self):
        # c=1% -> F = 100 * exp(0.04)
        f = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0, borrow_cost=0.01)
        assert f == pytest.approx(100.0 * math.exp(0.04))

    def test_zero_time_returns_spot(self):
        assert cost_of_carry_fair_value(100.0, 0.05, 0.02, 0.0) == pytest.approx(100.0)

    def test_zero_spot_returns_zero(self):
        assert cost_of_carry_fair_value(0.0, 0.05, 0.02, 1.0) == 0.0

    def test_negative_rate(self):
        # r=-1%, q=0 -> F = 100 * exp(-0.01) < 100
        f = cost_of_carry_fair_value(100.0, -0.01, 0.0, 1.0)
        assert f == pytest.approx(100.0 * math.exp(-0.01))
        assert f < 100.0

    def test_dividend_yield_above_rate_gives_discount(self):
        # q > r -> future below spot
        assert cost_of_carry_fair_value(100.0, 0.02, 0.05, 1.0) < 100.0

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            cost_of_carry_fair_value(-1.0, 0.05, 0.0, 1.0)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time_to_expiry"):
            cost_of_carry_fair_value(100.0, 0.05, 0.0, -1.0)


# ---------------------------------------------------------------------------
# discrete_dividend_fair_value
# ---------------------------------------------------------------------------

class TestDiscreteDividend:

    def test_no_dividends_matches_cost_of_carry_zero_yield(self):
        f = discrete_dividend_fair_value(100.0, 0.05, 1.0, [])
        assert f == pytest.approx(cost_of_carry_fair_value(100.0, 0.05, 0.0, 1.0))
        assert f == pytest.approx(100.0 * math.exp(0.05))

    def test_single_dividend_hand_calc(self):
        # PV = 2 * exp(-0.05*0.5); F = (100 - PV) * exp(0.05)
        divs = [(0.5, 2.0)]
        pv = 2.0 * math.exp(-0.05 * 0.5)
        expected = (100.0 - pv) * math.exp(0.05)
        assert discrete_dividend_fair_value(100.0, 0.05, 1.0, divs) == pytest.approx(expected)

    def test_dividend_lowers_forward(self):
        base = discrete_dividend_fair_value(100.0, 0.05, 1.0, [])
        with_div = discrete_dividend_fair_value(100.0, 0.05, 1.0, [(0.5, 3.0)])
        assert with_div < base

    def test_dividends_after_expiry_ignored(self):
        # A dividend with t_ex > T must not affect the price.
        no_late = discrete_dividend_fair_value(100.0, 0.05, 1.0, [])
        with_late = discrete_dividend_fair_value(100.0, 0.05, 1.0, [(2.0, 5.0)])
        assert with_late == pytest.approx(no_late)

    def test_multiple_dividends(self):
        divs = [(0.25, 1.0), (0.75, 1.0)]
        pv = math.exp(-0.05 * 0.25) + math.exp(-0.05 * 0.75)
        expected = (100.0 - pv) * math.exp(0.05)
        assert discrete_dividend_fair_value(100.0, 0.05, 1.0, divs) == pytest.approx(expected)

    def test_zero_time_returns_spot(self):
        assert discrete_dividend_fair_value(100.0, 0.05, 0.0, []) == pytest.approx(100.0)

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            discrete_dividend_fair_value(-1.0, 0.05, 1.0, [])


# ---------------------------------------------------------------------------
# implied_repo_rate
# ---------------------------------------------------------------------------

class TestImpliedRepoRate:

    def test_inverts_cost_of_carry(self):
        # A fairly priced future implies a financing rate equal to r.
        f = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0)
        assert implied_repo_rate(f, 100.0, 0.02, 1.0) == pytest.approx(0.05)

    def test_futures_at_spot_zero_yield_gives_zero(self):
        # F == S, q=0 -> ln(1)/T = 0
        assert implied_repo_rate(100.0, 100.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_rich_future_raises_implied_rate(self):
        cheap = implied_repo_rate(102.0, 100.0, 0.0, 1.0)
        rich = implied_repo_rate(105.0, 100.0, 0.0, 1.0)
        assert rich > cheap

    def test_zero_time_raises(self):
        with pytest.raises(ValueError, match="time_to_expiry"):
            implied_repo_rate(103.0, 100.0, 0.02, 0.0)

    def test_zero_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            implied_repo_rate(103.0, 0.0, 0.02, 1.0)

    def test_non_positive_futures_raises(self):
        with pytest.raises(ValueError, match="futures_price"):
            implied_repo_rate(0.0, 100.0, 0.02, 1.0)


# ---------------------------------------------------------------------------
# futures_roll_return
# ---------------------------------------------------------------------------

class TestFuturesRollReturn:

    def test_backwardation_positive_hand_calc(self):
        front_exp = pd.Timestamp("2024-03-15")
        back_exp = pd.Timestamp("2024-06-15")  # 92 days
        dt = (back_exp - front_exp).total_seconds() / (365.25 * 24 * 3600)
        expected = (105.0 / 100.0 - 1.0) / dt
        roll = futures_roll_return(105.0, 100.0, front_exp, back_exp)
        assert roll == pytest.approx(expected)
        assert roll > 0  # backwardation

    def test_contango_negative(self):
        front_exp = pd.Timestamp("2024-03-15")
        back_exp = pd.Timestamp("2024-06-15")
        roll = futures_roll_return(98.0, 100.0, front_exp, back_exp)
        assert roll < 0  # contango

    def test_flat_curve_zero(self):
        front_exp = pd.Timestamp("2024-03-15")
        back_exp = pd.Timestamp("2024-06-15")
        assert futures_roll_return(100.0, 100.0, front_exp, back_exp) == pytest.approx(0.0)

    def test_back_before_front_raises(self):
        with pytest.raises(ValueError, match="back_expiry"):
            futures_roll_return(105.0, 100.0,
                                pd.Timestamp("2024-06-15"), pd.Timestamp("2024-03-15"))

    def test_equal_expiries_raises(self):
        d = pd.Timestamp("2024-03-15")
        with pytest.raises(ValueError, match="back_expiry"):
            futures_roll_return(105.0, 100.0, d, d)

    def test_non_positive_back_price_raises(self):
        with pytest.raises(ValueError, match="back_price"):
            futures_roll_return(105.0, 0.0,
                                pd.Timestamp("2024-03-15"), pd.Timestamp("2024-06-15"))


# ---------------------------------------------------------------------------
# trs_breakeven_spread
# ---------------------------------------------------------------------------

class TestTRSBreakevenSpread:

    def test_fair_future_gives_zero_spread(self):
        f = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0)
        spread = trs_breakeven_spread(f, 100.0, 0.05, 1.0, 0.02)
        assert spread == pytest.approx(0.0, abs=1e-12)

    def test_rich_future_positive_spread(self):
        fair = cost_of_carry_fair_value(100.0, 0.05, 0.02, 1.0)
        spread = trs_breakeven_spread(fair * 1.01, 100.0, 0.05, 1.0, 0.02)
        assert spread > 0

    def test_equals_implied_repo_minus_rate(self):
        spread = trs_breakeven_spread(104.0, 100.0, 0.05, 1.0, 0.02)
        expected = implied_repo_rate(104.0, 100.0, 0.02, 1.0) - 0.05
        assert spread == pytest.approx(expected)

    def test_zero_time_raises(self):
        with pytest.raises(ValueError, match="time_to_expiry"):
            trs_breakeven_spread(104.0, 100.0, 0.05, 0.0, 0.02)


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------

def test_pricing_has_no_beacon_imports():
    """The pricing module must not depend on other Beacon modules."""
    import inspect

    import beacon.derivatives.pricing as pricing

    source = inspect.getsource(pricing)
    assert "from beacon" not in source
    assert "import beacon" not in source
