# tests/test_derivatives_futures.py
"""Unit tests for beacon.derivatives.futures.IndexFuture."""
import math

import pandas as pd
import pytest

from beacon.derivatives.base import DerivativeBase
from beacon.derivatives.futures import IndexFuture
from beacon.derivatives.pricing import cost_of_carry_fair_value

# ---------------------------------------------------------------------------
# Helpers — an E-mini-style contract with T = 1 year (ACT/365)
# ---------------------------------------------------------------------------

VAL_DATE = pd.Timestamp("2023-01-01")
EXPIRY = "2024-01-01"  # 365 days after VAL_DATE -> T = 1.0
MARKET = {"risk_free_rate": 0.05, "dividend_yield": 0.02}


def _make(**overrides):
    kwargs = {
        "derivative_id": "ESZ4",
        "underlying_id": "SPX",
        "currency": "USD",
        "expiry_date": EXPIRY,
        "contract_multiplier": 50.0,
        "tick_size": 0.25,
        "tick_value": 12.5,
    }
    kwargs.update(overrides)
    return IndexFuture(**kwargs)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_is_derivative_base(self):
        assert isinstance(_make(), DerivativeBase)

    def test_stores_fields(self):
        f = _make()
        assert f.underlying_type == "INDEX"
        assert f.contract_multiplier == 50.0
        assert f.tick_size == 0.25
        assert f.tick_value == 12.5

    @pytest.mark.parametrize("field", ["contract_multiplier", "tick_size", "tick_value"])
    def test_non_positive_contract_params_raise(self,
                                                field):
        with pytest.raises(ValueError, match=field):
            _make(**{field: 0.0})

    def test_inherits_base_validation(self):
        with pytest.raises(ValueError, match="currency"):
            _make(currency="")


# ---------------------------------------------------------------------------
# fair_value
# ---------------------------------------------------------------------------

class TestFairValue:

    def test_cost_of_carry_value(self):
        # F = 4000 * exp((0.05 - 0.02) * 1) = 4000 * exp(0.03)
        f = _make()
        fv = f.fair_value(4000.0, VAL_DATE, MARKET)
        assert fv == pytest.approx(4000.0 * math.exp(0.03))

    def test_delegates_to_pricing_function(self):
        f = _make()
        fv = f.fair_value(4000.0, VAL_DATE, MARKET)
        expected = cost_of_carry_fair_value(4000.0, 0.05, 0.02, 1.0)
        assert fv == pytest.approx(expected)

    def test_borrow_cost_applied(self):
        f = _make()
        fv = f.fair_value(4000.0, VAL_DATE, {**MARKET, "borrow_cost": 0.01})
        assert fv == pytest.approx(4000.0 * math.exp(0.04))

    def test_missing_market_data_defaults_to_zero(self):
        f = _make()
        assert f.fair_value(4000.0, VAL_DATE, {}) == pytest.approx(4000.0)

    def test_at_expiry_returns_spot(self):
        f = _make()
        assert f.fair_value(4000.0, pd.Timestamp(EXPIRY), MARKET) == pytest.approx(4000.0)


# ---------------------------------------------------------------------------
# basis / annualised_basis
# ---------------------------------------------------------------------------

class TestBasis:

    def test_basis_positive_in_contango(self):
        f = _make()
        assert f.basis(4030.0, 4000.0) == pytest.approx(30.0)

    def test_basis_negative_in_backwardation(self):
        f = _make()
        assert f.basis(3980.0, 4000.0) == pytest.approx(-20.0)

    def test_annualised_basis_recovers_financing_rate(self):
        # F priced with r=0.05, q=0 -> ln(F/S)/T = 0.05
        f = _make()
        fut = 4000.0 * math.exp(0.05)
        assert f.annualised_basis(fut, 4000.0, VAL_DATE) == pytest.approx(0.05)

    def test_annualised_basis_negative_deep_backwardation(self):
        f = _make()
        rate = f.annualised_basis(3600.0, 4000.0, VAL_DATE)  # future well below spot
        assert rate < 0

    def test_annualised_basis_at_expiry_raises(self):
        f = _make()
        with pytest.raises(ValueError, match="time_to_expiry"):
            f.annualised_basis(4000.0, 4000.0, pd.Timestamp(EXPIRY))


# ---------------------------------------------------------------------------
# daily_settlement_pnl
# ---------------------------------------------------------------------------

class TestDailySettlementPnl:

    def test_long_gain(self):
        f = _make()
        # (4005 - 4000) * 50 * 2 = 500
        assert f.daily_settlement_pnl(4005.0, 4000.0, contracts=2) == pytest.approx(500.0)

    def test_default_one_contract(self):
        f = _make()
        assert f.daily_settlement_pnl(4005.0, 4000.0) == pytest.approx(250.0)

    def test_short_position(self):
        f = _make()
        # short 1 contract, price up 5 -> loss of 250
        assert f.daily_settlement_pnl(4005.0, 4000.0, contracts=-1) == pytest.approx(-250.0)

    def test_price_unchanged_zero_pnl(self):
        f = _make()
        assert f.daily_settlement_pnl(4000.0, 4000.0, contracts=10) == 0.0


# ---------------------------------------------------------------------------
# roll_cost
# ---------------------------------------------------------------------------

class TestRollCost:

    def test_contango_positive(self):
        f = _make()
        assert f.roll_cost(4000.0, 4020.0) == pytest.approx(20.0)

    def test_backwardation_negative(self):
        f = _make()
        assert f.roll_cost(4000.0, 3985.0) == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# mark_to_market
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_keys_present(self):
        f = _make()
        mtm = f.mark_to_market(4130.0, 4000.0, VAL_DATE, MARKET)
        assert set(mtm) == {"fair_value", "basis", "theoretical_edge", "time_to_expiry"}

    def test_values(self):
        f = _make()
        mtm = f.mark_to_market(4130.0, 4000.0, VAL_DATE, MARKET)
        fair = 4000.0 * math.exp(0.03)
        assert mtm["fair_value"] == pytest.approx(fair)
        assert mtm["basis"] == pytest.approx(130.0)          # market - spot
        assert mtm["theoretical_edge"] == pytest.approx(fair - 4130.0)
        assert mtm["time_to_expiry"] == pytest.approx(1.0)

    def test_at_expiry(self):
        f = _make()
        mtm = f.mark_to_market(4005.0, 4000.0, pd.Timestamp(EXPIRY), MARKET)
        assert mtm["fair_value"] == pytest.approx(4000.0)
        assert mtm["time_to_expiry"] == 0.0
        assert mtm["theoretical_edge"] == pytest.approx(4000.0 - 4005.0)
