# tests/test_derivatives_etf_future.py
"""Unit tests for beacon.derivatives.futures.ETFFuture."""
import math
import pytest
import pandas as pd

from beacon.derivatives.futures import IndexFuture, ETFFuture
from beacon.derivatives.base import DerivativeBase
from beacon.derivatives.pricing import (
    discrete_dividend_fair_value,
    cost_of_carry_fair_value,
)


VAL_DATE = pd.Timestamp("2023-01-01")
EXPIRY = "2024-01-01"  # T = 1.0 (ACT/365)


def _make(**overrides):
    kwargs = dict(
        derivative_id="SPYZ4",
        underlying_id="SPY",
        currency="USD",
        expiry_date=EXPIRY,
        contract_multiplier=100.0,
        tick_size=0.01,
        tick_value=1.0,
    )
    kwargs.update(overrides)
    return ETFFuture(**kwargs)


# ---------------------------------------------------------------------------
# Construction / hierarchy
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_extends_index_future(self):
        f = _make()
        assert isinstance(f, IndexFuture)
        assert isinstance(f, DerivativeBase)

    def test_underlying_type_is_etf(self):
        assert _make().underlying_type == "ETF"

    def test_inherits_contract_params(self):
        f = _make()
        assert f.contract_multiplier == 100.0
        assert f.tick_size == 0.01
        assert f.tick_value == 1.0

    def test_inherits_validation(self):
        with pytest.raises(ValueError, match="contract_multiplier"):
            _make(contract_multiplier=0.0)


# ---------------------------------------------------------------------------
# fair_value — discrete dividend model
# ---------------------------------------------------------------------------

class TestFairValueDiscrete:

    def test_uses_discrete_dividend_model(self):
        f = _make()
        divs = [(0.5, 4.0)]
        md = {"risk_free_rate": 0.05, "discrete_dividends": divs}
        fv = f.fair_value(400.0, VAL_DATE, md)
        expected = discrete_dividend_fair_value(400.0, 0.05, 1.0, divs)
        assert fv == pytest.approx(expected)

    def test_discrete_dividend_lowers_price(self):
        f = _make()
        no_div = f.fair_value(400.0, VAL_DATE, {"risk_free_rate": 0.05})
        with_div = f.fair_value(
            400.0, VAL_DATE,
            {"risk_free_rate": 0.05, "discrete_dividends": [(0.5, 6.0)]},
        )
        assert with_div < no_div

    def test_dividends_after_expiry_ignored(self):
        f = _make()
        base = f.fair_value(400.0, VAL_DATE, {"risk_free_rate": 0.05,
                                              "discrete_dividends": []})
        late = f.fair_value(400.0, VAL_DATE,
                            {"risk_free_rate": 0.05,
                             "discrete_dividends": [(2.0, 10.0)]})  # past expiry
        assert late == pytest.approx(base)

    def test_at_expiry_returns_spot(self):
        f = _make()
        fv = f.fair_value(400.0, pd.Timestamp(EXPIRY),
                          {"risk_free_rate": 0.05,
                           "discrete_dividends": [(0.5, 4.0)]})
        assert fv == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# fair_value — fallback to continuous model
# ---------------------------------------------------------------------------

class TestFairValueFallback:

    def test_no_discrete_key_falls_back_to_continuous(self):
        f = _make()
        md = {"risk_free_rate": 0.05, "dividend_yield": 0.02}
        fv = f.fair_value(400.0, VAL_DATE, md)
        # Matches the inherited IndexFuture continuous model.
        expected = IndexFuture.fair_value(f, 400.0, VAL_DATE, md)
        assert fv == pytest.approx(expected)
        assert fv == pytest.approx(cost_of_carry_fair_value(400.0, 0.05, 0.02, 1.0))

    def test_empty_discrete_list_falls_back(self):
        f = _make()
        md = {"risk_free_rate": 0.05, "dividend_yield": 0.02, "discrete_dividends": []}
        fv = f.fair_value(400.0, VAL_DATE, md)
        assert fv == pytest.approx(cost_of_carry_fair_value(400.0, 0.05, 0.02, 1.0))


# ---------------------------------------------------------------------------
# Discrete vs continuous comparison
# ---------------------------------------------------------------------------

class TestDiscreteVsContinuous:

    def test_discrete_and_continuous_both_discount_vs_zero_dividend(self):
        f = _make()
        zero_div = f.fair_value(400.0, VAL_DATE, {"risk_free_rate": 0.05})
        continuous = f.fair_value(
            400.0, VAL_DATE, {"risk_free_rate": 0.05, "dividend_yield": 0.02})
        discrete = f.fair_value(
            400.0, VAL_DATE,
            {"risk_free_rate": 0.05, "discrete_dividends": [(0.5, 8.0)]})
        assert continuous < zero_div
        assert discrete < zero_div

    def test_discrete_differs_from_continuous(self):
        """A ~2% discrete dividend does not price identically to a 2% yield."""
        f = _make()
        continuous = f.fair_value(
            400.0, VAL_DATE, {"risk_free_rate": 0.05, "dividend_yield": 0.02})
        discrete = f.fair_value(
            400.0, VAL_DATE,
            {"risk_free_rate": 0.05, "discrete_dividends": [(0.5, 8.0)]})
        assert discrete != pytest.approx(continuous, abs=1e-6)

    def test_discrete_takes_precedence_over_yield(self):
        """When both are supplied, the discrete model is used (yield ignored)."""
        f = _make()
        md = {"risk_free_rate": 0.05, "dividend_yield": 0.02,
              "discrete_dividends": [(0.5, 4.0)]}
        fv = f.fair_value(400.0, VAL_DATE, md)
        expected = discrete_dividend_fair_value(400.0, 0.05, 1.0, [(0.5, 4.0)])
        assert fv == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Inherited analytics still work with the overridden fair_value
# ---------------------------------------------------------------------------

class TestInheritedBehaviour:

    def test_mark_to_market_uses_discrete_fair_value(self):
        f = _make()
        md = {"risk_free_rate": 0.05, "discrete_dividends": [(0.5, 4.0)]}
        mtm = f.mark_to_market(410.0, 400.0, VAL_DATE, md)
        expected_fv = discrete_dividend_fair_value(400.0, 0.05, 1.0, [(0.5, 4.0)])
        assert mtm["fair_value"] == pytest.approx(expected_fv)
        assert mtm["basis"] == pytest.approx(10.0)
        assert mtm["theoretical_edge"] == pytest.approx(expected_fv - 410.0)
        assert mtm["time_to_expiry"] == pytest.approx(1.0)

    def test_daily_settlement_pnl_inherited(self):
        f = _make()
        # (401 - 400) * 100 * 3 = 300
        assert f.daily_settlement_pnl(401.0, 400.0, contracts=3) == pytest.approx(300.0)
