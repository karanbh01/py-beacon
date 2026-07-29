# tests/test_derivatives_swaps.py
"""Unit tests for beacon.derivatives.swaps.TotalReturnSwap."""
import pandas as pd
import pytest

from beacon.derivatives.base import DerivativeBase
from beacon.derivatives.swaps import TotalReturnSwap

NOTIONAL = 10_000_000.0
START = "2024-01-01"
END = "2025-01-01"
# 90 days after START -> ACT/360 fraction = 0.25
VAL_DATE = pd.Timestamp("2024-03-31")


def _make(**overrides):
    kwargs = {
        "derivative_id": "TRS1",
        "underlying_id": "SPX",
        "currency": "USD",
        "start_date": START,
        "end_date": END,
        "notional": NOTIONAL,
        "spread_bps": 50.0,          # 0.50%
        "reference_rate": "SOFR",
        "payment_frequency": "QUARTERLY",
        "reset_type": "UNFUNDED",
    }
    kwargs.update(overrides)
    return TotalReturnSwap(**kwargs)


def _days_between(a,
                  b):
    return (pd.Timestamp(b) - pd.Timestamp(a)).days


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_is_derivative_base(self):
        assert isinstance(_make(), DerivativeBase)

    def test_stores_fields(self):
        t = _make()
        assert t.start_date == pd.Timestamp(START)
        assert t.end_date == pd.Timestamp(END)
        assert t.expiry_date == pd.Timestamp(END)   # base expiry maps to end_date
        assert t.notional == NOTIONAL
        assert t.spread == pytest.approx(0.005)
        assert t.reset_type == "UNFUNDED"
        assert t.payment_frequency == "QUARTERLY"

    def test_reset_type_uppercased(self):
        assert _make(reset_type="funded").reset_type == "FUNDED"

    def test_payment_frequency_uppercased(self):
        assert _make(payment_frequency="monthly").payment_frequency == "MONTHLY"

    def test_invalid_payment_frequency_raises(self):
        with pytest.raises(ValueError, match="payment_frequency"):
            _make(payment_frequency="DAILY")

    def test_invalid_reset_type_raises(self):
        with pytest.raises(ValueError, match="reset_type"):
            _make(reset_type="COLLATERALISED")

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end_date must be after"):
            _make(start_date="2025-01-01", end_date="2024-01-01")

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError, match="notional"):
            _make(notional=0.0)


# ---------------------------------------------------------------------------
# financing_cost
# ---------------------------------------------------------------------------

class TestFinancingCost:

    def test_unfunded_accrual_hand_calc(self):
        # rate = ref + spread = 0.05 + 0.005 = 0.055; dcf = 90/360 = 0.25
        # financing = 10,000,000 * 0.055 * 0.25 = 137,500
        t = _make()
        cost = t.financing_cost(VAL_DATE, pd.Timestamp(START), 0.05)
        assert cost == pytest.approx(137_500.0)

    def test_funded_accrues_spread_only(self):
        # FUNDED -> rate = spread = 0.005; 10,000,000 * 0.005 * 0.25 = 12,500
        t = _make(reset_type="FUNDED")
        cost = t.financing_cost(VAL_DATE, pd.Timestamp(START), 0.05)
        assert cost == pytest.approx(12_500.0)

    def test_zero_period_zero_cost(self):
        t = _make()
        assert t.financing_cost(pd.Timestamp(START), pd.Timestamp(START), 0.05) == 0.0

    def test_negative_period_raises(self):
        t = _make()
        with pytest.raises(ValueError, match="on or after"):
            t.financing_cost(pd.Timestamp("2023-12-01"), pd.Timestamp(START), 0.05)

    def test_scales_with_days(self):
        t = _make()
        half = t.financing_cost(pd.Timestamp("2024-02-15"), pd.Timestamp(START), 0.05)
        full = t.financing_cost(VAL_DATE, pd.Timestamp(START), 0.05)
        assert full > half


# ---------------------------------------------------------------------------
# fair_value
# ---------------------------------------------------------------------------

class TestFairValue:

    def test_receiver_pnl_hand_calc(self):
        # S_t/S_0 = 110/100 -> TR leg = 10,000,000 * 0.10 = 1,000,000
        # financing = 137,500 -> net = 862,500
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        fv = t.fair_value(110.0, VAL_DATE, md)
        assert fv == pytest.approx(1_000_000.0 - 137_500.0)

    def test_negative_return_plus_financing(self):
        # Price falls 5% -> TR leg = -500,000; net = -500,000 - 137,500
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        fv = t.fair_value(95.0, VAL_DATE, md)
        assert fv == pytest.approx(-500_000.0 - 137_500.0)

    def test_flat_price_is_negative_financing(self):
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        assert t.fair_value(100.0, VAL_DATE, md) == pytest.approx(-137_500.0)

    def test_default_initial_price_gives_zero_return(self):
        # No initial_price -> S_0 = spot -> TR leg = 0, only financing accrues.
        t = _make()
        md = {"reference_rate": 0.05, "last_reset_date": pd.Timestamp(START)}
        assert t.fair_value(120.0, VAL_DATE, md) == pytest.approx(-137_500.0)

    def test_non_positive_initial_price_raises(self):
        t = _make()
        with pytest.raises(ValueError, match="initial_price"):
            t.fair_value(110.0, VAL_DATE, {"initial_price": 0.0})


# ---------------------------------------------------------------------------
# mark_to_market
# ---------------------------------------------------------------------------

class TestMarkToMarket:

    def test_keys_present(self):
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        mtm = t.mark_to_market(0.0, 110.0, VAL_DATE, md)
        assert set(mtm) == {"total_return_leg", "financing_leg", "net_mtm", "accrued_days"}

    def test_leg_decomposition(self):
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        mtm = t.mark_to_market(0.0, 110.0, VAL_DATE, md)
        assert mtm["total_return_leg"] == pytest.approx(1_000_000.0)
        assert mtm["financing_leg"] == pytest.approx(137_500.0)
        assert mtm["net_mtm"] == pytest.approx(862_500.0)
        assert mtm["accrued_days"] == _days_between(START, VAL_DATE)

    def test_net_mtm_matches_fair_value(self):
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        mtm = t.mark_to_market(0.0, 108.0, VAL_DATE, md)
        fv = t.fair_value(108.0, VAL_DATE, md)
        assert mtm["net_mtm"] == pytest.approx(fv)

    def test_accrued_days(self):
        t = _make()
        md = {"initial_price": 100.0, "reference_rate": 0.05,
              "last_reset_date": pd.Timestamp(START)}
        mtm = t.mark_to_market(0.0, 110.0, VAL_DATE, md)
        assert mtm["accrued_days"] == 90
