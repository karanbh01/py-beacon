"""Tests for TotalReturnSwap economics."""

import pandas as pd
import pytest

from beacon.derivatives import DerivativeBase, TotalReturnSwap


def _swap(**overrides):
    kwargs = {
        "derivative_id": "trs-1",
        "underlying_id": "SPX",
        "currency": "USD",
        "start_date": "2025-01-02",
        "end_date": "2026-01-02",
        "notional": 1_000_000.0,
        "spread_bps": 25.0,
        "reference_rate": 0.05,
        "payment_frequency": "QUARTERLY",
    }
    kwargs.update(overrides)
    return TotalReturnSwap(**kwargs)


def test_total_return_swap_extends_derivative_base_and_stores_terms():
    swap = _swap(reset_type="FUNDED")

    assert isinstance(swap, DerivativeBase)
    assert swap.derivative_id == "trs-1"
    assert swap.underlying_id == "SPX"
    assert swap.currency == "USD"
    assert swap.start_date == pd.Timestamp("2025-01-02")
    assert swap.expiry_date == pd.Timestamp("2026-01-02")
    assert swap.notional == pytest.approx(1_000_000.0)
    assert swap.spread_rate == pytest.approx(0.0025)
    assert swap.reference_rate == pytest.approx(0.05)
    assert swap.payment_frequency == "QUARTERLY"
    assert swap.reset_type == "FUNDED"


def test_financing_cost_accrues_with_act_365_day_count():
    swap = _swap(notional=1_000_000.0, spread_bps=50.0)

    assert swap.financing_cost(pd.Timestamp("2025-04-02"), pd.Timestamp("2025-01-02"), 0.04) == pytest.approx(
        1_000_000.0 * (0.04 + 0.005) * 90 / 365.0
    )


def test_fair_value_is_total_return_leg_less_accrued_financing():
    swap = _swap(notional=1_000_000.0, spread_bps=25.0, reference_rate=0.05)

    value = swap.fair_value(
        spot_price=110.0,
        valuation_date=pd.Timestamp("2025-04-02"),
        market_data={"initial_spot_price": 100.0, "last_reset_date": pd.Timestamp("2025-01-02")},
    )

    expected_total_return = 1_000_000.0 * (110.0 / 100.0 - 1.0)
    expected_financing = 1_000_000.0 * (0.05 + 0.0025) * 90 / 365.0
    assert value == pytest.approx(expected_total_return - expected_financing)


def test_fair_value_can_be_negative_when_financing_exceeds_return():
    swap = _swap(notional=1_000_000.0, spread_bps=100.0, reference_rate=0.08)

    value = swap.fair_value(
        spot_price=99.0,
        valuation_date=pd.Timestamp("2025-04-02"),
        market_data={"initial_spot_price": 100.0},
    )

    assert value < 0.0


def test_mark_to_market_returns_requested_components():
    swap = _swap(notional=500_000.0, spread_bps=30.0, reference_rate=0.04)

    mtm = swap.mark_to_market(
        market_price=0.0,
        spot_price=105.0,
        valuation_date=pd.Timestamp("2025-02-01"),
        market_data={"initial_spot_price": 100.0, "last_reset_date": pd.Timestamp("2025-01-02")},
    )

    expected_total_return = 500_000.0 * 0.05
    expected_financing = 500_000.0 * (0.04 + 0.003) * 30 / 365.0
    assert mtm == {
        "total_return_leg": pytest.approx(expected_total_return),
        "financing_leg": pytest.approx(expected_financing),
        "net_mtm": pytest.approx(expected_total_return - expected_financing),
        "accrued_days": 30,
    }


def test_mark_to_market_uses_market_data_reference_rate_override():
    swap = _swap(reference_rate=0.01, spread_bps=0.0)

    mtm = swap.mark_to_market(
        market_price=0.0,
        spot_price=100.0,
        valuation_date=pd.Timestamp("2025-02-01"),
        market_data={
            "initial_spot_price": 100.0,
            "last_reset_date": pd.Timestamp("2025-01-02"),
            "reference_rate": 0.06,
        },
    )

    assert mtm["financing_leg"] == pytest.approx(1_000_000.0 * 0.06 * 30 / 365.0)
    assert mtm["net_mtm"] == pytest.approx(-mtm["financing_leg"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("derivative_id", ""),
        ("underlying_id", ""),
        ("currency", ""),
        ("notional", 0.0),
        ("payment_frequency", ""),
        ("reset_type", "BAD"),
    ],
)
def test_constructor_rejects_invalid_inputs(field, value):
    with pytest.raises(ValueError):
        _swap(**{field: value})


def test_constructor_rejects_end_date_before_start_date():
    with pytest.raises(ValueError, match="end_date"):
        _swap(end_date="2024-12-31")


def test_total_return_leg_rejects_non_positive_initial_spot():
    with pytest.raises(ValueError, match="initial_spot_price"):
        _swap().total_return_leg(100.0, 0.0)
