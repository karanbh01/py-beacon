"""Tests for IndexFuture pricing and analytics."""

import math

import pandas as pd
import pytest

from beacon.derivatives import IndexFuture
from beacon.derivatives.pricing import cost_of_carry_fair_value


def _future(**overrides):
    kwargs = {
        "derivative_id": "ifut-1",
        "underlying_id": "SPX",
        "currency": "USD",
        "expiry_date": "2026-01-02",
        "contract_multiplier": 50.0,
        "tick_size": 0.25,
        "tick_value": 12.5,
    }
    kwargs.update(overrides)
    return IndexFuture(**kwargs)


def test_constructor_stores_contract_terms():
    future = _future()

    assert future.derivative_id == "ifut-1"
    assert future.underlying_id == "SPX"
    assert future.currency == "USD"
    assert future.expiry_date == pd.Timestamp("2026-01-02")
    assert future.contract_multiplier == pytest.approx(50.0)
    assert future.tick_size == pytest.approx(0.25)
    assert future.tick_value == pytest.approx(12.5)


def test_fair_value_uses_cost_of_carry_pricing_helper():
    future = _future(expiry_date="2026-07-02")
    valuation_date = pd.Timestamp("2025-01-02")
    time_to_expiry = future.time_to_expiry(valuation_date)

    assert future.fair_value(5000.0, valuation_date, 0.05, 0.015, 0.002) == pytest.approx(
        cost_of_carry_fair_value(5000.0, 0.05, 0.015, time_to_expiry, 0.002)
    )


def test_fair_value_at_expiry_returns_spot_price():
    future = _future(expiry_date="2026-01-02")

    assert future.fair_value(5000.0, pd.Timestamp("2026-01-02"), 0.05, 0.02) == pytest.approx(5000.0)


def test_basis_and_deep_backwardation():
    future = _future()

    assert future.basis(4700.0, 5000.0) == pytest.approx(-300.0)
    assert future.annualised_basis(4700.0, 5000.0, pd.Timestamp("2025-01-02")) < 0.0


def test_annualised_basis_matches_log_f_over_s_over_t():
    future = _future(expiry_date="2026-01-02")
    valuation_date = pd.Timestamp("2025-01-02")
    time_to_expiry = future.time_to_expiry(valuation_date)

    assert future.annualised_basis(5100.0, 5000.0, valuation_date) == pytest.approx(
        math.log(5100.0 / 5000.0) / time_to_expiry
    )


def test_annualised_basis_returns_zero_at_expiry():
    assert _future(expiry_date="2026-01-02").annualised_basis(5100.0, 5000.0, pd.Timestamp("2026-01-02")) == pytest.approx(0.0)


def test_daily_settlement_pnl_uses_multiplier_and_contracts():
    assert _future(contract_multiplier=50.0).daily_settlement_pnl(5100.0, 5080.0, contracts=3) == pytest.approx(3000.0)
    assert _future(contract_multiplier=50.0).daily_settlement_pnl(5080.0, 5100.0, contracts=2) == pytest.approx(-2000.0)


def test_roll_cost_is_back_minus_front():
    assert _future().roll_cost(5000.0, 5025.0) == pytest.approx(25.0)
    assert _future().roll_cost(5025.0, 5000.0) == pytest.approx(-25.0)


def test_mark_to_market_returns_expected_metrics():
    future = _future(expiry_date="2026-01-02")
    valuation_date = pd.Timestamp("2025-01-02")
    fair_value = future.fair_value(5000.0, valuation_date, 0.05, 0.02)

    mtm = future.mark_to_market(5125.0, 5000.0, valuation_date, 0.05, 0.02)

    assert mtm["fair_value"] == pytest.approx(fair_value)
    assert mtm["basis"] == pytest.approx(125.0)
    assert mtm["theoretical_edge"] == pytest.approx(5125.0 - fair_value)
    assert mtm["time_to_expiry"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("derivative_id", ""),
        ("underlying_id", ""),
        ("currency", ""),
        ("contract_multiplier", 0.0),
        ("tick_size", 0.0),
        ("tick_value", 0.0),
    ],
)
def test_constructor_rejects_invalid_contract_terms(field, value):
    with pytest.raises(ValueError):
        _future(**{field: value})
