"""Tests for derivative base abstractions."""

from typing import Dict

import pandas as pd
import pytest

from beacon.derivatives import DerivativeBase


class ConcreteDerivative(DerivativeBase):
    def fair_value(
        self,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> float:
        return spot_price * self.notional

    def mark_to_market(
        self,
        market_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> Dict[str, float]:
        fair_value = self.fair_value(spot_price, valuation_date, market_data)
        market_value = market_price * self.notional
        return {"fair_value": fair_value, "market_value": market_value, "pnl": market_value - fair_value}


def _derivative(**overrides):
    kwargs = {
        "derivative_id": "deriv-1",
        "underlying_id": "SPY",
        "underlying_type": "ETF",
        "currency": "USD",
        "expiry_date": "2026-01-02",
        "notional": 100.0,
    }
    kwargs.update(overrides)
    return ConcreteDerivative(**kwargs)


def test_derivative_base_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DerivativeBase("d", "u", "ETF", "USD", "2026-01-02", 100.0)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("derivative_id", "", "derivative_id"),
        ("underlying_id", "", "underlying_id"),
        ("underlying_type", "", "underlying_type"),
        ("currency", "", "currency"),
        ("notional", 0.0, "notional"),
        ("notional", -1.0, "notional"),
    ],
)
def test_construction_validates_required_fields(field, value, error):
    with pytest.raises(ValueError, match=error):
        _derivative(**{field: value})


def test_construction_stores_required_fields_and_normalizes_types():
    derivative = _derivative(expiry_date="2026-01-02", notional=25)

    assert derivative.derivative_id == "deriv-1"
    assert derivative.underlying_id == "SPY"
    assert derivative.underlying_type == "ETF"
    assert derivative.currency == "USD"
    assert derivative.expiry_date == pd.Timestamp("2026-01-02")
    assert derivative.notional == pytest.approx(25.0)


def test_time_to_expiry_uses_act_365_and_floors_after_expiry():
    derivative = _derivative(expiry_date="2026-01-02")

    assert derivative.time_to_expiry(pd.Timestamp("2025-01-02")) == pytest.approx(365 / 365.0)
    assert derivative.time_to_expiry(pd.Timestamp("2026-01-03")) == pytest.approx(0.0)


def test_subclass_implements_fair_value_and_mark_to_market_contract():
    derivative = _derivative(notional=10.0)

    assert derivative.fair_value(50.0, pd.Timestamp("2025-01-02"), {}) == pytest.approx(500.0)
    assert derivative.mark_to_market(52.0, 50.0, pd.Timestamp("2025-01-02"), {}) == {
        "fair_value": 500.0,
        "market_value": 520.0,
        "pnl": 20.0,
    }
