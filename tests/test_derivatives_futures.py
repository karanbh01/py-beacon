"""Tests for future contract pricing."""

import math

import pytest

from beacon.derivatives import ETFFuture, IndexFuture
from beacon.derivatives.pricing import present_value_discrete_dividends


def test_etf_future_extends_index_future():
    assert issubclass(ETFFuture, IndexFuture)


def test_index_future_prices_with_continuous_yield_model():
    future = IndexFuture(
        contract_id="idx-fut-1",
        underlying_ticker="SPX",
        maturity_date="2026-12-31",
        market_data={
            "spot_price": 100.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "time_to_maturity": 1.5,
        },
    )

    assert future.fair_value() == pytest.approx(100.0 * math.exp((0.05 - 0.02) * 1.5))


def test_etf_future_uses_discrete_dividend_model_when_dividends_are_provided():
    future = ETFFuture(
        contract_id="etf-fut-1",
        underlying_ticker="SPY",
        maturity_date="2026-12-31",
        market_data={
            "spot_price": 100.0,
            "risk_free_rate": 0.05,
            "dividend_yield": 0.02,
            "time_to_maturity": 1.0,
            "discrete_dividends": [(0.25, 1.0), (0.75, 1.0)],
        },
    )

    dividend_pv = present_value_discrete_dividends([(0.25, 1.0), (0.75, 1.0)], 0.05, 1.0)
    expected = (100.0 - dividend_pv) * math.exp(0.05 * 1.0)

    assert future.fair_value() == pytest.approx(expected)
    assert future.fair_value() != pytest.approx(100.0 * math.exp((0.05 - 0.02) * 1.0))


def test_etf_future_falls_back_to_continuous_yield_model_without_discrete_dividends():
    market_data = {
        "spot_price": 100.0,
        "risk_free_rate": 0.05,
        "dividend_yield": 0.02,
        "time_to_maturity": 1.0,
    }
    etf_future = ETFFuture(
        contract_id="etf-fut-2",
        underlying_ticker="SPY",
        maturity_date="2026-12-31",
        market_data=market_data,
    )
    index_future = IndexFuture(
        contract_id="idx-fut-2",
        underlying_ticker="SPX",
        maturity_date="2026-12-31",
        market_data=market_data,
    )

    assert etf_future.fair_value() == pytest.approx(index_future.fair_value())


def test_discrete_dividend_pricing_ignores_dividends_outside_tenor():
    future = ETFFuture(
        contract_id="etf-fut-3",
        underlying_ticker="SPY",
        maturity_date="2026-12-31",
        market_data={
            "spot_price": 100.0,
            "risk_free_rate": 0.05,
            "time_to_maturity": 1.0,
            "discrete_dividends": [(0.5, 1.0), (1.5, 100.0)],
        },
    )

    expected = (100.0 - math.exp(-0.05 * 0.5)) * math.exp(0.05)

    assert future.fair_value() == pytest.approx(expected)
