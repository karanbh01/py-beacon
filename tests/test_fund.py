"""Tests for IndexFund and ETF."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.fund.base import IndexFund
from beacon.fund.etf import ETF
from beacon.portfolio.base import Portfolio


class SyntheticDataFetcher:
    """Minimal DataFetcher test double returning deterministic close prices."""

    def __init__(self, prices):
        self.prices = prices

    def fetch_prices(self, ticker, start_date, end_date):
        dates = pd.bdate_range(start=start_date, end=end_date)
        value = self.prices[ticker]
        if isinstance(value, dict):
            closes = [value.get(date.strftime("%Y-%m-%d"), value[next(iter(value))]) for date in dates]
        else:
            closes = [value for _ in dates]
        return pd.DataFrame(
            {"Close": closes, "Adj Close": closes},
            index=dates,
        )


class SyntheticAnalytics:
    def calculate_tracking_error(self, etf_returns, benchmark_returns):
        return float((etf_returns - benchmark_returns).std())

    def calculate_tracking_difference(self, etf_returns, benchmark_returns):
        return float(((1 + etf_returns).prod() - 1) - ((1 + benchmark_returns).prod() - 1))


def _index_definition(**attrs):
    defaults = {"index_name": "Synthetic Index"}
    defaults.update(attrs)
    return SimpleNamespace(**defaults)


def _index_agent():
    agent = MagicMock()
    agent.select_constituents.return_value = []
    agent.calculate_constituent_weights.return_value = {}
    return agent


def _fund_portfolio():
    portfolio = Portfolio("fund-portfolio", initial_cash=100.0)
    portfolio.execute_buy("AAA", quantity=2, price=50.0, date=pd.Timestamp("2025-01-02"))
    return portfolio


def _fund(**overrides):
    defaults = dict(
        fund_id="fund-1",
        target_index_definition=_index_definition(),
        index_agent=_index_agent(),
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher({"AAA": 55.0, "BENCH": 100.0}),
        management_fee_bps=25,
    )
    defaults.update(overrides)
    return IndexFund(**defaults)


def test_index_fund_constructs_with_valid_parameters():
    fund = _fund()

    assert fund.fund_id == "fund-1"
    assert fund.target_index_definition.index_name == "Synthetic Index"
    assert fund.management_fee_bps == 25
    assert fund.portfolio.portfolio_id == "fund-portfolio"


def test_index_fund_calculate_nav_uses_latest_synthetic_prices():
    fund = _fund(data_provider=SyntheticDataFetcher({"AAA": 60.0}))

    nav = fund.calculate_nav(pd.Timestamp("2025-01-03"))

    assert nav == pytest.approx(120.0)
    assert fund.portfolio.holdings["AAA"].current_price == pytest.approx(60.0)


def test_index_fund_management_fee_does_not_increase_nav():
    current_date = pd.Timestamp("2025-01-03")
    no_fee_fund = _fund(
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher({"AAA": 60.0}),
        management_fee_bps=0,
    )
    fee_fund = _fund(
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher({"AAA": 60.0}),
        management_fee_bps=100,
    )

    assert fee_fund.calculate_nav(current_date) <= no_fee_fund.calculate_nav(current_date)


def test_etf_constructs_with_valid_parameters():
    etf = ETF(
        fund_id="etf-1",
        etf_ticker="ETF",
        target_index_definition=_index_definition(),
        index_agent=_index_agent(),
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher({"AAA": 50.0, "BENCH": 100.0}),
        management_fee_bps=10,
        creation_unit_size=25_000,
    )

    assert etf.fund_id == "etf-1"
    assert etf.etf_ticker == "ETF"
    assert etf.creation_unit_size == 25_000
    assert etf.management_fee_bps == 10


def test_etf_simulate_market_price_returns_nav_for_basic_case():
    etf = ETF(
        fund_id="etf-1",
        etf_ticker="ETF",
        target_index_definition=_index_definition(),
        index_agent=_index_agent(),
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher({"AAA": 62.5, "BENCH": 100.0}),
    )

    market_price = etf.simulate_market_price(pd.Timestamp("2025-01-03"))

    assert market_price == pytest.approx(125.0)
    assert etf.market_price == pytest.approx(125.0)


def test_etf_get_tracking_performance_returns_expected_metrics():
    etf = ETF(
        fund_id="etf-1",
        etf_ticker="ETF",
        target_index_definition=_index_definition(benchmark_ticker_for_tracking="BENCH"),
        index_agent=_index_agent(),
        portfolio=_fund_portfolio(),
        data_provider=SyntheticDataFetcher(
            {
                "AAA": {
                    "2025-01-02": 50.0,
                    "2025-01-03": 55.0,
                    "2025-01-06": 60.0,
                },
                "BENCH": {
                    "2025-01-02": 100.0,
                    "2025-01-03": 104.0,
                    "2025-01-06": 108.0,
                },
            }
        ),
    )

    metrics = etf.get_tracking_performance(
        "2025-01-02",
        "2025-01-06",
        analysis_module=SyntheticAnalytics(),
    )

    assert set(metrics) == {"tracking_error", "tracking_difference"}
    assert metrics["tracking_error"] >= 0
    assert metrics["tracking_difference"] == pytest.approx(0.2 - 0.08)
