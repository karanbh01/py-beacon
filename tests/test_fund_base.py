import pandas as pd
import pytest
from unittest.mock import MagicMock

from beacon.asset.equity import Equity
from beacon.fund.base import IndexFund
from beacon.portfolio.base import Portfolio


def _mock_data_provider(prices: dict):
    provider = MagicMock()

    def _fetch(identifier, start=None, end=None, columns=None):
        asset_prices = prices.get(identifier, {})
        if start in asset_prices:
            return pd.DataFrame({"CLOSE": [asset_prices[start]]})
        return pd.DataFrame()

    provider.fetch_market_data = MagicMock(side_effect=_fetch)
    return provider


def _make_fund(portfolio, data_provider, management_fee_bps=0):
    date = pd.Timestamp("2025-01-02")
    asset_a = Equity(name="Asset A", currency="USD", ticker="A", exchange="NYSE")
    asset_b = Equity(name="Asset B", currency="USD", ticker="B", exchange="NYSE")

    index_definition = MagicMock()
    index_definition.index_name = "Test Index"

    index_agent = MagicMock()
    index_agent._get_universe.return_value = [asset_a, asset_b]
    index_agent.select_constituents.return_value = [asset_a, asset_b]
    index_agent.calculate_constituent_weights.return_value = {
        asset_a: 0.6,
        asset_b: 0.4,
    }

    fund = IndexFund(
        fund_id="fund",
        target_index_definition=index_definition,
        index_agent=index_agent,
        portfolio=portfolio,
        data_provider=data_provider,
        management_fee_bps=management_fee_bps,
    )
    return fund, index_agent, date


def test_rebalance_to_index_delegates_execution_to_backtest_engine():
    data_provider = _mock_data_provider(
        {
            "A": {"2025-01-02": 100.0},
            "B": {"2025-01-02": 50.0},
        }
    )
    portfolio = Portfolio(portfolio_id="fund_portfolio", initial_cash=10_000.0)
    fund, index_agent, date = _make_fund(portfolio, data_provider)

    fund.rebalance_to_index(date)

    index_agent._get_universe.assert_called_once_with(date)
    index_agent.select_constituents.assert_called_once()
    index_agent.calculate_constituent_weights.assert_called_once()
    assert fund.portfolio is portfolio
    assert len(portfolio.transactions) == 2
    assert portfolio.holdings["A"].quantity == pytest.approx(60.0)
    assert portfolio.holdings["B"].quantity == pytest.approx(80.0)
    assert portfolio.cash_balance == pytest.approx(0.0)
    assert fund.calculate_nav(date) == pytest.approx(10_000.0)


def test_calculate_nav_applies_daily_management_fee():
    data_provider = _mock_data_provider({"A": {"2025-01-02": 100.0}})
    portfolio = Portfolio(portfolio_id="fund_portfolio", initial_cash=10_000.0)
    portfolio.execute_buy(
        asset_id="A",
        quantity=100.0,
        price=100.0,
        date=pd.Timestamp("2025-01-02"),
    )
    fund, _, date = _make_fund(portfolio, data_provider, management_fee_bps=252)

    assert fund.calculate_nav(date) == pytest.approx(9_999.0)
