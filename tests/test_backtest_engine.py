# tests/test_backtest_engine.py
"""Unit tests for the rewritten BacktestEngine."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from beacon.backtest.engine import BacktestEngine, TradeInstruction
from beacon.backtest.result import BacktestResult
from beacon.portfolio.base import Portfolio
from beacon.index.result import IndexResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_data_provider(prices: dict, col="CLOSE"):
    """Create a mock DataFetcher.

    prices: dict mapping asset_id -> dict mapping date_str -> price.
    Example: {"AAPL": {"2025-01-02": 100.0, "2025-01-03": 105.0}}
    """
    provider = MagicMock()

    def _fetch(identifier, start=None, end=None, columns=None):
        asset_prices = prices.get(identifier, {})
        if start in asset_prices:
            return pd.DataFrame({col: [asset_prices[start]]})
        return pd.DataFrame()

    provider.fetch_market_data = MagicMock(side_effect=_fetch)
    return provider


def _bday(offset=0, base="2025-01-02"):
    """Return a business day Timestamp."""
    return pd.bdate_range(start=base, periods=offset + 1, freq="B")[-1]


def _make_index_result(weight_snapshots):
    """Create a minimal IndexResult from weight snapshots."""
    dates = sorted(weight_snapshots.keys())
    levels = pd.Series(100.0, index=pd.DatetimeIndex(dates))
    return IndexResult(
        index_id="test_idx",
        index_levels=levels,
        divisor_history=pd.Series(1.0, index=levels.index),
        constituent_snapshots={d: list(w.keys()) for d, w in weight_snapshots.items()},
        weight_snapshots=weight_snapshots,
    )


# 5 business days starting 2025-01-02
DATES = pd.bdate_range(start="2025-01-02", periods=5, freq="B")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_requires_one_weight_source(self):
        dp = MagicMock()
        with pytest.raises(ValueError, match="One of"):
            BacktestEngine("2025-01-02", "2025-01-10", 10000.0, dp)

    def test_rejects_both_weight_sources(self):
        dp = MagicMock()
        idx = _make_index_result({DATES[0]: {"A": 0.5}})
        with pytest.raises(ValueError, match="not both"):
            BacktestEngine("2025-01-02", "2025-01-10", 10000.0, dp,
                           target_index_result=idx,
                           target_weights={DATES[0]: {"A": 0.5}})

    def test_accepts_index_result(self):
        dp = MagicMock()
        idx = _make_index_result({DATES[0]: {"A": 0.5}})
        engine = BacktestEngine("2025-01-02", "2025-01-10", 10000.0, dp,
                                target_index_result=idx)
        assert engine.target_index_result is idx

    def test_accepts_custom_weights(self):
        dp = MagicMock()
        w = {DATES[0]: {"A": 0.5}}
        engine = BacktestEngine("2025-01-02", "2025-01-10", 10000.0, dp,
                                target_weights=w)
        assert engine._weight_schedule is w


# ---------------------------------------------------------------------------
# Run — basic behaviour
# ---------------------------------------------------------------------------

class TestRunBasic:

    def test_returns_backtest_result(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_nav_series_length(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert len(result.portfolio_nav) == len(DATES)

    def test_cash_history_length(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert len(result.cash_history) == len(DATES)

    def test_empty_date_range(self):
        dp = MagicMock()
        w = {DATES[0]: {"A": 1.0}}
        # Weekend range — no business days
        engine = BacktestEngine("2025-01-04", "2025-01-05",
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert result.portfolio_nav.empty

    def test_initial_capital_stored(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert result.initial_capital == 10000.0


# ---------------------------------------------------------------------------
# Rebalancing
# ---------------------------------------------------------------------------

class TestRebalancing:

    def test_fully_invested_on_rebalance_date(self):
        """After rebalance into 100% A, cash should be ~0."""
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        # After first day rebalance, cash should be ~0
        assert result.cash_history.iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_two_asset_split(self):
        """50/50 split between two assets."""
        prices = {
            "A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES},
            "B": {d.strftime("%Y-%m-%d"): 50.0 for d in DATES},
        }
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 0.5, "B": 0.5}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        # NAV should stay at 10000 (prices don't change)
        assert result.portfolio_nav.iloc[-1] == pytest.approx(10000.0, abs=1.0)

    def test_sells_before_buys(self):
        """Rotation from A to B should work — sells free up cash for buys."""
        prices = {
            "A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES},
            "B": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES},
        }
        dp = _mock_data_provider(prices)
        w = {
            DATES[0]: {"A": 1.0},
            DATES[2]: {"B": 1.0},  # rotate from A to B on day 3
        }
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        # After rotation, should still have ~10000 total
        assert result.portfolio_nav.iloc[-1] == pytest.approx(10000.0, abs=1.0)
        # Transactions: buy A, sell A, buy B = 3
        assert len(result.transactions) == 3

    def test_price_appreciation_reflected_in_nav(self):
        """Rising prices should increase NAV."""
        base_price = 100.0
        prices = {"A": {}}
        for i, d in enumerate(DATES):
            prices["A"][d.strftime("%Y-%m-%d")] = base_price + i * 10
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert result.portfolio_nav.iloc[-1] > result.portfolio_nav.iloc[0]

    def test_no_rebalance_without_schedule(self):
        """If rebalance date is outside range, portfolio stays cash."""
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        future = pd.Timestamp("2026-01-02")
        w = {future: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert len(result.transactions) == 0
        assert result.cash_history.iloc[-1] == pytest.approx(10000.0)


# ---------------------------------------------------------------------------
# Weight history
# ---------------------------------------------------------------------------

class TestWeightHistory:

    def test_weight_columns_present(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert "A_weight" in result.actual_weight_history.columns


# ---------------------------------------------------------------------------
# IndexResult integration
# ---------------------------------------------------------------------------

class TestIndexResultIntegration:

    def test_target_index_result_passed_through(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        idx = _make_index_result({DATES[0]: {"A": 1.0}})
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_index_result=idx)
        result = engine.run()
        assert result.target_index_result is idx

    def test_custom_weights_no_target(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        result = engine.run()
        assert result.target_index_result is None


# ---------------------------------------------------------------------------
# _generate_trades
# ---------------------------------------------------------------------------

class TestGenerateTrades:

    def _make_engine(self, prices, target_weights=None, cost_bps=0.0):
        dp = _mock_data_provider(prices)
        w = target_weights or {DATES[0]: {"A": 1.0}}
        return BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                              10000.0, dp, target_weights=w,
                              transaction_cost_bps=cost_bps)

    def test_buy_new_asset(self):
        """Cash-only portfolio -> buy 100% A."""
        prices = {"A": {DATES[0].strftime("%Y-%m-%d"): 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert len(trades) == 1
        assert trades[0].side == "BUY"
        assert trades[0].asset_id == "A"
        assert trades[0].quantity == pytest.approx(100.0)
        assert trades[0].cost == 0.0

    def test_sell_full_position(self):
        """Holding A -> target has no A -> sell all."""
        prices = {"A": {DATES[0].strftime("%Y-%m-%d"): 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=0.0)
        portfolio.execute_buy("A", 50, 100.0)  # need cash first
        # Re-create with cash
        portfolio = Portfolio("p", initial_cash=5000.0)
        portfolio.execute_buy("A", 50, 100.0)
        trades = engine._generate_trades(portfolio, {}, DATES[0])
        assert len(trades) == 1
        assert trades[0].side == "SELL"
        assert trades[0].quantity == pytest.approx(50.0)

    def test_sells_before_buys(self):
        """Trades list has sells first, then buys."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}, "B": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 100, 100.0)
        # Rotate from A to B
        trades = engine._generate_trades(portfolio, {"B": 1.0}, DATES[0])
        sell_indices = [i for i, t in enumerate(trades) if t.side == "SELL"]
        buy_indices = [i for i, t in enumerate(trades) if t.side == "BUY"]
        assert len(sell_indices) > 0
        assert len(buy_indices) > 0
        assert max(sell_indices) < min(buy_indices)

    def test_trim_overweight(self):
        """Asset overweight vs target -> partial sell."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 100, 100.0)
        # A is 100% of portfolio, target 50%
        trades = engine._generate_trades(portfolio, {"A": 0.5}, DATES[0])
        sells = [t for t in trades if t.side == "SELL"]
        assert len(sells) == 1
        assert sells[0].quantity == pytest.approx(50.0)

    def test_buy_underweight(self):
        """Asset underweight vs target -> buy more."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 50, 100.0)
        # A is 50% (5000/10000), target 100%
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        buys = [t for t in trades if t.side == "BUY"]
        assert len(buys) == 1
        assert buys[0].quantity == pytest.approx(50.0)

    def test_no_trades_when_at_target(self):
        """Already at target weight -> no trades."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 100, 100.0)
        # A is 100% of portfolio, target 100%
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert len(trades) == 0

    def test_empty_portfolio_zero_value(self):
        """Zero-value portfolio -> no trades."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=0.0)
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert len(trades) == 0

    def test_missing_price_skips_asset(self):
        """No price available -> asset skipped."""
        engine = self._make_engine({})  # no prices at all
        portfolio = Portfolio("p", initial_cash=10000.0)
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert len(trades) == 0

    def test_transaction_cost_bps_on_buy(self):
        """Transaction costs calculated correctly for buys."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices, cost_bps=10.0)  # 10 bps = 0.1%
        portfolio = Portfolio("p", initial_cash=10000.0)
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert len(trades) == 1
        assert trades[0].side == "BUY"
        # notional = 100 * 100 = 10000, cost = 10000 * 10/10000 = 10.0
        assert trades[0].cost == pytest.approx(10.0)

    def test_transaction_cost_bps_on_sell(self):
        """Transaction costs calculated correctly for sells."""
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices, cost_bps=20.0)  # 20 bps = 0.2%
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 100, 100.0)
        trades = engine._generate_trades(portfolio, {}, DATES[0])
        assert len(trades) == 1
        assert trades[0].side == "SELL"
        # notional = 100 * 100 = 10000, cost = 10000 * 20/10000 = 20.0
        assert trades[0].cost == pytest.approx(20.0)

    def test_transaction_cost_deducted_in_run(self):
        """Integration: costs reduce NAV when running backtest."""
        d_str = {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}
        prices = {"A": d_str}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 0.5}}  # target 50% in A
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w,
                                transaction_cost_bps=100.0)  # 1%
        result = engine.run()
        # Buy 50 shares at $100 = $5000 notional, cost = $50
        # NAV = 5000 (holdings) + 5000 - 5000 - 50 (cash) = 9950
        assert result.portfolio_nav.iloc[-1] == pytest.approx(9950.0, abs=1.0)
        # Verify the transaction cost was recorded
        buy_txns = [t for t in result.transactions if t.transaction_type == "BUY"]
        assert len(buy_txns) == 1
        assert buy_txns[0].transaction_cost == pytest.approx(50.0)

    def test_returns_trade_instruction_type(self):
        d = DATES[0].strftime("%Y-%m-%d")
        prices = {"A": {d: 100.0}}
        engine = self._make_engine(prices)
        portfolio = Portfolio("p", initial_cash=10000.0)
        trades = engine._generate_trades(portfolio, {"A": 1.0}, DATES[0])
        assert all(isinstance(t, TradeInstruction) for t in trades)
