# tests/test_backtest_engine.py
"""Unit tests for the rewritten BacktestEngine."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from beacon.backtest.engine import BacktestEngine
from beacon.backtest.result import BacktestResult
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
