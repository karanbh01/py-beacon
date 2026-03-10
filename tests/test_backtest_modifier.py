# tests/test_backtest_modifier.py
"""Unit tests for BacktestModifier, DriftThresholdModifier, and engine integration."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.backtest.rules import BacktestModifier, DriftThresholdModifier
from beacon.backtest.engine import BacktestEngine, TradeInstruction
from beacon.portfolio.base import Portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATES = pd.bdate_range(start="2025-01-02", periods=5, freq="B")


def _mock_data_provider(prices, col="CLOSE"):
    provider = MagicMock()

    def _fetch(identifier, start=None, end=None, columns=None):
        asset_prices = prices.get(identifier, {})
        if start in asset_prices:
            return pd.DataFrame({col: [asset_prices[start]]})
        return pd.DataFrame()

    provider.fetch_market_data = MagicMock(side_effect=_fetch)
    return provider


# ---------------------------------------------------------------------------
# DriftThresholdModifier — construction
# ---------------------------------------------------------------------------

class TestDriftThresholdConstruction:

    def test_valid_threshold(self):
        m = DriftThresholdModifier(threshold=0.05)
        assert m.threshold == 0.05

    def test_zero_threshold(self):
        m = DriftThresholdModifier(threshold=0.0)
        assert m.threshold == 0.0

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DriftThresholdModifier(threshold=-0.01)


# ---------------------------------------------------------------------------
# DriftThresholdModifier — should_skip_rebalance
# ---------------------------------------------------------------------------

class TestDriftThresholdSkip:

    def test_skips_when_drift_within_threshold(self):
        m = DriftThresholdModifier(threshold=0.10)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 50, 100.0)
        # A is 50% (5000/10000), target 55% -> drift 5% < 10%
        assert m.should_skip_rebalance(DATES[0], portfolio, {"A": 0.55})

    def test_does_not_skip_when_drift_exceeds_threshold(self):
        m = DriftThresholdModifier(threshold=0.05)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 50, 100.0)
        # A is 50%, target 100% -> drift 50% > 5%
        assert not m.should_skip_rebalance(DATES[0], portfolio, {"A": 1.0})

    def test_skips_when_exactly_at_threshold(self):
        m = DriftThresholdModifier(threshold=0.10)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 40, 100.0)
        # A is 40% (4000/10000), target 50% -> drift 10% == threshold
        assert m.should_skip_rebalance(DATES[0], portfolio, {"A": 0.50})

    def test_considers_new_assets_not_in_portfolio(self):
        m = DriftThresholdModifier(threshold=0.05)
        portfolio = Portfolio("p", initial_cash=10000.0)
        # No holdings. Target has A at 50% -> drift is 50%
        assert not m.should_skip_rebalance(DATES[0], portfolio, {"A": 0.50})

    def test_considers_assets_not_in_target(self):
        m = DriftThresholdModifier(threshold=0.05)
        portfolio = Portfolio("p", initial_cash=10000.0)
        portfolio.execute_buy("A", 100, 100.0)
        # A is 100%, target is empty -> drift 100%
        assert not m.should_skip_rebalance(DATES[0], portfolio, {})

    def test_empty_portfolio_and_target_skips(self):
        m = DriftThresholdModifier(threshold=0.05)
        portfolio = Portfolio("p", initial_cash=10000.0)
        assert m.should_skip_rebalance(DATES[0], portfolio, {})


# ---------------------------------------------------------------------------
# DriftThresholdModifier — adjust_trades
# ---------------------------------------------------------------------------

class TestDriftThresholdAdjustTrades:

    def test_pass_through(self):
        m = DriftThresholdModifier(threshold=0.05)
        trades = [
            TradeInstruction("A", "BUY", 10, 100.0, 0.0),
            TradeInstruction("B", "SELL", 5, 50.0, 0.0),
        ]
        portfolio = Portfolio("p", initial_cash=10000.0)
        result = m.adjust_trades(trades, DATES[0], portfolio)
        assert result is trades


# ---------------------------------------------------------------------------
# Engine integration with modifier
# ---------------------------------------------------------------------------

class TestEngineModifierIntegration:

    def test_modifier_skips_rebalance(self):
        """With a high drift threshold, rebalance is skipped."""
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        # Threshold so high it will always skip
        modifier = DriftThresholdModifier(threshold=10.0)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w,
                                modifiers=[modifier])
        result = engine.run()
        # No trades should happen
        assert len(result.transactions) == 0
        assert result.cash_history.iloc[-1] == pytest.approx(10000.0)

    def test_modifier_allows_rebalance(self):
        """With a tiny drift threshold, rebalance always proceeds."""
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        modifier = DriftThresholdModifier(threshold=0.0)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w,
                                modifiers=[modifier])
        result = engine.run()
        assert len(result.transactions) > 0
        # Should be fully invested
        assert result.cash_history.iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_custom_modifier_adjust_trades(self):
        """A custom modifier that filters out sells."""

        class NoSellModifier(BacktestModifier):
            def should_skip_rebalance(self, date, portfolio, target_weights):
                return False

            def adjust_trades(self, trades, date, portfolio):
                return [t for t in trades if t.side != "SELL"]

        prices = {
            "A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES},
            "B": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES},
        }
        dp = _mock_data_provider(prices)
        w = {
            DATES[0]: {"A": 1.0},
            DATES[2]: {"B": 1.0},  # would normally sell A, buy B
        }
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w,
                                modifiers=[NoSellModifier()])
        result = engine.run()
        # A should still be held since sells were filtered out
        sell_txns = [t for t in result.transactions if t.transaction_type == "SELL"]
        assert len(sell_txns) == 0

    def test_no_modifiers_by_default(self):
        prices = {"A": {d.strftime("%Y-%m-%d"): 100.0 for d in DATES}}
        dp = _mock_data_provider(prices)
        w = {DATES[0]: {"A": 1.0}}
        engine = BacktestEngine(str(DATES[0].date()), str(DATES[-1].date()),
                                10000.0, dp, target_weights=w)
        assert engine.modifiers == []
