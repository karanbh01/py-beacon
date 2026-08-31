# tests/test_etf_tracking.py
"""Unit tests for ETF.get_tracking_performance() over a BacktestResult.

Verifies the refactored method (which delegates to BacktestResult's tracking
methods) yields the same numbers as the previous ETFAnalytics-based path for
the same inputs.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.etf.analytics import ETFAnalytics
from beacon.backtest.result import BacktestResult
from beacon.fund.etf import ETF
from beacon.index.result import IndexResult
from beacon.portfolio.base import Portfolio

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DATES = pd.bdate_range(start="2024-01-02", periods=40, freq="B")


def _make_nav_and_levels():
    """Build a portfolio NAV series and an index-level series that drift apart.

    Both share the same date index so return alignment is a no-op, letting the
    BacktestResult methods and the ETFAnalytics functions be compared directly.
    """
    n = len(DATES)
    # Index compounds smoothly; NAV lags slightly with a little noise.
    idx = 100.0 * np.cumprod(1 + np.full(n, 0.001))
    drift = np.linspace(0.0, 0.002, n)          # widening tracking gap
    wobble = 0.0005 * np.sin(np.arange(n))       # deterministic noise
    nav = 100.0 * np.cumprod(1 + (0.001 - drift + wobble))

    nav_series = pd.Series(nav, index=DATES, name="nav")
    level_series = pd.Series(idx, index=DATES, name="level")
    return nav_series, level_series


def _make_backtest_result(nav_series,
                          level_series,
                          with_target=True):
    target = None
    if with_target:
        target = IndexResult(
            index_id="IDX",
            index_levels=level_series,
            divisor_history=pd.Series(1.0, index=level_series.index),
            constituent_snapshots={DATES[0]: ["A"]},
            weight_snapshots={DATES[0]: {"A": 1.0}},
        )
    # A portfolio whose recorded NAV follows the series exactly: one
    # holding marked to the NAV, cash zero, written through the history
    # recorder the engine drives.
    from beacon.backtest.result import Book
    from beacon.portfolio.base import Holding

    eve = nav_series.index[0] - pd.tseries.offsets.BDay(1)
    portfolio = Portfolio(portfolio_id="etf_pf", initial_cash=100.0,
                          inception=eve)

    for date, value in nav_series.items():
        holding = Holding(asset_id="NAV", quantity=1.0,
                          average_cost_price=float(value),
                          current_price=float(value),
                          market_value=float(value))
        portfolio._history.record(date, {"NAV": holding}, 0.0)

    return BacktestResult(
        portfolio=portfolio,
        index=Book.from_index(target) if target is not None else None,
    )


def _make_etf():
    return ETF(
        fund_id="ETF1",
        etf_ticker="TEST",
        target_index_definition=MagicMock(index_name="Test Index"),
        index_agent=MagicMock(),
        portfolio=Portfolio("etf_pf", initial_cash=100.0),
        data_provider=MagicMock(),
        management_fee_bps=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetTrackingPerformance:

    def test_returns_tracking_metrics(self):
        nav, levels = _make_nav_and_levels()
        result = _make_backtest_result(nav, levels)
        perf = _make_etf().get_tracking_performance(result)
        assert set(perf) == {"tracking_error", "tracking_difference"}
        assert isinstance(perf["tracking_error"], float)
        assert isinstance(perf["tracking_difference"], float)

    def test_matches_etf_analytics_implementation(self):
        """New BacktestResult-based path must equal the old ETFAnalytics path."""
        nav, levels = _make_nav_and_levels()
        result = _make_backtest_result(nav, levels)
        perf = _make_etf().get_tracking_performance(result)

        # Reproduce the previous implementation's inputs: aligned periodic returns.
        etf_returns = nav.pct_change().dropna()
        index_returns = levels.pct_change().dropna()
        aligned = pd.DataFrame({"etf": etf_returns, "idx": index_returns}).dropna()

        analytics = ETFAnalytics()
        expected_te = analytics.calculate_tracking_error(aligned["etf"], aligned["idx"])
        expected_td = analytics.calculate_tracking_difference(aligned["etf"], aligned["idx"])

        assert perf["tracking_error"] == pytest.approx(expected_te)
        assert perf["tracking_difference"] == pytest.approx(expected_td)

    def test_matches_backtest_result_methods(self):
        nav, levels = _make_nav_and_levels()
        result = _make_backtest_result(nav, levels)
        perf = _make_etf().get_tracking_performance(result)
        assert perf["tracking_error"] == pytest.approx(result.get_tracking_error())
        assert perf["tracking_difference"] == pytest.approx(result.get_tracking_difference())

    def test_error_when_no_target_index(self):
        nav, levels = _make_nav_and_levels()
        result = _make_backtest_result(nav, levels, with_target=False)
        perf = _make_etf().get_tracking_performance(result)
        assert "error" in perf
        assert "tracking_error" not in perf

    def test_raises_on_none_result(self):
        with pytest.raises(ValueError, match="BacktestResult must be provided"):
            _make_etf().get_tracking_performance(None)
