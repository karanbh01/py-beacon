"""Tests for ETF tracking performance delegation."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.backtest.result import BacktestResult
from beacon.fund.etf import ETF
from beacon.portfolio.base import Portfolio
from beacon.index.result import IndexResult


def _etf():
    return ETF(
        fund_id="etf-1",
        etf_ticker="ETF",
        target_index_definition=SimpleNamespace(index_name="Synthetic Index"),
        index_agent=MagicMock(),
        portfolio=Portfolio("p", initial_cash=1000.0),
        data_provider=MagicMock(),
    )


def _index_result(dates):
    return IndexResult(
        index_id="idx",
        index_levels=pd.Series([100.0, 102.0, 104.0], index=dates),
        divisor_history=pd.Series(1.0, index=dates),
        constituent_snapshots={dates[0]: ["A"]},
        weight_snapshots={dates[0]: {"A": 1.0}},
    )


def _backtest_result(with_target=True):
    dates = pd.bdate_range("2025-01-02", periods=3)
    return BacktestResult(
        portfolio_id="p",
        initial_capital=1000.0,
        portfolio_nav=pd.Series([1000.0, 1015.0, 1030.0], index=dates),
        cash_history=pd.Series([0.0, 0.0, 0.0], index=dates),
        transactions=[],
        actual_weight_history=pd.DataFrame(index=dates),
        target_index_result=_index_result(dates) if with_target else None,
    )


def test_get_tracking_performance_uses_backtest_result_methods():
    etf = _etf()
    result = _backtest_result(with_target=True)

    metrics = etf.get_tracking_performance(result)

    assert set(metrics) == {"tracking_error", "tracking_difference"}
    assert metrics["tracking_error"] == pytest.approx(result.get_tracking_error())
    assert metrics["tracking_difference"] == pytest.approx(result.get_tracking_difference())


def test_get_tracking_performance_returns_error_without_target_index():
    metrics = _etf().get_tracking_performance(_backtest_result(with_target=False))

    assert "error" in metrics


def test_get_tracking_performance_requires_backtest_result_like_object():
    with pytest.raises(TypeError):
        _etf().get_tracking_performance(object())


def test_get_tracking_performance_rejects_missing_result():
    with pytest.raises(ValueError):
        _etf().get_tracking_performance(None)
