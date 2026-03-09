# tests/test_backtest_result.py
"""Unit tests for BacktestResult and BacktestAssetView."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from beacon.backtest.result import BacktestResult
from beacon.backtest.asset_view import BacktestAssetView
from beacon.index.result import IndexResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nav(values, start="2025-01-02", freq="B"):
    """Build a NAV Series from a list of values."""
    dates = pd.bdate_range(start=start, periods=len(values), freq=freq)
    return pd.Series(values, index=dates, dtype=float)


def _make_weight_history(weights_per_asset, start="2025-01-02"):
    """Build a weight DataFrame.

    weights_per_asset: dict mapping asset_id -> list of weights (same length).
    """
    n = len(next(iter(weights_per_asset.values())))
    dates = pd.bdate_range(start=start, periods=n, freq="B")
    data = {f"{aid}_weight": ws for aid, ws in weights_per_asset.items()}
    return pd.DataFrame(data, index=dates)


def _make_result(**overrides):
    """Create a BacktestResult with sensible defaults."""
    defaults = dict(
        portfolio_id="test_bt",
        initial_capital=10000.0,
        portfolio_nav=_make_nav([10000, 10100, 10200, 10150, 10300]),
        cash_history=_make_nav([5000, 4000, 3000, 3000, 2500]),
        transactions=[],
        actual_weight_history=_make_weight_history(
            {"AAPL": [0.3, 0.31, 0.32, 0.31, 0.33],
             "MSFT": [0.2, 0.19, 0.18, 0.19, 0.17]},
        ),
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


def _make_index_result(values):
    """Create a minimal IndexResult for tracking tests."""
    levels = _make_nav(values)
    return IndexResult(
        index_id="target_idx",
        index_levels=levels,
        divisor_history=pd.Series(1.0, index=levels.index),
        constituent_snapshots={},
        weight_snapshots={},
    )


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_fields(self):
        r = _make_result()
        assert r.portfolio_id == "test_bt"
        assert r.initial_capital == 10000.0
        assert len(r.portfolio_nav) == 5
        assert r.target_index_result is None
        assert r._data_fetcher is None

    def test_repr(self):
        r = _make_result()
        s = repr(r)
        assert "test_bt" in s
        assert "dates=5" in s
        assert "transactions=0" in s
        assert "target_index=False" in s
        assert "data_bound=False" in s

    def test_repr_with_target_and_data(self):
        idx = _make_index_result([100, 101, 102, 103, 104])
        r = _make_result(target_index_result=idx)
        r.with_data(MagicMock())
        s = repr(r)
        assert "target_index=True" in s
        assert "data_bound=True" in s

    def test_data_fetcher_excluded_from_repr(self):
        r = _make_result()
        r.with_data(MagicMock())
        assert "_data_fetcher" not in repr(r)


class TestWithData:

    def test_returns_self(self):
        r = _make_result()
        fetcher = MagicMock()
        assert r.with_data(fetcher) is r

    def test_binds_fetcher(self):
        r = _make_result()
        fetcher = MagicMock()
        r.with_data(fetcher)
        assert r._data_fetcher is fetcher


class TestAsset:

    def test_raises_without_data(self):
        r = _make_result()
        with pytest.raises(RuntimeError, match="No DataFetcher bound"):
            r.asset("AAPL")

    def test_raises_for_unknown_asset(self):
        r = _make_result()
        r.with_data(MagicMock())
        with pytest.raises(KeyError, match="not found"):
            r.asset("UNKNOWN")

    def test_returns_backtest_asset_view(self):
        r = _make_result()
        r.with_data(MagicMock())
        view = r.asset("AAPL")
        assert isinstance(view, BacktestAssetView)
        assert view.asset_id == "AAPL"


class TestGetReturns:

    def test_returns_series(self):
        r = _make_result()
        returns = r.get_returns()
        assert len(returns) == 4  # 5 NAV values -> 4 returns
        expected_first = (10100 / 10000) - 1
        assert returns.iloc[0] == pytest.approx(expected_first)

    def test_empty_nav(self):
        r = _make_result(portfolio_nav=pd.Series(dtype=float))
        returns = r.get_returns()
        assert returns.empty


class TestTrackingError:

    def test_none_without_target(self):
        r = _make_result()
        assert r.get_tracking_error() is None

    def test_zero_for_identical_series(self):
        nav_values = [10000, 10100, 10200, 10150, 10300]
        idx = _make_index_result(nav_values)
        r = _make_result(target_index_result=idx)
        te = r.get_tracking_error()
        assert te == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_divergent_series(self):
        idx = _make_index_result([10000, 10200, 10400, 10100, 10500])
        r = _make_result(target_index_result=idx)
        te = r.get_tracking_error()
        assert te is not None
        assert te > 0


class TestTrackingDifference:

    def test_none_without_target(self):
        r = _make_result()
        assert r.get_tracking_difference() is None

    def test_zero_for_identical_series(self):
        nav_values = [10000, 10100, 10200, 10150, 10300]
        idx = _make_index_result(nav_values)
        r = _make_result(target_index_result=idx)
        td = r.get_tracking_difference()
        assert td == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_portfolio_outperforms(self):
        # Portfolio gains more than index
        idx = _make_index_result([10000, 10050, 10100, 10050, 10100])
        r = _make_result()  # NAV ends at 10300
        r.target_index_result = idx
        td = r.get_tracking_difference()
        assert td is not None
        assert td > 0

    def test_negative_when_portfolio_underperforms(self):
        idx = _make_index_result([10000, 10200, 10500, 10800, 11000])
        r = _make_result()  # NAV ends at 10300
        r.target_index_result = idx
        td = r.get_tracking_difference()
        assert td is not None
        assert td < 0


class TestSummary:

    def test_keys_without_target(self):
        r = _make_result()
        s = r.summary()
        assert "total_return" in s
        assert "annualised_return" in s
        assert "volatility" in s
        assert "sharpe_ratio" in s
        assert "max_drawdown" in s
        assert "tracking_error" not in s
        assert "tracking_difference" not in s

    def test_keys_with_target(self):
        idx = _make_index_result([10000, 10100, 10200, 10150, 10300])
        r = _make_result(target_index_result=idx)
        s = r.summary()
        assert "tracking_error" in s
        assert "tracking_difference" in s

    def test_total_return(self):
        r = _make_result()
        s = r.summary()
        # 10300 / 10000 - 1 = 0.03
        assert s["total_return"] == pytest.approx(0.03)

    def test_max_drawdown(self):
        # NAV: 10000, 10100, 10200, 10150, 10300
        # Drawdown from peak 10200 to 10150 = (10150-10200)/10200
        r = _make_result()
        s = r.summary()
        expected_dd = (10150 - 10200) / 10200
        assert s["max_drawdown"] == pytest.approx(expected_dd)

    def test_volatility_positive(self):
        r = _make_result()
        s = r.summary()
        assert s["volatility"] > 0

    def test_empty_nav_summary(self):
        r = _make_result(portfolio_nav=pd.Series(dtype=float))
        s = r.summary()
        assert s["total_return"] == 0.0
        assert s["volatility"] == 0.0


# ---------------------------------------------------------------------------
# BacktestAssetView tests
# ---------------------------------------------------------------------------

class TestBacktestAssetView:

    def _make_view(self):
        fetcher = MagicMock()
        wh = _make_weight_history(
            {"AAPL": [0.3, 0.31, 0.32, 0.0, 0.33]},
        )
        nav = _make_nav([10000, 10100, 10200, 10150, 10300])
        return BacktestAssetView("AAPL", fetcher, wh, nav)

    def test_repr(self):
        view = self._make_view()
        assert "BacktestAssetView" in repr(view)
        assert "AAPL" in repr(view)

    def test_weight_series_excludes_zero(self):
        view = self._make_view()
        ws = view.weight_series()
        assert len(ws) == 4  # one zero weight excluded
        assert all(w > 0 for w in ws)

    def test_weight_on_date_returns_value(self):
        view = self._make_view()
        dates = pd.bdate_range(start="2025-01-02", periods=5, freq="B")
        w = view.weight_on_date(dates[0])
        assert w == pytest.approx(0.3)

    def test_weight_on_date_returns_none_for_zero(self):
        view = self._make_view()
        dates = pd.bdate_range(start="2025-01-02", periods=5, freq="B")
        w = view.weight_on_date(dates[3])  # weight is 0.0
        assert w is None

    def test_weight_on_date_returns_none_before_history(self):
        view = self._make_view()
        w = view.weight_on_date(pd.Timestamp("2020-01-01"))
        assert w is None

    def test_weight_series_unknown_asset(self):
        fetcher = MagicMock()
        wh = _make_weight_history({"AAPL": [0.3, 0.31]})
        nav = _make_nav([10000, 10100])
        view = BacktestAssetView("UNKNOWN", fetcher, wh, nav)
        assert view.weight_series().empty
