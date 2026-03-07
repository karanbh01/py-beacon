# tests/test_index_asset_view.py
"""Unit tests for IndexAssetView."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.index.asset_view import IndexAssetView
from beacon.index.result import IndexResult
from beacon.asset.view import AssetView


@pytest.fixture
def weight_snapshots():
    return {
        pd.Timestamp("2025-01-01"): {"AAPL": 0.6, "MSFT": 0.4},
        pd.Timestamp("2025-04-01"): {"AAPL": 0.5, "GOOG": 0.5},
    }


@pytest.fixture
def index_levels():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    return pd.Series([1000, 1010, 1005, 1020, 1030], index=dates)


@pytest.fixture
def mock_fetcher():
    return MagicMock()


@pytest.fixture
def view(mock_fetcher, weight_snapshots, index_levels):
    return IndexAssetView(
        asset_id="AAPL",
        data_fetcher=mock_fetcher,
        weight_snapshots=weight_snapshots,
        index_levels=index_levels,
    )


# ── Construction ───────────────────────────────────────────────────────


class TestConstruction:
    def test_inherits_asset_view(self, view):
        assert isinstance(view, AssetView)

    def test_asset_id(self, view):
        assert view.asset_id == "AAPL"

    def test_repr(self, view):
        assert repr(view) == "IndexAssetView(asset_id='AAPL')"

    def test_base_methods_available(self, view, mock_fetcher):
        view.prices("2025-01-01", "2025-01-31")
        mock_fetcher.fetch_market_data.assert_called_once()


# ── weight_on_date ─────────────────────────────────────────────────────


class TestWeightOnDate:
    def test_exact_rebalance_date(self, view):
        assert view.weight_on_date(pd.Timestamp("2025-01-01")) == 0.6

    def test_between_rebalances(self, view):
        assert view.weight_on_date(pd.Timestamp("2025-02-15")) == 0.6

    def test_after_second_rebalance(self, view):
        assert view.weight_on_date(pd.Timestamp("2025-04-15")) == 0.5

    def test_before_any_rebalance(self, view):
        assert view.weight_on_date(pd.Timestamp("2024-12-31")) is None

    def test_asset_not_in_snapshot(self, mock_fetcher, weight_snapshots, index_levels):
        v = IndexAssetView("MSFT", mock_fetcher, weight_snapshots, index_levels)
        # MSFT is in first rebalance but not second
        assert v.weight_on_date(pd.Timestamp("2025-01-15")) == 0.4
        assert v.weight_on_date(pd.Timestamp("2025-04-15")) is None


# ── weight_series ──────────────────────────────────────────────────────


class TestWeightSeries:
    def test_returns_series(self, view):
        ws = view.weight_series()
        assert isinstance(ws, pd.Series)
        assert len(ws) == 2  # AAPL in both snapshots
        assert ws[pd.Timestamp("2025-01-01")] == 0.6
        assert ws[pd.Timestamp("2025-04-01")] == 0.5

    def test_excludes_absent_dates(self, mock_fetcher, weight_snapshots, index_levels):
        v = IndexAssetView("MSFT", mock_fetcher, weight_snapshots, index_levels)
        ws = v.weight_series()
        assert len(ws) == 1  # MSFT only in first snapshot
        assert pd.Timestamp("2025-04-01") not in ws.index

    def test_sorted_by_date(self, view):
        ws = view.weight_series()
        assert list(ws.index) == sorted(ws.index)


# ── contribution ───────────────────────────────────────────────────────


class TestContribution:
    def test_contribution_calculation(self, view, mock_fetcher):
        dates = pd.date_range("2025-01-01", periods=4, freq="B")
        mock_fetcher.fetch_market_data.return_value = pd.DataFrame(
            {"CLOSE": [100, 110, 121, 133.1]}, index=dates
        )
        contrib = view.contribution("2025-01-01", "2025-01-06")
        # Returns: 0.1, 0.1, 0.1
        # Weights (shifted): NaN, 0.6, 0.6
        # Contribution: 0.06, 0.06
        assert len(contrib) == 2
        assert abs(contrib.iloc[0] - 0.06) < 1e-9
        assert abs(contrib.iloc[1] - 0.06) < 1e-9

    def test_empty_returns(self, view, mock_fetcher):
        mock_fetcher.fetch_market_data.return_value = pd.DataFrame()
        contrib = view.contribution("2025-01-01", "2025-01-06")
        assert contrib.empty


# ── IndexResult.asset() integration ────────────────────────────────────


class TestIndexResultIntegration:
    def test_returns_index_asset_view(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        r = IndexResult(
            index_id="IDX",
            index_levels=pd.Series([100, 101, 102], index=dates),
            divisor_history=pd.Series([1, 1, 1], index=dates),
            constituent_snapshots={pd.Timestamp("2025-01-01"): ["AAPL"]},
            weight_snapshots={pd.Timestamp("2025-01-01"): {"AAPL": 1.0}},
        )
        r.with_data(MagicMock())
        view = r.asset("AAPL")
        assert isinstance(view, IndexAssetView)
        assert view.asset_id == "AAPL"
        assert view.weight_on_date(pd.Timestamp("2025-01-01")) == 1.0
