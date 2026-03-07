# tests/test_index_result.py
"""Unit tests for IndexResult."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.index.result import IndexResult


@pytest.fixture
def sample_result():
    dates = pd.date_range("2025-01-01", periods=5, freq="B")
    rebal_date = pd.Timestamp("2025-01-01")
    return IndexResult(
        index_id="TEST_IDX",
        index_levels=pd.Series([1000, 1010, 1005, 1020, 1030], index=dates),
        divisor_history=pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=dates),
        constituent_snapshots={rebal_date: ["AAPL", "MSFT", "GOOG"]},
        weight_snapshots={rebal_date: {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}},
    )


# ── Construction & repr ────────────────────────────────────────────────


class TestConstruction:
    def test_fields(self, sample_result):
        assert sample_result.index_id == "TEST_IDX"
        assert len(sample_result.index_levels) == 5
        assert len(sample_result.divisor_history) == 5
        assert "AAPL" in sample_result.constituent_snapshots[pd.Timestamp("2025-01-01")]

    def test_repr(self, sample_result):
        r = repr(sample_result)
        assert "TEST_IDX" in r
        assert "dates=5" in r
        assert "rebalances=1" in r
        assert "data_bound=False" in r

    def test_data_fetcher_excluded_from_repr(self, sample_result):
        sample_result.with_data(MagicMock())
        r = repr(sample_result)
        assert "data_bound=True" in r
        assert "DataFetcher" not in r


# ── with_data ──────────────────────────────────────────────────────────


class TestWithData:
    def test_returns_self(self, sample_result):
        fetcher = MagicMock()
        result = sample_result.with_data(fetcher)
        assert result is sample_result

    def test_binds_fetcher(self, sample_result):
        fetcher = MagicMock()
        sample_result.with_data(fetcher)
        assert sample_result._data_fetcher is fetcher


# ── asset ──────────────────────────────────────────────────────────────


class TestAsset:
    def test_raises_without_data(self, sample_result):
        with pytest.raises(RuntimeError, match="No DataFetcher bound"):
            sample_result.asset("AAPL")

    def test_raises_for_unknown_asset(self, sample_result):
        sample_result.with_data(MagicMock())
        with pytest.raises(KeyError, match="not found"):
            sample_result.asset("UNKNOWN")

    def test_returns_asset_view(self, sample_result):
        sample_result.with_data(MagicMock())
        view = sample_result.asset("AAPL")
        assert view.asset_id == "AAPL"

    def test_finds_asset_across_snapshots(self):
        dates = pd.date_range("2025-01-01", periods=2, freq="B")
        r = IndexResult(
            index_id="X",
            index_levels=pd.Series([100, 101], index=dates),
            divisor_history=pd.Series([1.0, 1.0], index=dates),
            constituent_snapshots={
                pd.Timestamp("2025-01-01"): ["A"],
                pd.Timestamp("2025-01-02"): ["B"],
            },
            weight_snapshots={},
        )
        r.with_data(MagicMock())
        assert r.asset("A").asset_id == "A"
        assert r.asset("B").asset_id == "B"


# ── get_returns ────────────────────────────────────────────────────────


class TestGetReturns:
    def test_returns_series(self, sample_result):
        returns = sample_result.get_returns()
        assert len(returns) == 4  # 5 levels -> 4 returns
        assert abs(returns.iloc[0] - 0.01) < 1e-9  # 1000 -> 1010 = 1%

    def test_empty_levels(self):
        r = IndexResult(
            index_id="X",
            index_levels=pd.Series(dtype=float),
            divisor_history=pd.Series(dtype=float),
            constituent_snapshots={},
            weight_snapshots={},
        )
        assert r.get_returns().empty


# ── get_weights_on_date ────────────────────────────────────────────────


class TestGetWeightsOnDate:
    def test_exact_rebalance_date(self, sample_result):
        w = sample_result.get_weights_on_date(pd.Timestamp("2025-01-01"))
        assert w == {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}

    def test_date_after_rebalance(self, sample_result):
        w = sample_result.get_weights_on_date(pd.Timestamp("2025-06-15"))
        assert w == {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}

    def test_date_before_any_rebalance(self, sample_result):
        w = sample_result.get_weights_on_date(pd.Timestamp("2024-12-31"))
        assert w == {}

    def test_multiple_rebalances(self):
        dates = pd.date_range("2025-01-01", periods=3, freq="B")
        r = IndexResult(
            index_id="X",
            index_levels=pd.Series([100, 101, 102], index=dates),
            divisor_history=pd.Series([1, 1, 1], index=dates),
            constituent_snapshots={},
            weight_snapshots={
                pd.Timestamp("2025-01-01"): {"A": 0.6, "B": 0.4},
                pd.Timestamp("2025-04-01"): {"A": 0.5, "C": 0.5},
            },
        )
        assert r.get_weights_on_date(pd.Timestamp("2025-02-15")) == {"A": 0.6, "B": 0.4}
        assert r.get_weights_on_date(pd.Timestamp("2025-04-01")) == {"A": 0.5, "C": 0.5}
        assert r.get_weights_on_date(pd.Timestamp("2025-05-01")) == {"A": 0.5, "C": 0.5}


# ── to_dataframe ───────────────────────────────────────────────────────


class TestToDataframe:
    def test_columns(self, sample_result):
        df = sample_result.to_dataframe()
        assert list(df.columns) == ["index_level", "divisor"]
        assert df.index.name == "date"
        assert len(df) == 5

    def test_values(self, sample_result):
        df = sample_result.to_dataframe()
        assert df["index_level"].iloc[0] == 1000
        assert df["divisor"].iloc[0] == 10.0
