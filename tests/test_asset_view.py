# tests/test_asset_view.py
"""Unit tests for AssetView."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.asset.view import AssetView


@pytest.fixture
def mock_fetcher():
    return MagicMock()


@pytest.fixture
def view(mock_fetcher):
    return AssetView(asset_id="AAPL", data_fetcher=mock_fetcher)


# ── Construction ───────────────────────────────────────────────────────


class TestConstruction:
    def test_valid(self, mock_fetcher):
        v = AssetView("AAPL", mock_fetcher)
        assert v.asset_id == "AAPL"

    def test_empty_asset_id_raises(self, mock_fetcher):
        with pytest.raises(ValueError, match="asset_id cannot be empty"):
            AssetView("", mock_fetcher)

    def test_none_fetcher_raises(self):
        with pytest.raises(ValueError, match="data_fetcher must be provided"):
            AssetView("AAPL", None)

    def test_repr(self, view):
        assert repr(view) == "AssetView(asset_id='AAPL')"


# ── prices ─────────────────────────────────────────────────────────────


class TestPrices:
    def test_delegates_to_fetcher(self, view, mock_fetcher):
        view.prices("2025-01-01", "2025-06-30")
        mock_fetcher.fetch_market_data.assert_called_once_with("AAPL", "2025-01-01", "2025-06-30")

    def test_returns_fetcher_result(self, view, mock_fetcher):
        expected = pd.DataFrame({"CLOSE": [100, 101]})
        mock_fetcher.fetch_market_data.return_value = expected
        result = view.prices("2025-01-01", "2025-01-02")
        pd.testing.assert_frame_equal(result, expected)


# ── returns ────────────────────────────────────────────────────────────


class TestReturns:
    def _make_price_df(self, prices, freq="D"):
        dates = pd.date_range("2025-01-01", periods=len(prices), freq=freq)
        return pd.DataFrame({"CLOSE": prices}, index=dates)

    def test_daily_returns(self, view, mock_fetcher):
        mock_fetcher.fetch_market_data.return_value = self._make_price_df([100, 110, 121])
        result = view.returns("2025-01-01", "2025-01-03", frequency="daily")
        assert len(result) == 2
        assert abs(result.iloc[0] - 0.1) < 1e-9
        assert abs(result.iloc[1] - 0.1) < 1e-9

    def test_weekly_returns(self, view, mock_fetcher):
        # 10 daily prices, resample to weekly
        mock_fetcher.fetch_market_data.return_value = self._make_price_df(list(range(100, 110)))
        result = view.returns("2025-01-01", "2025-01-10", frequency="weekly")
        assert len(result) > 0

    def test_monthly_returns(self, view, mock_fetcher):
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices = list(range(100, 160))
        mock_fetcher.fetch_market_data.return_value = pd.DataFrame({"CLOSE": prices}, index=dates)
        result = view.returns("2025-01-01", "2025-03-01", frequency="monthly")
        assert len(result) > 0

    def test_unsupported_frequency_raises(self, view):
        with pytest.raises(ValueError, match="Unsupported frequency"):
            view.returns("2025-01-01", "2025-01-10", frequency="quarterly")

    def test_empty_prices_returns_empty_series(self, view, mock_fetcher):
        mock_fetcher.fetch_market_data.return_value = pd.DataFrame()
        result = view.returns("2025-01-01", "2025-01-10")
        assert result.empty

    def test_custom_price_column(self, view, mock_fetcher):
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        mock_fetcher.fetch_market_data.return_value = pd.DataFrame(
            {"ADJ_CLOSE": [100, 110, 121]}, index=dates
        )
        result = view.returns("2025-01-01", "2025-01-03", price_column="ADJ_CLOSE")
        mock_fetcher.fetch_market_data.assert_called_with("AAPL", "2025-01-01", "2025-01-03", columns=["ADJ_CLOSE"])
        assert len(result) == 2


# ── reference_data ─────────────────────────────────────────────────────


class TestReferenceData:
    def test_delegates_to_fetcher(self, view, mock_fetcher):
        view.reference_data("2025-01-01")
        mock_fetcher.fetch_reference_data.assert_called_once_with("AAPL", "2025-01-01")

    def test_no_date(self, view, mock_fetcher):
        view.reference_data()
        mock_fetcher.fetch_reference_data.assert_called_once_with("AAPL", None)


# ── corporate_actions ──────────────────────────────────────────────────


class TestCorporateActions:
    def test_delegates_to_fetcher(self, view, mock_fetcher):
        view.corporate_actions("2025-01-01", "2025-12-31")
        mock_fetcher.fetch_market_data.assert_called_once_with(
            "AAPL", "2025-01-01", "2025-12-31", columns=["DIVIDEND", "SPLIT"]
        )
