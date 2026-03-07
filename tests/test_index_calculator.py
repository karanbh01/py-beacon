# tests/test_index_calculator.py
"""Unit tests for IndexCalculator._get_universe()."""
import pytest
import pandas as pd
from unittest.mock import MagicMock

from beacon.index.calculation import IndexCalculator
from beacon.asset.equity import Equity


@pytest.fixture
def mock_definition():
    defn = MagicMock()
    defn.index_name = "Test Index"
    defn.currency = "USD"
    defn.base_value = 1000.0
    defn.universe_identifiers = ["AAPL", "MSFT", "GOOG"]
    return defn


@pytest.fixture
def mock_data():
    return MagicMock()


@pytest.fixture
def calculator(mock_definition, mock_data):
    return IndexCalculator(mock_definition, mock_data)


def _make_ref_df(name, currency, exchange):
    return pd.DataFrame(
        {"NAME": [name], "CURRENCY": [currency], "EXCHANGE": [exchange]},
        index=pd.Index(["ID"], name="IDENTIFIER"),
    )


class TestGetUniverse:
    def test_resolves_all_identifiers(self, calculator, mock_data):
        mock_data.fetch_reference_data.side_effect = [
            _make_ref_df("Apple Inc", "USD", "NASDAQ"),
            _make_ref_df("Microsoft", "USD", "NASDAQ"),
            _make_ref_df("Alphabet", "USD", "NASDAQ"),
        ]
        assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 3
        assert all(isinstance(a, Equity) for a in assets)
        assert assets[0].ticker == "AAPL"
        assert assets[0].name == "Apple Inc"
        assert assets[1].ticker == "MSFT"
        assert assets[2].ticker == "GOOG"

    def test_none_universe_returns_empty(self, calculator, mock_data):
        calculator.definition.universe_identifiers = None
        assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert assets == []
        mock_data.fetch_reference_data.assert_not_called()

    def test_skips_unresolvable_identifiers(self, calculator, mock_data, caplog):
        mock_data.fetch_reference_data.side_effect = [
            _make_ref_df("Apple Inc", "USD", "NASDAQ"),
            pd.DataFrame(),  # MSFT not found
            _make_ref_df("Alphabet", "USD", "NASDAQ"),
        ]
        with caplog.at_level("WARNING"):
            assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 2
        assert assets[0].ticker == "AAPL"
        assert assets[1].ticker == "GOOG"
        assert "No reference data for 'MSFT'" in caplog.text

    def test_skips_on_exception(self, calculator, mock_data, caplog):
        mock_data.fetch_reference_data.side_effect = [
            _make_ref_df("Apple Inc", "USD", "NASDAQ"),
            Exception("connection error"),
            _make_ref_df("Alphabet", "USD", "NASDAQ"),
        ]
        with caplog.at_level("WARNING"):
            assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 2
        assert "Failed to resolve 'MSFT'" in caplog.text

    def test_uses_defaults_for_missing_columns(self, calculator, mock_data):
        # Reference data missing NAME and EXCHANGE columns
        mock_data.fetch_reference_data.return_value = pd.DataFrame(
            {"CURRENCY": ["EUR"]},
            index=pd.Index(["AAPL"], name="IDENTIFIER"),
        )
        calculator.definition.universe_identifiers = ["AAPL"]
        assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 1
        assert assets[0].name == "AAPL"  # defaults to identifier
        assert assets[0].currency == "EUR"
        assert assets[0].exchange == "UNKNOWN"

    def test_passes_date_to_fetcher(self, calculator, mock_data):
        mock_data.fetch_reference_data.return_value = _make_ref_df("Apple", "USD", "NASDAQ")
        calculator.definition.universe_identifiers = ["AAPL"]
        calculator._get_universe(pd.Timestamp("2025-06-15"))
        mock_data.fetch_reference_data.assert_called_once_with("AAPL", "2025-06-15")
