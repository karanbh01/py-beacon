# tests/test_index_calculator.py
"""Unit tests for IndexCalculator._get_universe() and IndexCalculator.run()."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

from beacon.index.calculation import IndexCalculator
from beacon.index.result import IndexResult
from beacon.asset.equity import Equity


@pytest.fixture
def mock_definition():
    defn = MagicMock()
    defn.index_name = "Test Index"
    defn.index_id = "TEST_IDX"
    defn.currency = "USD"
    defn.base_value = 1000.0
    defn.base_date = pd.Timestamp("2025-01-02")
    defn.universe_identifiers = ["AAPL", "MSFT", "GOOG"]
    defn.rebalancing_frequency = "MONTHLY"
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


# ---------------------------------------------------------------------------
# Helper assets for run() tests
# ---------------------------------------------------------------------------
AAPL = Equity(name="Apple", currency="USD", ticker="AAPL", exchange="NASDAQ")
MSFT = Equity(name="Microsoft", currency="USD", ticker="MSFT", exchange="NASDAQ")


def _stub_calculator(mock_definition, mock_data):
    """Create a calculator with internal methods patched for run() tests."""
    calc = IndexCalculator(mock_definition, mock_data)
    return calc


class TestRun:
    """Tests for IndexCalculator.run()."""

    def test_end_date_required(self, calculator):
        with pytest.raises(ValueError, match="end_date must be provided"):
            calculator.run()

    def test_end_date_before_base_date_raises(self, calculator):
        with pytest.raises(ValueError, match="precedes base_date"):
            calculator.run(end_date="2024-12-01")

    def test_returns_index_result(self, calculator):
        """run() returns an IndexResult with data_fetcher bound."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL, MSFT]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL, MSFT]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 0.6, MSFT: 0.4}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 6000.0, MSFT: 4000.0}):
                        result = calculator.run(end_date="2025-01-02")

        assert isinstance(result, IndexResult)
        assert result.index_id == "TEST_IDX"
        assert result._data_fetcher is calculator.data

    def test_base_date_level_equals_base_value(self, calculator):
        """On base date the index level should equal base_value."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 5000.0}):
                        result = calculator.run(end_date="2025-01-02")

        assert result.index_levels.iloc[0] == 1000.0

    def test_base_date_records_snapshots(self, calculator):
        """Base date should create constituent and weight snapshots."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL, MSFT]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL, MSFT]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 0.6, MSFT: 0.4}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 6000.0, MSFT: 4000.0}):
                        result = calculator.run(end_date="2025-01-02")

        base = pd.Timestamp("2025-01-02")
        assert base in result.constituent_snapshots
        assert set(result.constituent_snapshots[base]) == {"AAPL", "MSFT"}
        assert result.weight_snapshots[base] == {"AAPL": 0.6, "MSFT": 0.4}

    def test_regular_day_uses_calculate_index_level(self, calculator):
        """After base date, regular days call calculate_index_level."""
        # base_date = 2025-01-02, run to 2025-01-03 (two business days)
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 5000.0}):
                        with patch.object(calculator, 'calculate_index_level',
                                          return_value=(1050.0, 5.0)) as mock_calc:
                            result = calculator.run(end_date="2025-01-03")

        # calculate_index_level should be called for Jan 3 (regular day)
        assert mock_calc.called
        # Jan 3 level should be what calculate_index_level returned
        assert result.index_levels[pd.Timestamp("2025-01-03")] == 1050.0

    def test_rebalance_date_reconstitutes(self, calculator):
        """On a rebalance date, the universe is re-resolved and weights recalculated."""
        base = pd.Timestamp("2025-01-02")
        # Make get_rebalance_dates return a date within our range
        rebal_date = pd.Timestamp("2025-01-06")  # Monday
        calculator.definition.get_rebalance_dates.return_value = [rebal_date]

        call_count = {'universe': 0}

        def fake_get_universe(date):
            call_count['universe'] += 1
            return [AAPL, MSFT]

        def fake_select(universe, current_date):
            return universe

        def fake_weights(constituents, current_date):
            return {a: 1.0 / len(constituents) for a in constituents}

        def fake_mv(weights_dict, date):
            return {a: 5000.0 for a in weights_dict}

        with patch.object(calculator, '_get_universe', side_effect=fake_get_universe):
            with patch.object(calculator, 'select_constituents', side_effect=fake_select):
                with patch.object(calculator, 'calculate_constituent_weights', side_effect=fake_weights):
                    with patch.object(calculator, '_get_constituent_market_values', side_effect=fake_mv):
                        with patch.object(calculator, 'calculate_index_level',
                                          return_value=(1000.0, 10.0)):
                            result = calculator.run(end_date="2025-01-06")

        # _get_universe called twice: base date + rebalance date
        assert call_count['universe'] == 2
        # Rebalance date should have a snapshot
        assert rebal_date in result.constituent_snapshots
        assert rebal_date in result.weight_snapshots

    def test_start_date_clamped_to_base_date(self, calculator):
        """If start_date < base_date, it's clamped to base_date."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 5000.0}):
                        result = calculator.run(start_date="2024-01-01", end_date="2025-01-02")

        # Should still start from base_date
        assert result.index_levels.index[0] == pd.Timestamp("2025-01-02")

    def test_empty_range_returns_empty_result(self, calculator):
        """If no trading days in range, return empty IndexResult."""
        # base_date is 2025-01-02 (Thursday), request a weekend range after it
        calculator.definition.base_date = pd.Timestamp("2025-01-04")  # Saturday
        result = calculator.run(start_date="2025-01-04", end_date="2025-01-05")
        assert isinstance(result, IndexResult)
        assert result.index_levels.empty

    def test_idempotent_multiple_calls(self, calculator):
        """Calling run() twice produces identical results (no side effects)."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 5000.0}):
                        r1 = calculator.run(end_date="2025-01-02")
                        r2 = calculator.run(end_date="2025-01-02")

        pd.testing.assert_series_equal(r1.index_levels, r2.index_levels)
        pd.testing.assert_series_equal(r1.divisor_history, r2.divisor_history)

    def test_divisor_history_populated(self, calculator):
        """Every trading day should have a divisor entry."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 5000.0}):
                        with patch.object(calculator, 'calculate_index_level',
                                          return_value=(1010.0, 5.0)):
                            result = calculator.run(end_date="2025-01-03")

        assert len(result.divisor_history) == len(result.index_levels)
        assert all(d > 0 for d in result.divisor_history.values)

    def test_zero_market_value_base_date(self, calculator):
        """When base date MV is zero, divisor defaults to 1.0."""
        with patch.object(calculator, '_get_universe', return_value=[AAPL]):
            with patch.object(calculator, 'select_constituents', return_value=[AAPL]):
                with patch.object(calculator, 'calculate_constituent_weights',
                                  return_value={AAPL: 1.0}):
                    with patch.object(calculator, '_get_constituent_market_values',
                                      return_value={AAPL: 0.0}):
                        result = calculator.run(end_date="2025-01-02")

        assert result.divisor_history.iloc[0] == 1.0
