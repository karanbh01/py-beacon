# tests/test_index_calculator.py
"""Unit tests for IndexCalculator._get_universe() and IndexCalculator.run()."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from beacon.asset.bond import Bond
from beacon.asset.equity import Equity
from beacon.exceptions import CalculationError, UnexpectedCalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.result import IndexResult


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
    # Set explicitly: a bare MagicMock attribute is truthy, so capping would
    # try to compare weights against a mock instead of taking the uncapped
    # path these tests intend.
    defn.max_constituent_weight = None
    # Same reasoning for the fields added since. A mock is not a null, so a
    # comparison against one raises rather than taking the default path — the
    # cost of mocking the definition rather than building one.
    defn.calendar = None
    defn.rebalance_day_rule = "FIRST_BUSINESS_DAY"
    defn.return_type = "PRICE"
    defn.withholding_tax_rate = 0.0
    defn.effective_lag_sessions = 0
    # Same again, and now load-bearing: a truthy mock here makes the
    # market-value path float-adjusted, and BN-184 refuses a float-adjusted
    # index whose free-float factor is a mock rather than a number. Real
    # schemes default this to False.
    defn.weighting_scheme.use_free_float = False
    return defn


@pytest.fixture
def mock_data():
    return MagicMock()


@pytest.fixture
def calculator(mock_definition,
               mock_data):
    return IndexCalculator(mock_definition, mock_data)


def _make_ref_df(rows):
    """A reference frame for ``{identifier: (name, currency, exchange)}``.

    Indexed by identifier and carrying every requested name at once, which is
    the shape the real fetcher returns for the batch read `_get_universe` makes
    (BN-192).
    """
    return pd.DataFrame(
        {"NAME": [name for name, _, _ in rows.values()],
         "CURRENCY": [currency for _, currency, _ in rows.values()],
         "EXCHANGE": [exchange for _, _, exchange in rows.values()]},
        index=pd.Index(list(rows), name="IDENTIFIER"),
    )


class TestGetUniverse:
    def test_resolves_all_identifiers(self,
                                      calculator,
                                      mock_data):
        mock_data.fetch_reference_data.return_value = _make_ref_df({
            "AAPL": ("Apple Inc", "USD", "NASDAQ"),
            "MSFT": ("Microsoft", "USD", "NASDAQ"),
            "GOOG": ("Alphabet", "USD", "NASDAQ"),
        })
        assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 3
        assert all(isinstance(a, Equity) for a in assets)
        assert assets[0].ticker == "AAPL"
        assert assets[0].name == "Apple Inc"
        assert assets[1].ticker == "MSFT"
        assert assets[2].ticker == "GOOG"

    def test_none_universe_is_refused(self,
                                      calculator,
                                      mock_data):
        """A definition with no universe is refused, not calculated over nothing.

        This used to return an empty universe, which runs the whole
        calculation and publishes an index over no names (BN-184).
        """
        calculator.definition.universe_identifiers = None

        with pytest.raises(CalculationError, match="no universe_identifiers"):
            calculator._get_universe(pd.Timestamp("2025-01-01"))

        mock_data.fetch_reference_data.assert_not_called()

    def test_the_refusal_reaches_run(self,
                                     calculator):
        """And it is not absorbed on the way out of `run` (BN-184)."""
        calculator.definition.universe_identifiers = None

        with pytest.raises(CalculationError, match="no universe_identifiers"):
            calculator.run(end_date="2025-01-03")

    def test_skips_unresolvable_identifiers(self,
                                            calculator,
                                            mock_data,
                                            caplog):
        # MSFT is simply absent from the batch, which is what the reference
        # data returns for a name it has never heard of.
        mock_data.fetch_reference_data.return_value = _make_ref_df({
            "AAPL": ("Apple Inc", "USD", "NASDAQ"),
            "GOOG": ("Alphabet", "USD", "NASDAQ"),
        })
        with caplog.at_level("WARNING"):
            assets = calculator._get_universe(pd.Timestamp("2025-01-01"))
        assert len(assets) == 2
        assert assets[0].ticker == "AAPL"
        assert assets[1].ticker == "GOOG"
        assert "No reference data for 'MSFT'" in caplog.text

    def test_a_failing_lookup_propagates(self,
                                         calculator,
                                         mock_data):
        """A lookup that fails is not a name the universe does not contain.

        This used to be caught and recorded as a skip, so a broken fetcher and
        an unknown identifier were spelled the same way — and any refusal the
        data layer raised would have been converted back into a smaller
        universe here (BN-184).
        """
        mock_data.fetch_reference_data.side_effect = ConnectionError(
            "connection error")

        with pytest.raises(ConnectionError, match="connection error"):
            calculator._get_universe(pd.Timestamp("2025-01-01"))

    def test_a_failing_lookup_reaches_run(self,
                                          calculator,
                                          mock_data):
        """And it is not absorbed between `_get_universe` and `run` (BN-184)."""
        mock_data.fetch_reference_data.side_effect = ConnectionError("no route")

        with pytest.raises(ConnectionError, match="no route"):
            calculator.run(end_date="2025-01-03")

    def test_uses_defaults_for_missing_columns(self,
                                               calculator,
                                               mock_data):
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

    def test_passes_date_to_fetcher(self,
                                    calculator,
                                    mock_data):
        mock_data.fetch_reference_data.return_value = _make_ref_df(
            {"AAPL": ("Apple", "USD", "NASDAQ")})
        calculator.definition.universe_identifiers = ["AAPL"]
        calculator._get_universe(pd.Timestamp("2025-06-15"))
        # One read for the whole universe, not one per name (BN-192).
        mock_data.fetch_reference_data.assert_called_once_with(["AAPL"],
                                                               "2025-06-15")


# ---------------------------------------------------------------------------
# Helper assets for run() tests
# ---------------------------------------------------------------------------
AAPL = Equity(name="Apple", currency="USD", ticker="AAPL", exchange="NASDAQ")
MSFT = Equity(name="Microsoft", currency="USD", ticker="MSFT", exchange="NASDAQ")


def _stub_calculator(mock_definition,
                     mock_data):
    """Create a calculator with internal methods patched for run() tests."""
    calc = IndexCalculator(mock_definition, mock_data)
    return calc


def _price_the_universe(mock_data,
                        price=100.0):
    """Give the mock fetcher a real price, so run() can value its holdings.

    These tests used to leave `fetch_market_data` a bare MagicMock, whose
    `.empty` is truthy — so every constituent came back unpriceable, took zero
    units, contributed nothing, and the run published a level series over an
    index holding nothing at all. That is exactly the chain BN-191 refuses, so
    the run now stops instead; pricing the universe is what these tests always
    meant, and the assertions below are load-bearing rather than vacuous once
    it is there.
    """
    mock_data.fetch_market_data.return_value = pd.DataFrame(
        {"CLOSE": [price]}, index=[pd.Timestamp("2025-01-02")])


class TestRun:
    """Tests for IndexCalculator.run()."""

    def test_end_date_required(self,
                               calculator):
        with pytest.raises(ValueError, match="end_date must be provided"):
            calculator.run()

    def test_end_date_before_base_date_raises(self,
                                              calculator):
        with pytest.raises(ValueError, match="precedes base_date"):
            calculator.run(end_date="2024-12-01")

    def test_returns_index_result(self,
                                  calculator,
                                  mock_data):
        """run() returns an IndexResult with data_fetcher bound."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL, MSFT]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL, MSFT]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 0.6, MSFT: 0.4}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 6000.0, MSFT: 4000.0}),
        ):
            result = calculator.run(end_date="2025-01-02")

        assert isinstance(result, IndexResult)
        assert result.index_id == "TEST_IDX"
        assert result._data_fetcher is calculator.data

    def test_base_date_level_equals_base_value(self,
                                               calculator,
                                               mock_data):
        """On base date the index level should equal base_value."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 5000.0}),
        ):
            result = calculator.run(end_date="2025-01-02")

        assert result.index_levels.iloc[0] == 1000.0

    def test_base_date_records_snapshots(self,
                                         calculator,
                                         mock_data):
        """Base date should create constituent and weight snapshots."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL, MSFT]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL, MSFT]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 0.6, MSFT: 0.4}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 6000.0, MSFT: 4000.0}),
        ):
            result = calculator.run(end_date="2025-01-02")

        base = pd.Timestamp("2025-01-02")
        assert base in result.constituent_snapshots
        assert set(result.constituent_snapshots[base]) == {"AAPL", "MSFT"}
        assert result.weight_snapshots[base] == {"AAPL": 0.6, "MSFT": 0.4}

    def test_regular_day_values_the_held_units(self,
                                               calculator,
                                               mock_data):
        """After the base date, an ordinary day revalues the index's holdings.

        The index holds units fixed at the last rebalance (BN-103), so a
        regular day is just those units marked at today's prices over the
        divisor — no reselection, no reweighting.
        """
        _price_the_universe(mock_data)

        # base_date = 2025-01-02, run to 2025-01-03 (two business days)
        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 5000.0}),
            patch.object(calculator, 'level_from_units',
                         return_value=1050.0) as mock_level,
        ):
            result = calculator.run(end_date="2025-01-03")

        assert mock_level.called
        assert result.index_levels[pd.Timestamp("2025-01-03")] == 1050.0

    def test_rebalance_date_reconstitutes(self,
                                          calculator,
                                          mock_data):
        """On a rebalance date, the universe is re-resolved and weights recalculated."""
        _price_the_universe(mock_data)

        # Make get_rebalance_dates return a date within our range
        rebal_date = pd.Timestamp("2025-01-06")  # Monday
        calculator.definition.get_rebalance_dates.return_value = [rebal_date]

        call_count = {'universe': 0}

        def fake_get_universe(date):
            call_count['universe'] += 1
            return [AAPL, MSFT]

        def fake_select(universe,
                        current_date):
            return universe

        def fake_weights(constituents,
                         current_date):
            return {a: 1.0 / len(constituents) for a in constituents}

        def fake_mv(weights_dict,
                    date):
            return dict.fromkeys(weights_dict, 5000.0)

        with (
            patch.object(calculator, '_get_universe', side_effect=fake_get_universe),
            patch.object(calculator, 'select_constituents', side_effect=fake_select),
            patch.object(calculator, 'calculate_constituent_weights',
                         side_effect=fake_weights),
            patch.object(calculator, '_get_constituent_market_values',
                         side_effect=fake_mv),
            patch.object(calculator, 'calculate_index_level',
                         return_value=(1000.0, 10.0)),
        ):
            result = calculator.run(end_date="2025-01-06")

        # _get_universe called twice: base date + rebalance date
        assert call_count['universe'] == 2
        # Rebalance date should have a snapshot
        assert rebal_date in result.constituent_snapshots
        assert rebal_date in result.weight_snapshots

    def test_start_date_clamped_to_base_date(self,
                                             calculator,
                                             mock_data):
        """If start_date < base_date, it's clamped to base_date."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 5000.0}),
        ):
            result = calculator.run(start_date="2024-01-01", end_date="2025-01-02")

        # Should still start from base_date
        assert result.index_levels.index[0] == pd.Timestamp("2025-01-02")

    def test_empty_range_returns_empty_result(self,
                                              calculator):
        """If no trading days in range, return empty IndexResult."""
        # base_date is 2025-01-02 (Thursday), request a weekend range after it
        calculator.definition.base_date = pd.Timestamp("2025-01-04")  # Saturday
        result = calculator.run(start_date="2025-01-04", end_date="2025-01-05")
        assert isinstance(result, IndexResult)
        assert result.index_levels.empty

    def test_idempotent_multiple_calls(self,
                                       calculator,
                                       mock_data):
        """Calling run() twice produces identical results (no side effects)."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 5000.0}),
        ):
            r1 = calculator.run(end_date="2025-01-02")
            r2 = calculator.run(end_date="2025-01-02")

        pd.testing.assert_series_equal(r1.index_levels, r2.index_levels)
        pd.testing.assert_series_equal(r1.divisor_history, r2.divisor_history)

    def test_divisor_history_populated(self,
                                       calculator,
                                       mock_data):
        """Every trading day should have a divisor entry."""
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 5000.0}),
            patch.object(calculator, 'calculate_index_level',
                         return_value=(1010.0, 5.0)),
        ):
            result = calculator.run(end_date="2025-01-03")

        assert len(result.divisor_history) == len(result.index_levels)
        assert all(d > 0 for d in result.divisor_history.values)

    def test_zero_market_value_base_date_is_refused(self,
                                                    calculator,
                                                    mock_data):
        """A base date worth nothing has no scale to anchor an index to.

        This used to fall back to a divisor of 1.0, which publishes a level
        series that is internally coherent and measures nothing. The refusal
        reaches the caller of `run`, not just `initialize_divisor` (BN-184).
        """
        _price_the_universe(mock_data)

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            patch.object(calculator, '_get_constituent_market_values',
                         return_value={AAPL: 0.0}),
            pytest.raises(CalculationError, match="no scale to anchor"),
        ):
            calculator.run(end_date="2025-01-02")


class TestAdjustDivisorForRebalance:
    """Tests for IndexCalculator.adjust_divisor_for_rebalance()."""

    def test_basic_adjustment(self):
        """new_divisor = old_divisor * (new_mv / old_mv)."""
        # old_divisor=10, old_mv=10000, new_mv=12000
        # expected = 10 * (12000 / 10000) = 12.0
        result = IndexCalculator.adjust_divisor_for_rebalance(10.0, 10000.0, 12000.0)
        assert result == pytest.approx(12.0)

    def test_unchanged_composition(self):
        """When market values are identical, divisor stays the same."""
        result = IndexCalculator.adjust_divisor_for_rebalance(5.0, 8000.0, 8000.0)
        assert result == pytest.approx(5.0)

    def test_level_continuity(self):
        """Index level before and after rebalance should match within tolerance."""
        old_divisor = 10.0
        old_mv = 10000.0
        new_mv = 12000.0

        level_before = old_mv / old_divisor  # 1000.0

        new_divisor = IndexCalculator.adjust_divisor_for_rebalance(
            old_divisor, old_mv, new_mv
        )
        level_after = new_mv / new_divisor  # should also be 1000.0

        assert level_before == pytest.approx(level_after)

    def test_manually_computed_values(self):
        """Verify against hand-calculated expected divisor."""
        # old_divisor=25.0, old_mv=50000, new_mv=60000
        # expected = 25 * (60000 / 50000) = 30.0
        result = IndexCalculator.adjust_divisor_for_rebalance(25.0, 50000.0, 60000.0)
        assert result == pytest.approx(30.0)

        # old_divisor=8.5, old_mv=17000, new_mv=8500
        # expected = 8.5 * (8500 / 17000) = 4.25
        result = IndexCalculator.adjust_divisor_for_rebalance(8.5, 17000.0, 8500.0)
        assert result == pytest.approx(4.25)

    def test_zero_old_divisor_raises(self):
        with pytest.raises(ValueError, match="old_divisor must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(0.0, 10000.0, 12000.0)

    def test_negative_old_divisor_raises(self):
        with pytest.raises(ValueError, match="old_divisor must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(-1.0, 10000.0, 12000.0)

    def test_zero_old_market_value_raises(self):
        with pytest.raises(ValueError, match="old_market_value must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(10.0, 0.0, 12000.0)

    def test_negative_old_market_value_raises(self):
        with pytest.raises(ValueError, match="old_market_value must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(10.0, -5000.0, 12000.0)

    def test_zero_new_market_value_raises(self):
        with pytest.raises(ValueError, match="new_market_value must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(10.0, 10000.0, 0.0)

    def test_negative_new_market_value_raises(self):
        with pytest.raises(ValueError, match="new_market_value must be positive"):
            IndexCalculator.adjust_divisor_for_rebalance(10.0, 10000.0, -3000.0)


class TestHandleCorporateAction:
    """Tests for IndexCalculator.handle_corporate_action()."""

    @pytest.fixture
    def ca_calculator(self,
                      mock_definition,
                      mock_data):
        """Calculator with weighting_scheme.use_free_float = False."""
        mock_definition.weighting_scheme = MagicMock()
        mock_definition.weighting_scheme.use_free_float = False
        return IndexCalculator(mock_definition, mock_data)

    def _make_action(self,
                     action_type="SPECIAL_DIVIDEND",
                     asset=None,
                     value=2.0,
                     ex_date="2025-03-01"):
        return {"type": action_type, "asset": asset, "value": value, "ex_date": ex_date}

    def test_special_dividend_adjusts_divisor(self,
                                              ca_calculator,
                                              mock_data):
        """Known special dividend scenario with hand-calculated expected divisor."""
        # AAPL pays $2/share special dividend, 1000 shares outstanding
        # reduction = 2 * 1000 = 2000 (same currency, no FF)
        # mv_before = 100000, mv_after = 98000
        # new_divisor = 10 * (98000 / 100000) = 9.8
        mock_data.fetch_shares_outstanding.return_value = 1000
        action = self._make_action(asset=AAPL, value=2.0)

        result = ca_calculator.handle_corporate_action(
            action, [AAPL, MSFT], 100000.0, 10.0
        )
        assert result == pytest.approx(9.8)

    def test_special_dividend_with_free_float(self,
                                              ca_calculator,
                                              mock_data):
        """Special dividend with free-float factor applied."""
        ca_calculator.definition.weighting_scheme.use_free_float = True
        mock_data.fetch_shares_outstanding.return_value = 1000
        mock_data.fetch_free_float_factor.return_value = 0.5

        # reduction = 2 * 1000 * 0.5 = 1000
        # mv_after = 100000 - 1000 = 99000
        # new_divisor = 10 * (99000 / 100000) = 9.9
        action = self._make_action(asset=AAPL, value=2.0)
        result = ca_calculator.handle_corporate_action(
            action, [AAPL], 100000.0, 10.0
        )
        assert result == pytest.approx(9.9)

    def test_special_dividend_with_fx(self,
                                      ca_calculator,
                                      mock_data):
        """Special dividend in foreign currency applies FX conversion."""
        gbp_asset = Equity(name="BP", currency="GBP", ticker="BP", exchange="LSE")
        mock_data.fetch_shares_outstanding.return_value = 500
        mock_data.fetch_fx_rates.return_value = pd.Series([1.25])  # GBP->USD

        # reduction = 4 * 500 * 1.25 = 2500
        # mv_after = 50000 - 2500 = 47500
        # new_divisor = 5.0 * (47500 / 50000) = 4.75
        action = self._make_action(asset=gbp_asset, value=4.0)
        result = ca_calculator.handle_corporate_action(
            action, [gbp_asset], 50000.0, 5.0
        )
        assert result == pytest.approx(4.75)

    def test_non_constituent_returns_unchanged(self,
                                               ca_calculator,
                                               mock_data):
        """If asset is not in constituents, divisor is unchanged."""
        action = self._make_action(asset=AAPL)
        result = ca_calculator.handle_corporate_action(
            action, [MSFT], 100000.0, 10.0  # AAPL not in [MSFT]
        )
        assert result == 10.0

    def test_missing_asset_is_refused(self,
                                      ca_calculator):
        """A malformed action is not an action with no effect (BN-184)."""
        action = self._make_action(asset=None)

        with pytest.raises(CalculationError, match="missing its asset"):
            ca_calculator.handle_corporate_action(action, [AAPL], 100000.0, 10.0)

    def test_missing_ex_date_is_refused(self,
                                        ca_calculator):
        """No ex_date means no day to apply the action on (BN-184)."""
        action = {"type": "SPECIAL_DIVIDEND", "asset": AAPL, "value": 2.0, "ex_date": None}

        with pytest.raises(CalculationError, match="no ex_date"):
            ca_calculator.handle_corporate_action(action, [AAPL], 100000.0, 10.0)

    @pytest.mark.parametrize("action_type",
                             ["RIGHTS_ISSUE", "SPIN_OFF", "STOCK_DIVIDEND", "MERGER"])
    def test_unimplemented_types_are_refused(self,
                                             ca_calculator,
                                             action_type):
        """Unimplemented and no-effect must not share an answer (BN-184).

        These four used to return the divisor unchanged, which is exactly what
        an action on a non-constituent returns — so "Beacon cannot adjust for
        this" was indistinguishable from "this does not move the index", and
        the level stayed wrong from the ex-date onward.
        """
        action = self._make_action(action_type=action_type, asset=AAPL)

        with pytest.raises(CalculationError, match="is not implemented"):
            ca_calculator.handle_corporate_action(action, [AAPL], 100000.0, 10.0)

    def test_a_no_effect_action_still_returns_unchanged(self,
                                                        ca_calculator):
        """The refusals above did not swallow the genuine no-op (BN-184).

        An action on a name the index does not hold moves none of its market
        value, so the divisor that preserves continuity is the one in force.
        That is the one case "unchanged" still means.
        """
        for action_type in ("RIGHTS_ISSUE", "SPIN_OFF", "BIZARRE_EVENT"):
            action = self._make_action(action_type=action_type, asset=AAPL)

            assert ca_calculator.handle_corporate_action(
                action, [MSFT], 100000.0, 10.0) == 10.0

    def test_unknown_action_type_is_refused(self,
                                            ca_calculator):
        """An unknown action is not a no-effect action (BN-184)."""
        action = self._make_action(action_type="BIZARRE_EVENT", asset=AAPL)

        with pytest.raises(CalculationError, match="is not recognised"):
            ca_calculator.handle_corporate_action(action, [AAPL], 100000.0, 10.0)

    def test_no_shares_is_refused(self,
                                  ca_calculator,
                                  mock_data):
        """An unsizeable dividend leaves the level overstated forever (BN-184)."""
        mock_data.fetch_shares_outstanding.return_value = 0
        action = self._make_action(asset=AAPL)

        with pytest.raises(CalculationError, match="no shares outstanding"):
            ca_calculator.handle_corporate_action(action, [AAPL], 100000.0, 10.0)

    def test_missing_fx_is_refused(self,
                                   ca_calculator,
                                   mock_data):
        """BN-188's gap, on the corporate-action path (BN-184)."""
        gbp_asset = Equity(name="BP", currency="GBP", ticker="BP", exchange="LSE")
        mock_data.fetch_shares_outstanding.return_value = 500
        mock_data.fetch_fx_rates.return_value = pd.Series(dtype=float)
        action = self._make_action(asset=gbp_asset, value=4.0)

        with pytest.raises(CalculationError, match="no GBP/USD rate"):
            ca_calculator.handle_corporate_action(action, [gbp_asset], 50000.0, 5.0)

    def test_non_positive_market_value_is_refused(self,
                                                  ca_calculator,
                                                  mock_data):
        """The continuity ratio is undefined, and was stepped around (BN-184)."""
        mock_data.fetch_shares_outstanding.return_value = 500
        action = self._make_action(asset=AAPL, value=2.0)

        with pytest.raises(CalculationError, match=r"is worth 0\.0 before"):
            ca_calculator.handle_corporate_action(action, [AAPL], 0.0, 10.0)

    def test_an_action_consuming_the_whole_index_is_refused(self,
                                                            ca_calculator,
                                                            mock_data):
        """Either the dividend or the market value is wrong (BN-184)."""
        mock_data.fetch_shares_outstanding.return_value = 1000
        action = self._make_action(asset=AAPL, value=50.0)  # reduction 50,000

        with pytest.raises(CalculationError, match="consumes the whole index"):
            ca_calculator.handle_corporate_action(action, [AAPL], 1000.0, 10.0)

    def test_a_negligible_reduction_still_returns_unchanged(self,
                                                            ca_calculator,
                                                            mock_data):
        """The one arithmetic no-op inside the dividend path survives (BN-184).

        `divisor * (mv - 0) / mv` is the divisor; returning it unchanged is the
        continuity adjustment, not a stand-in for one.
        """
        mock_data.fetch_shares_outstanding.return_value = 1000
        action = self._make_action(asset=AAPL, value=0.0)

        assert ca_calculator.handle_corporate_action(
            action, [AAPL], 100000.0, 10.0) == 10.0

    def test_level_continuity_after_special_dividend(self,
                                                     ca_calculator,
                                                     mock_data):
        """Index level before and after special dividend adjustment should match."""
        mock_data.fetch_shares_outstanding.return_value = 500
        action = self._make_action(asset=AAPL, value=10.0)

        old_divisor = 20.0
        mv_before = 200000.0
        level_before = mv_before / old_divisor  # 10000.0

        new_divisor = ca_calculator.handle_corporate_action(
            action, [AAPL], mv_before, old_divisor
        )
        # reduction = 10 * 500 = 5000 (same ccy, no FF)
        mv_after = mv_before - 5000.0
        level_after = mv_after / new_divisor

        assert level_before == pytest.approx(level_after)


class TestANonEquityIsRefusedInTheCalculation:
    """BN-185: the market-value and corporate-action paths refuse, not absorb.

    These three sites used to skip the constituent with a warning, value it at
    0.0, and return the divisor unchanged in silence. Each answer computed a
    coherent index over a subset of its own universe, which is why none of them
    ever looked wrong from downstream.
    """

    @pytest.fixture
    def bond(self):
        return Bond(name="Treasury 10Y", currency="USD", asset_id="GOVT10Y",
                    maturity_date="2035-01-01", issuer="US Treasury")

    @pytest.fixture
    def ca_calculator(self,
                      mock_definition,
                      mock_data):
        mock_definition.weighting_scheme = MagicMock()
        mock_definition.weighting_scheme.use_free_float = False
        return IndexCalculator(mock_definition, mock_data)

    def test_constituent_market_values_refuses_rather_than_skipping(self,
                                                                    calculator,
                                                                    bond):
        with pytest.raises(CalculationError) as raised:
            calculator._get_constituent_market_values(
                {bond: 1.0}, pd.Timestamp("2025-03-03"))

        message = str(raised.value)

        assert "GOVT10Y" in message, "the refusal does not name the asset"
        assert "Bond" in message, "the refusal does not name the actual type"

    def test_a_skipped_constituent_no_longer_shrinks_the_aggregate(self,
                                                                   calculator,
                                                                   bond,
                                                                   mock_data):
        """The old answer: AAPL valued, the bond dropped, the total still summed."""
        mock_data.fetch_market_data.return_value = pd.DataFrame(
            {"CLOSE": [100.0]}, index=[pd.Timestamp("2025-03-03")])
        mock_data.fetch_shares_outstanding.return_value = 1000

        with pytest.raises(CalculationError, match="GOVT10Y"):
            calculator._get_constituent_market_values(
                {AAPL: 0.5, bond: 0.5}, pd.Timestamp("2025-03-03"))

    def test_unit_value_refuses_rather_than_returning_zero(self,
                                                           calculator,
                                                           bond):
        with pytest.raises(CalculationError) as raised:
            calculator.asset_unit_value(bond, pd.Timestamp("2025-03-03"))

        assert "GOVT10Y" in str(raised.value)

    def test_a_missing_price_is_still_worth_zero(self,
                                                 calculator,
                                                 bond,
                                                 mock_data):
        """The two zeroes were spelled the same; only one of them was right.

        A priceless equity is worth nothing *today* and the index carries the
        previous level forward. A bond is not worth nothing — it is not a thing
        this pipeline can value at all, and the refusal is raised before the
        broad `except` below so the handler cannot turn it back into 0.0.

        Since BN-191 the priceless equity comes back None rather than 0.0 —
        "could not be priced", not "priced at zero" — and the tolerance lives
        with the caller that is entitled to it: `holding_values` still values
        it at zero for the day, which is what makes the carry-forward work.
        """
        date = pd.Timestamp("2025-03-03")
        mock_data.fetch_market_data.return_value = pd.DataFrame()

        assert calculator.asset_unit_value(AAPL, date) is None
        assert calculator.holding_values({AAPL: 10.0}, date) == {AAPL: 0.0}

        with pytest.raises(CalculationError):
            calculator.asset_unit_value(bond, date)

    def test_a_corporate_action_refuses_rather_than_passing_the_divisor_through(
            self,
            ca_calculator,
            bond):
        """Unchanged was also the no-op answer, so the two were indistinguishable."""
        action = {"type": "SPECIAL_DIVIDEND", "asset": bond, "value": 2.0,
                  "ex_date": "2025-03-01"}

        with pytest.raises(CalculationError) as raised:
            ca_calculator.handle_corporate_action(action, [bond], 100000.0, 10.0)

        assert "GOVT10Y" in str(raised.value)

    def test_a_non_constituent_still_returns_unchanged(self,
                                                       ca_calculator,
                                                       bond):
        """The gate sits behind the constituency check, where it belongs.

        An action on a name the index does not hold cannot move its divisor
        whatever the name is, so that answer is about the index rather than
        about the asset type — and it is still the right one.
        """
        action = {"type": "SPECIAL_DIVIDEND", "asset": bond, "value": 2.0,
                  "ex_date": "2025-03-01"}

        assert ca_calculator.handle_corporate_action(
            action, [AAPL], 100000.0, 10.0) == 10.0


class TestASubstitutedWeightingIsRefused:
    """BN-184: the calculator applies the scheme it was given, or says so.

    Two sites in the catalogue of #197 rescaled or back-filled a methodology's
    own output and reported it in a log. Each produced an index that is
    internally consistent, fully plausible, and not the one that was
    specified — the failure #192 demonstrated, one layer up.
    """

    def test_weights_not_summing_to_one_are_refused(self,
                                                    calculator,
                                                    mock_definition):
        """Rescaling a scheme's output is the scheme not being applied.

        The old answer renormalised in silence, so a scheme with a bug — or
        one whose data went missing for half its names — published a different
        allocation under its own name.
        """
        mock_definition.weighting_scheme.scheme_name = "Lopsided"
        mock_definition.weighting_scheme.calculate_weights.return_value = {
            AAPL: 0.3, MSFT: 0.3}

        with pytest.raises(CalculationError, match="sum to"):
            calculator.calculate_constituent_weights(
                [AAPL, MSFT], pd.Timestamp("2025-03-03"))

    def test_the_refusal_is_not_rewrapped_as_a_scheme_failure(self,
                                                              calculator,
                                                              mock_definition):
        """It is raised outside the `except Exception` around the scheme call.

        That handler converts anything the scheme raises into a
        `WeightingScheme-…` CalculationError. Renormalising inside it would
        have reported the calculator's own refusal as the scheme's failure.
        """
        mock_definition.weighting_scheme.scheme_name = "Lopsided"
        mock_definition.weighting_scheme.calculate_weights.return_value = {AAPL: 0.5}

        with pytest.raises(CalculationError) as raised:
            calculator.calculate_constituent_weights([AAPL],
                                                     pd.Timestamp("2025-03-03"))

        assert "would publish an allocation the scheme did not produce" in str(raised.value)
        # And it stays a refusal, not a fault: BN-194 gave the wrapped case its
        # own class, so this assertion is now what keeps the two apart.
        assert not isinstance(raised.value, UnexpectedCalculationError)

    def test_weights_already_summing_to_one_pass_through(self,
                                                         calculator,
                                                         mock_definition):
        """The refusal did not swallow the ordinary case."""
        mock_definition.weighting_scheme.calculate_weights.return_value = {
            AAPL: 0.5, MSFT: 0.5}

        weights = calculator.calculate_constituent_weights(
            [AAPL, MSFT], pd.Timestamp("2025-03-03"))

        assert weights == {AAPL: 0.5, MSFT: 0.5}

    def test_a_missing_free_float_is_refused(self,
                                             calculator,
                                             mock_definition,
                                             mock_data):
        """The twin of the fallback BN-179 removed from the weighting.

        A float-adjusted scheme falling back to the full market cap weights the
        name as though every share were freely traded — a different index, not
        a rounding error.
        """
        mock_definition.weighting_scheme.use_free_float = True
        mock_definition.weighting_scheme.scheme_name = "FloatAdjustedMarketCap"
        mock_data.fetch_market_data.return_value = pd.DataFrame(
            {"CLOSE": [100.0]}, index=[pd.Timestamp("2025-03-03")])
        mock_data.fetch_shares_outstanding.return_value = 1000
        mock_data.fetch_free_float_factor.return_value = None

        with pytest.raises(CalculationError, match="no usable free-float factor"):
            calculator._get_constituent_market_values(
                {AAPL: 1.0}, pd.Timestamp("2025-03-03"))

    def test_the_free_float_refusal_reaches_run(self,
                                                calculator,
                                                mock_definition,
                                                mock_data):
        """And is not absorbed by the handlers between it and `run` (BN-184).

        `_asset_market_value` used to sit under a bare `except Exception` that
        logged and returned 0.0, which would have converted this refusal into a
        constituent worth nothing and carried on.
        """
        mock_definition.weighting_scheme.use_free_float = True
        mock_definition.weighting_scheme.scheme_name = "FloatAdjustedMarketCap"
        mock_data.fetch_free_float_factor.return_value = None
        mock_data.fetch_market_data.return_value = pd.DataFrame(
            {"CLOSE": [100.0]}, index=[pd.Timestamp("2025-01-02")])
        mock_data.fetch_shares_outstanding.return_value = 1000

        with (
            patch.object(calculator, '_get_universe', return_value=[AAPL]),
            patch.object(calculator, 'select_constituents', return_value=[AAPL]),
            patch.object(calculator, 'calculate_constituent_weights',
                         return_value={AAPL: 1.0}),
            pytest.raises(CalculationError, match="no usable free-float factor"),
        ):
            calculator.run(end_date="2025-01-03")


class TestASchemeThatCrashesIsNotReportedAsADecision:
    """BN-194: the `except Exception` around the scheme call wraps a *fault*.

    Every refusal in this file is deliberate and carries a remedy in its
    message. Whatever a scheme raises is neither, and both arrived under one
    published code — so a reader was told the engine had refused when in fact
    something broke, and went looking for a request to change that does not
    exist.
    """

    def test_a_crashing_scheme_raises_the_unexpected_subclass(self,
                                                              calculator,
                                                              mock_definition):
        mock_definition.weighting_scheme.scheme_name = "EqualWeighted"
        mock_definition.weighting_scheme.calculate_weights.side_effect = (
            ZeroDivisionError("division by zero"))

        with pytest.raises(UnexpectedCalculationError) as raised:
            calculator.calculate_constituent_weights([AAPL],
                                                     pd.Timestamp("2025-03-03"))

        assert raised.value.original_type == "ZeroDivisionError"
        assert isinstance(raised.value.__cause__, ZeroDivisionError)

    def test_it_is_still_a_calculation_error(self,
                                             calculator,
                                             mock_definition):
        """Existing `except CalculationError` callers keep catching it."""
        mock_definition.weighting_scheme.scheme_name = "EqualWeighted"
        mock_definition.weighting_scheme.calculate_weights.side_effect = (
            KeyError("CLOSE"))

        with pytest.raises(CalculationError) as raised:
            calculator.calculate_constituent_weights([AAPL],
                                                     pd.Timestamp("2025-03-03"))

        assert isinstance(raised.value, UnexpectedCalculationError)
        assert raised.value.original_type == "KeyError"

    def test_the_scheme_name_is_still_carried(self,
                                              calculator,
                                              mock_definition):
        """The `WeightingScheme-` prefix survives as a name, and only that.

        Nothing branches on it: the class, and the code it maps to, are what
        separate a crash from a refusal.
        """
        mock_definition.weighting_scheme.scheme_name = "EqualWeighted"
        mock_definition.weighting_scheme.calculate_weights.side_effect = (
            ZeroDivisionError("division by zero"))

        with pytest.raises(UnexpectedCalculationError) as raised:
            calculator.calculate_constituent_weights([AAPL],
                                                     pd.Timestamp("2025-03-03"))

        assert raised.value.calculation_name == "WeightingScheme-EqualWeighted"
        assert raised.value.details == "division by zero"
