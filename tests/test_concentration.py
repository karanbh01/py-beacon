# tests/test_concentration.py
"""Unit tests for concentration and drift analytics. No HTTP involved."""
import pytest

from beacon.analysis import (
    concentration,
    drift_from_target,
    drift_history,
    effective_number_of_assets,
    herfindahl_index,
    top_n_weight,
)
from beacon.exceptions import CalculationError

EQUAL_FOUR = dict.fromkeys(["AAA", "BBB", "CCC", "DDD"], 0.25)
CONCENTRATED = {"AAA": 0.7, "BBB": 0.1, "CCC": 0.1, "DDD": 0.1}


class TestHerfindahlIndex:

    def test_equal_weights_give_one_over_n(self):
        """Four equal weights: 4 x 0.25^2 = 0.25."""
        assert herfindahl_index(EQUAL_FOUR) == pytest.approx(0.25)

    def test_single_position_gives_one(self):
        assert herfindahl_index({"AAA": 1.0}) == pytest.approx(1.0)

    def test_hand_checked_concentrated_case(self):
        """0.7^2 + 3 x 0.1^2 = 0.49 + 0.03 = 0.52."""
        assert herfindahl_index(CONCENTRATED) == pytest.approx(0.52)

    def test_concentration_raises_the_index(self):
        assert herfindahl_index(CONCENTRATED) > herfindahl_index(EQUAL_FOUR)

    def test_empty_weights_give_zero(self):
        assert herfindahl_index({}) == 0.0

    def test_bounded_between_one_over_n_and_one(self):
        for weights in (EQUAL_FOUR, CONCENTRATED, {"AAA": 0.5, "BBB": 0.5}):
            index = herfindahl_index(weights)
            assert 1.0 / len(weights) - 1e-12 <= index <= 1.0 + 1e-12

    def test_short_position_concentrates_like_a_long_one(self):
        """Squaring is blind to sign; the docstring says so."""
        assert herfindahl_index({"AAA": -0.5, "BBB": 0.5}) == pytest.approx(0.5)

    def test_partial_investment_warns(self,
                                      caplog):
        """Not an error, but it rescales the measure."""
        with caplog.at_level("WARNING"):
            herfindahl_index({"AAA": 0.3, "BBB": 0.3})

        assert "not 1.0" in caplog.text


class TestEffectiveNumberOfAssets:

    def test_equal_weights_give_the_asset_count(self):
        """The acceptance criterion."""
        assert effective_number_of_assets(EQUAL_FOUR) == pytest.approx(4.0)

    @pytest.mark.parametrize("count", [1, 2, 5, 10, 100])
    def test_equal_weights_of_any_size_give_that_size(self,
                                                      count):
        weights = {f"A{i}": 1.0 / count for i in range(count)}

        assert effective_number_of_assets(weights) == pytest.approx(float(count))

    def test_hand_checked_concentrated_case(self):
        """1 / 0.52 = 1.923..."""
        assert effective_number_of_assets(CONCENTRATED) == pytest.approx(1.0 / 0.52)

    def test_is_never_more_than_the_asset_count(self):
        assert effective_number_of_assets(CONCENTRATED) <= len(CONCENTRATED)

    def test_concentration_reduces_it(self):
        assert (effective_number_of_assets(CONCENTRATED)
                < effective_number_of_assets(EQUAL_FOUR))

    def test_empty_weights_give_zero(self):
        assert effective_number_of_assets({}) == 0.0

    def test_all_zero_weights_give_zero_rather_than_dividing(self):
        assert effective_number_of_assets({"AAA": 0.0, "BBB": 0.0}) == 0.0


class TestConcentrationSummary:

    def test_reports_the_largest_position(self):
        metrics = concentration(CONCENTRATED)

        assert metrics.largest_asset_id == "AAA"
        assert metrics.largest_weight == pytest.approx(0.7)

    def test_reports_the_asset_count(self):
        assert concentration(EQUAL_FOUR).assets == 4

    def test_agrees_with_the_standalone_functions(self):
        metrics = concentration(CONCENTRATED)

        assert metrics.herfindahl_index == pytest.approx(herfindahl_index(CONCENTRATED))
        assert metrics.effective_assets == pytest.approx(
            effective_number_of_assets(CONCENTRATED))

    def test_empty_weights_are_a_legitimate_state(self):
        """An index between inception and base date holds nothing."""
        metrics = concentration({})

        assert metrics.assets == 0
        assert metrics.largest_asset_id is None
        assert metrics.effective_assets == 0.0


class TestTopNWeight:

    def test_hand_checked_top_two(self):
        """0.7 + 0.1 = 0.8."""
        assert top_n_weight(CONCENTRATED, 2) == pytest.approx(0.8)

    def test_top_one_is_the_largest_weight(self):
        assert top_n_weight(CONCENTRATED, 1) == pytest.approx(0.7)

    def test_asking_for_more_than_held_sums_everything(self):
        assert top_n_weight(EQUAL_FOUR, 99) == pytest.approx(1.0)

    def test_is_monotonic_in_n(self):
        values = [top_n_weight(CONCENTRATED, n) for n in (1, 2, 3, 4)]

        assert values == sorted(values)

    def test_rejects_a_non_positive_count(self):
        with pytest.raises(CalculationError, match="must be positive"):
            top_n_weight(EQUAL_FOUR, 0)

    def test_empty_weights_give_zero(self):
        assert top_n_weight({}, 3) == 0.0


class TestDriftFromTarget:

    def test_matching_weights_have_no_drift(self):
        metrics = drift_from_target(EQUAL_FOUR, EQUAL_FOUR)

        assert metrics.max_absolute == pytest.approx(0.0)
        assert metrics.total_absolute == pytest.approx(0.0)
        assert metrics.turnover == pytest.approx(0.0)

    def test_hand_checked_drift(self):
        current = {"AAA": 0.30, "BBB": 0.20, "CCC": 0.25, "DDD": 0.25}

        metrics = drift_from_target(current, EQUAL_FOUR)

        assert metrics.per_asset["AAA"] == pytest.approx(0.05)
        assert metrics.per_asset["BBB"] == pytest.approx(-0.05)
        assert metrics.per_asset["CCC"] == pytest.approx(0.0)
        assert metrics.max_absolute == pytest.approx(0.05)
        assert metrics.total_absolute == pytest.approx(0.10)
        assert metrics.turnover == pytest.approx(0.05)

    def test_turnover_is_half_the_total_drift(self):
        """Every overweight funds an underweight, so a round trip is halved."""
        current = {"AAA": 0.40, "BBB": 0.10, "CCC": 0.25, "DDD": 0.25}

        metrics = drift_from_target(current, EQUAL_FOUR)

        assert metrics.turnover == pytest.approx(metrics.total_absolute / 2.0)

    def test_names_the_worst_drifter(self):
        current = {"AAA": 0.40, "BBB": 0.20, "CCC": 0.25, "DDD": 0.15}

        assert drift_from_target(current, EQUAL_FOUR).max_absolute_asset_id == "AAA"

    def test_ties_are_broken_deterministically(self):
        """An equal over- and underweight differ only in the last float bit.

        A plain max would let that noise decide, so the answer could differ
        between platforms. Ties resolve to the first asset in sort order.
        """
        current = {"AAA": 0.35, "BBB": 0.15, "CCC": 0.25, "DDD": 0.25}

        metrics = drift_from_target(current, EQUAL_FOUR)

        assert metrics.max_absolute_asset_id == "AAA"
        assert abs(metrics.per_asset["AAA"]) == pytest.approx(
            abs(metrics.per_asset["BBB"]))

    def test_tie_break_is_stable_under_key_order(self):
        """Insertion order must not change the answer either."""
        forwards = {"AAA": 0.35, "BBB": 0.15, "CCC": 0.25, "DDD": 0.25}
        backwards = {"DDD": 0.25, "CCC": 0.25, "BBB": 0.15, "AAA": 0.35}

        assert (drift_from_target(forwards, EQUAL_FOUR).max_absolute_asset_id
                == drift_from_target(backwards, EQUAL_FOUR).max_absolute_asset_id)

    def test_a_sold_position_shows_as_drift(self):
        """Absence is a zero weight, not an omission."""
        current = {"AAA": 0.5, "BBB": 0.5}

        metrics = drift_from_target(current, EQUAL_FOUR)

        assert metrics.per_asset["CCC"] == pytest.approx(-0.25)
        assert metrics.per_asset["DDD"] == pytest.approx(-0.25)

    def test_an_unwanted_holding_shows_as_drift(self):
        current = {**EQUAL_FOUR, "EEE": 0.1}

        metrics = drift_from_target(current, EQUAL_FOUR)

        assert metrics.per_asset["EEE"] == pytest.approx(0.1)

    def test_covers_every_asset_in_either_set(self):
        metrics = drift_from_target({"AAA": 1.0}, {"BBB": 1.0})

        assert set(metrics.per_asset) == {"AAA", "BBB"}

    def test_drift_signs_are_directional(self):
        metrics = drift_from_target({"AAA": 0.6, "BBB": 0.4},
                                    {"AAA": 0.5, "BBB": 0.5})

        assert metrics.per_asset["AAA"] > 0    # overweight
        assert metrics.per_asset["BBB"] < 0    # underweight

    def test_empty_inputs_are_handled(self):
        metrics = drift_from_target({}, {})

        assert metrics.per_asset == {}
        assert metrics.max_absolute_asset_id is None


class TestDriftHistory:

    def test_computes_drift_at_each_snapshot(self):
        history = {
            "2024-01-02": EQUAL_FOUR,
            "2024-02-01": {"AAA": 0.30, "BBB": 0.20, "CCC": 0.25, "DDD": 0.25},
        }

        drifts = drift_history(history, EQUAL_FOUR)

        assert drifts["2024-01-02"].max_absolute == pytest.approx(0.0)
        assert drifts["2024-02-01"].max_absolute == pytest.approx(0.05)

    def test_snapshots_come_back_in_label_order(self):
        history = {"2024-03-01": EQUAL_FOUR,
                   "2024-01-02": EQUAL_FOUR,
                   "2024-02-01": EQUAL_FOUR}

        assert list(drift_history(history, EQUAL_FOUR)) == [
            "2024-01-02", "2024-02-01", "2024-03-01"]

    def test_empty_history_gives_an_empty_result(self):
        assert drift_history({}, EQUAL_FOUR) == {}
