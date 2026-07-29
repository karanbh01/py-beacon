# tests/test_capping.py
"""Unit tests for weight capping."""
import math
import re

import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.index.capping import (
    MAX_PASSES,
    TOLERANCE,
    apply_cap,
    minimum_feasible_cap,
)
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted, MarketCapWeighted


def total(weights: dict[str, float]) -> float:
    """Sum of weights."""
    return sum(weights.values())


class TestNoOp:

    def test_none_cap_returns_input_unchanged(self):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        capped, report = apply_cap(weights, None)

        assert capped == weights
        assert report.was_capped is False

    def test_empty_weights(self):
        capped, report = apply_cap({}, 0.1)

        assert capped == {}
        assert report.was_capped is False

    def test_cap_of_one_is_a_no_op(self):
        weights = {"A": 0.9, "B": 0.1}

        capped, report = apply_cap(weights, 1.0)

        assert capped == weights
        assert report.passes == 0

    def test_non_binding_cap_leaves_weights_alone(self):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        capped, report = apply_cap(weights, 0.6)

        assert capped == weights
        assert report.was_capped is False

    def test_weight_exactly_at_the_cap_does_not_bind(self):
        weights = {"A": 0.5, "B": 0.5}

        capped, report = apply_cap(weights, 0.5)

        assert capped == weights
        assert report.was_capped is False
        assert report.passes == 0


class TestSinglePass:

    def test_excess_is_redistributed_pro_rata(self):
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        capped, _ = apply_cap(weights, 0.4)

        assert capped["A"] == pytest.approx(0.4)
        # 0.1 spread over B and C in a 3:2 ratio.
        assert capped["B"] == pytest.approx(0.36)
        assert capped["C"] == pytest.approx(0.24)
        assert total(capped) == pytest.approx(1.0)

    def test_report_names_the_capped_asset_and_its_original_weight(self):
        _, report = apply_cap({"A": 0.5, "B": 0.3, "C": 0.2}, 0.4)

        assert report.capped == {"A": 0.5}
        assert report.redistributed == pytest.approx(0.1)
        assert report.passes == 1
        assert report.cap == 0.4


class TestIteration:

    def test_a_second_name_can_breach_after_redistribution(self):
        """The point of iterating: capping A pushes B over the cap."""
        weights = {"A": 0.7, "B": 0.2, "C": 0.1}

        capped, report = apply_cap(weights, 0.34)

        assert sorted(report.capped) == ["A", "B"]
        assert report.passes == 2
        assert capped["A"] == pytest.approx(0.34)
        assert capped["B"] == pytest.approx(0.34)
        assert total(capped) == pytest.approx(1.0)

    def test_no_weight_exceeds_the_cap(self):
        weights = {"A": 0.6, "B": 0.25, "C": 0.1, "D": 0.05}

        capped, _ = apply_cap(weights, 0.3)

        assert all(weight <= 0.3 * (1 + TOLERANCE) for weight in capped.values())

    def test_exactly_feasible_cap_pins_every_name(self):
        weights = {"A": 0.7, "B": 0.2, "C": 0.1}

        capped, _ = apply_cap(weights, 1 / 3)

        assert all(weight == pytest.approx(1 / 3) for weight in capped.values())
        assert total(capped) == pytest.approx(1.0)


class TestFeasibility:

    def test_infeasible_cap_raises(self):
        with pytest.raises(CalculationError, match="cannot be satisfied"):
            apply_cap({"A": 0.5, "B": 0.5}, 0.4)

    def test_infeasible_message_names_the_minimum(self):
        with pytest.raises(CalculationError, match=re.escape("50.0000%")):
            apply_cap({"A": 0.6, "B": 0.4}, 0.3)

    def test_minimum_feasible_cap(self):
        assert minimum_feasible_cap(10) == pytest.approx(0.1)
        assert minimum_feasible_cap(1) == pytest.approx(1.0)

    def test_minimum_feasible_cap_rejects_zero(self):
        with pytest.raises(ValueError):
            minimum_feasible_cap(0)

    @pytest.mark.parametrize("cap", [0.0, -0.1, 1.5])
    def test_cap_outside_the_unit_interval_raises(self,
                                                  cap):
        with pytest.raises(ValueError, match=re.escape("cap must be in (0, 1]")):
            apply_cap({"A": 0.6, "B": 0.4}, cap)

    def test_single_constituent_below_full_weight_is_infeasible(self):
        with pytest.raises(CalculationError):
            apply_cap({"A": 1.0}, 0.5)

    def test_pass_bound_is_generous_enough_to_never_bind(self):
        """The loop shrinks the uncapped set each pass, so n passes suffice."""
        weights = {f"A{i}": 1 / 50 for i in range(50)}
        weights["A0"] = 0.5
        remaining = (1.0 - 0.5) / 49
        weights.update({f"A{i}": remaining for i in range(1, 50)})

        _, report = apply_cap(weights, 0.05)

        assert report.passes < MAX_PASSES


class TestComposesWithSchemes:
    """Capping is applied by the calculator, so it works with any scheme."""

    @pytest.mark.parametrize("scheme", [EqualWeighted(), MarketCapWeighted()])
    def test_definition_accepts_a_cap_for_any_scheme(self,
                                                     scheme):
        definition = IndexDefinition(index_id="X",
                                     index_name="X",
                                     base_date="2024-01-02",
                                     base_value=1000.0,
                                     currency="USD",
                                     eligibility_rules=[],
                                     weighting_scheme=scheme,
                                     rebalancing_frequency="MONTHLY",
                                     universe_identifiers=["A", "B", "C"],
                                     max_constituent_weight=0.4)

        assert definition.max_constituent_weight == 0.4

    @pytest.mark.parametrize("cap", [0.0, 1.5, -1.0])
    def test_definition_rejects_an_out_of_range_cap(self,
                                                    cap):
        with pytest.raises(ValueError, match="max_constituent_weight"):
            IndexDefinition(index_id="X",
                            index_name="X",
                            base_date="2024-01-02",
                            base_value=1000.0,
                            currency="USD",
                            eligibility_rules=[],
                            weighting_scheme=EqualWeighted(),
                            rebalancing_frequency="MONTHLY",
                            universe_identifiers=["A", "B", "C"],
                            max_constituent_weight=cap)

    def test_uncapped_definition_defaults_to_none(self):
        definition = IndexDefinition(index_id="X",
                                     index_name="X",
                                     base_date="2024-01-02",
                                     base_value=1000.0,
                                     currency="USD",
                                     eligibility_rules=[],
                                     weighting_scheme=EqualWeighted(),
                                     rebalancing_frequency="MONTHLY",
                                     universe_identifiers=["A", "B", "C"])

        assert definition.max_constituent_weight is None


class TestCalculatorIntegration:
    """cap_weights() is pure: it reports rather than storing state."""

    def _calculator(self,
                    cap):
        from unittest.mock import MagicMock

        from beacon.index.calculation import IndexCalculator

        definition = IndexDefinition(index_id="X",
                                     index_name="X",
                                     base_date="2024-01-02",
                                     base_value=1000.0,
                                     currency="USD",
                                     eligibility_rules=[],
                                     weighting_scheme=EqualWeighted(),
                                     rebalancing_frequency="MONTHLY",
                                     universe_identifiers=["A", "B", "C"],
                                     max_constituent_weight=cap)

        return IndexCalculator(definition, MagicMock())

    def _assets(self):
        from beacon.asset.equity import Equity

        return [Equity(name=n, currency="USD", ticker=n, exchange="NYSE")
                for n in ("A", "B", "C")]

    def test_cap_weights_applies_the_definition_cap(self):
        assets = self._assets()
        weights = dict(zip(assets, [0.5, 0.3, 0.2], strict=True))

        capped, report = self._calculator(0.4).cap_weights(weights)

        assert capped[assets[0]] == pytest.approx(0.4)
        assert report.capped == {"A": 0.5}

    def test_cap_weights_is_a_no_op_without_a_cap(self):
        assets = self._assets()
        weights = dict(zip(assets, [0.5, 0.3, 0.2], strict=True))

        capped, report = self._calculator(None).cap_weights(weights)

        assert capped == weights
        assert report.was_capped is False

    def test_calling_twice_gives_the_same_answer(self):
        """The calculator holds no capping state, so this must be idempotent."""
        assets = self._assets()
        weights = dict(zip(assets, [0.5, 0.3, 0.2], strict=True))
        calculator = self._calculator(0.4)

        first, _ = calculator.cap_weights(weights)
        second, _ = calculator.cap_weights(weights)

        assert first == second


class TestResultProvenance:

    def test_capped_assets_on_date_reads_the_report(self):
        from beacon.index.capping import CapReport
        from beacon.index.result import IndexResult

        date = pd.Timestamp("2024-01-02")
        result = IndexResult(index_id="X",
                             index_levels=pd.Series(dtype=float),
                             divisor_history=pd.Series(dtype=float),
                             constituent_snapshots={},
                             weight_snapshots={},
                             cap_reports={date: CapReport(cap=0.4,
                                                          capped={"A": 0.5},
                                                          redistributed=0.1,
                                                          passes=1)})

        assert result.capped_assets_on_date(date) == {"A": 0.5}
        assert result.capped_assets_on_date(pd.Timestamp("2024-02-01")) == {}

    def test_uncapped_result_has_no_reports(self):
        from beacon.index.result import IndexResult

        result = IndexResult(index_id="X",
                             index_levels=pd.Series(dtype=float),
                             divisor_history=pd.Series(dtype=float),
                             constituent_snapshots={},
                             weight_snapshots={})

        assert result.cap_reports == {}


class TestFloatingPointBehaviour:

    def test_sum_holds_across_many_constituents(self):
        weights = {f"A{i}": (i + 1) / 5050 for i in range(100)}

        capped, _ = apply_cap(weights, 0.015)

        assert math.isclose(total(capped), 1.0, rel_tol=1e-9)
        assert max(capped.values()) <= 0.015 * (1 + TOLERANCE)
