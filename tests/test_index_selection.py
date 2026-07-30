# tests/test_index_selection.py
"""BN-102: the single selection walk, and the provenance it records."""
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.index.calculation.selection import (
    UNIVERSE_POSITION,
    SelectionResult,
    SelectionStep,
    select_with_provenance,
)
from beacon.index.methodology import EligibilityRuleBase

DATE = pd.Timestamp("2024-06-03")
UNIVERSE_IDS = ["AAA", "BBB", "CCC", "DDD"]


class Allow(EligibilityRuleBase):
    """Admits only the identifiers it was given."""

    def __init__(self,
                 allowed: set[str],
                 name: str = "AllowRule"):
        super().__init__(rule_name=name)
        self.allowed = allowed

    def is_eligible(self,
                    asset,
                    current_date,
                    market_data_provider,
                    context=None) -> bool:
        return asset.asset_id in self.allowed


class Explode(EligibilityRuleBase):
    """Raises on a named asset, to exercise the error path."""

    def __init__(self,
                 victim: str):
        super().__init__(rule_name="ExplodeRule")
        self.victim = victim

    def is_eligible(self,
                    asset,
                    current_date,
                    market_data_provider,
                    context=None) -> bool:
        if asset.asset_id == self.victim:
            raise RuntimeError("rule blew up")

        return True


def universe() -> list[Equity]:
    """Four assets, in a fixed order."""
    return [Equity(name=f"{identifier} Corp", currency="USD",
                   ticker=identifier, exchange="NYSE")
            for identifier in UNIVERSE_IDS]


def run(rules) -> SelectionResult:
    """Select over the standard universe."""
    return select_with_provenance(universe(), rules, DATE, data_fetcher=None)


class TestSurvivors:

    def test_no_rules_keeps_everything(self):
        result = run([])

        assert result.survivor_ids == UNIVERSE_IDS

    def test_a_rule_removes_what_it_rejects(self):
        result = run([Allow({"AAA", "BBB"})])

        assert result.survivor_ids == ["AAA", "BBB"]

    def test_rules_compose(self):
        result = run([Allow({"AAA", "BBB", "CCC"}), Allow({"BBB", "CCC", "DDD"})])

        assert result.survivor_ids == ["BBB", "CCC"]

    def test_universe_order_is_preserved(self):
        """Weighting and reporting both read this order; it should not shuffle."""
        result = run([Allow({"DDD", "AAA"})])

        assert result.survivor_ids == ["AAA", "DDD"]

    def test_an_empty_universe_survives_nothing(self):
        result = select_with_provenance([], [Allow({"AAA"})], DATE, None)

        assert result.survivors == []

    def test_everything_can_be_excluded(self):
        result = run([Allow(set())])

        assert result.survivors == []
        assert len(result.exclusions) == len(UNIVERSE_IDS)


class TestFunnel:

    def test_the_first_step_is_the_universe(self):
        result = run([Allow({"AAA"})])

        assert result.steps[0].position == UNIVERSE_POSITION
        assert result.steps[0].is_universe
        assert result.steps[0].remaining == len(UNIVERSE_IDS)

    def test_there_is_one_step_per_rule_plus_the_universe(self):
        result = run([Allow({"AAA", "BBB"}), Allow({"AAA"})])

        assert len(result.steps) == 3
        assert len(result.rule_steps) == 2

    def test_each_step_reports_what_it_removed(self):
        result = run([Allow({"AAA", "BBB", "CCC"}), Allow({"AAA"})])

        assert result.steps[1].excluded == ["DDD"]
        assert result.steps[2].excluded == ["BBB", "CCC"]

    def test_each_step_reports_what_remains(self):
        result = run([Allow({"AAA", "BBB", "CCC"}), Allow({"AAA"})])

        assert [step.remaining for step in result.steps] == [4, 3, 1]

    def test_steps_carry_the_rule_type(self):
        result = run([Allow({"AAA"}, name="LiquidityRule")])

        assert result.steps[1].rule_name == "LiquidityRule"

    def test_exclusions_are_sorted(self):
        """Stable output, so a rendered waterfall does not reorder run to run."""
        result = run([Allow(set())])

        assert result.steps[1].excluded == sorted(UNIVERSE_IDS)

    def test_the_universe_step_names_no_rule(self):
        result = run([Allow({"AAA"})])

        assert result.steps[0].rule_name == ""
        assert result.steps[0].excluded == []


class TestProvenance:

    def test_every_excluded_asset_has_a_reason(self):
        result = run([Allow({"AAA", "BBB"}), Allow({"AAA"})])

        assert set(result.exclusions) == {"CCC", "DDD", "BBB"}

    def test_survivors_have_no_reason(self):
        result = run([Allow({"AAA", "BBB"})])

        assert "AAA" not in result.exclusions
        assert result.excluded_by("AAA") is None

    def test_the_first_rule_to_exclude_owns_the_asset(self):
        """The single-owner property.

        An asset leaves the surviving set the moment it fails, so no later rule
        ever sees it and no name can be blamed on two rules. Without that the
        funnel could only say how many are left, not why any one is missing.
        """
        strict_then_stricter = [Allow({"AAA"}), Allow(set())]

        result = run(strict_then_stricter)

        # BBB, CCC and DDD all fail the second rule too, but the first one
        # removed them and keeps the attribution.
        assert result.exclusions["BBB"] == 1
        assert result.exclusions["CCC"] == 1
        assert result.exclusions["DDD"] == 1
        assert result.exclusions["AAA"] == 2

    def test_excluded_by_resolves_to_the_step(self):
        result = run([Allow({"AAA", "BBB", "CCC"}, name="FirstRule"),
                      Allow({"AAA"}, name="SecondRule")])

        assert result.excluded_by("DDD").rule_name == "FirstRule"
        assert result.excluded_by("BBB").rule_name == "SecondRule"

    def test_an_unknown_asset_has_no_step(self):
        result = run([Allow({"AAA"})])

        assert result.excluded_by("NOT_IN_UNIVERSE") is None

    def test_positions_index_the_steps_list(self):
        """The contract preview relies on to map positions to its own rule ids."""
        result = run([Allow({"AAA", "BBB"}), Allow({"AAA"})])

        for asset_id, position in result.exclusions.items():
            assert result.steps[position].position == position
            assert asset_id in result.steps[position].excluded


class TestRuleFailure:

    def test_a_raising_rule_excludes_rather_than_admits(self):
        """A rule that throws has not said the asset is eligible, and defaulting
        to inclusion would put a name into a live index on the strength of a
        bug."""
        result = run([Explode("CCC")])

        assert "CCC" not in result.survivor_ids
        assert result.exclusions["CCC"] == 1

    def test_the_other_assets_are_unaffected(self):
        result = run([Explode("CCC")])

        assert result.survivor_ids == ["AAA", "BBB", "DDD"]

    def test_it_is_logged_as_an_error(self,
                                     caplog):
        """A result-affecting failure, not a routine exclusion — the asset
        would very likely have qualified."""
        with caplog.at_level("ERROR"):
            run([Explode("CCC")])

        assert "ExplodeRule" in caplog.text
        assert "CCC" in caplog.text

    def test_a_routine_exclusion_is_only_debug(self,
                                               caplog):
        with caplog.at_level("ERROR"):
            run([Allow({"AAA"})])

        assert caplog.text == ""


class TestStepShape:

    def test_a_universe_step_knows_what_it_is(self):
        assert SelectionStep(position=UNIVERSE_POSITION, remaining=5).is_universe

    def test_a_rule_step_knows_what_it_is(self):
        assert not SelectionStep(position=1, remaining=5).is_universe

    def test_steps_default_to_no_exclusions(self):
        assert SelectionStep(position=0, remaining=0).excluded == []


class TestCalculatorProjection:
    """`select_constituents` is the survivors of the same walk, nothing else."""

    @pytest.fixture
    def calculator(self):
        from unittest.mock import MagicMock

        from beacon.index.calculation import IndexCalculator
        from beacon.index.constructor import IndexDefinition
        from beacon.index.methodology import EqualWeighted

        definition = IndexDefinition(
            index_id="SEL", index_name="Selection Test",
            base_date="2024-01-01", base_value=1000.0, currency="USD",
            eligibility_rules=[Allow({"AAA", "BBB", "CCC"}),
                               Allow({"AAA", "BBB"})],
            weighting_scheme=EqualWeighted(),
            rebalancing_frequency="QUARTERLY",
            universe_identifiers=UNIVERSE_IDS)

        return IndexCalculator(definition, MagicMock())

    def test_the_two_agree_on_survivors(self,
                                        calculator):
        """The regression guard: they are now the same walk, and must stay so."""
        thin = calculator.select_constituents(universe(), DATE)
        full = calculator.select_with_provenance(universe(), DATE)

        assert [asset.asset_id for asset in thin] == full.survivor_ids

    def test_the_fuller_call_carries_the_funnel(self,
                                                calculator):
        result = calculator.select_with_provenance(universe(), DATE)

        assert [step.remaining for step in result.steps] == [4, 3, 2]
        assert result.exclusions == {"DDD": 1, "CCC": 2}

    def test_an_empty_universe_still_reports_a_funnel(self,
                                                      calculator):
        """A caller rendering a waterfall should not have to special-case it."""
        result = calculator.select_with_provenance([], DATE)

        assert result.survivors == []
        assert result.steps[0].is_universe
        assert result.steps[0].remaining == 0

    def test_an_empty_universe_warns(self,
                                     calculator,
                                     caplog):
        with caplog.at_level("WARNING"):
            calculator.select_constituents([], DATE)

        assert "empty universe" in caplog.text

    def test_the_index_name_is_still_logged(self,
                                            calculator,
                                            caplog):
        """Log messages preserved through the merge: a refactor that quietly
        changed them would be a behavioural change wearing a refactor's
        clothes."""
        with caplog.at_level("INFO"):
            calculator.select_constituents(universe(), DATE)

        assert "Selection Test" in caplog.text
        assert "Universe size: 4" in caplog.text
        assert "Selected 2 constituents" in caplog.text
