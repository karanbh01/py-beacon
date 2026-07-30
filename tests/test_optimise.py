# tests/test_optimise.py
"""BN-92: constrained tracking-error minimisation."""
import numpy as np
import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.optimise import (
    BINDING_TOLERANCE,
    Cardinality,
    FullInvestment,
    GroupBounds,
    OptimisationResult,
    PositionBounds,
    TurnoverBudget,
    count_holdings,
    minimise_tracking_error,
)
from beacon.risk.model import estimate_risk_model

TARGET = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
UNIVERSE = list(TARGET.index)

# Tight enough to catch a solve that settled early, loose enough not to demand
# more of SLSQP than its convergence criterion promises.
PRECISION = 1e-8

# A starting portfolio well away from the target, so a turnover budget has
# something to bind on.
CURRENT_HOLDINGS = {"A": 0.2, "B": 0.2, "C": 0.6}


def solve(*constraints,
          target: pd.Series = TARGET,
          risk_model=None) -> OptimisationResult:
    """Run the optimiser over the module's target."""
    return minimise_tracking_error(target,
                                   list(constraints) or None,
                                   risk_model=risk_model)


@pytest.fixture
def correlated_risk_model():
    """A risk model in which C is very nearly a copy of A.

    Built so a risk-aware solve has something to be right about: when A is
    capped, its weight should move to C rather than being shared out evenly,
    because C carries almost the same risk.
    """
    generator = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=400)
    first = generator.normal(0.0, 0.01, 400)

    panel = pd.DataFrame(
        {"A": first,
         "B": generator.normal(0.0, 0.01, 400),
         "C": first + generator.normal(0.0, 0.0005, 400)},
        index=dates)

    return estimate_risk_model(panel, intensity=0.0)


class TestUnconstrainedSolve:

    def test_full_investment_alone_reproduces_the_target(self):
        """The closest feasible portfolio to a feasible target is the target."""
        result = solve(FullInvestment())

        pd.testing.assert_series_equal(result.weights,
                                       TARGET.rename("optimal_weight"),
                                       atol=PRECISION)

    def test_no_constraints_defaults_to_full_investment(self):
        result = solve()

        assert result.binding_labels() == ["full investment at 100.0000%"]

    def test_the_active_position_is_zero(self):
        result = solve(FullInvestment())

        assert result.active_weights.abs().max() < PRECISION

    def test_a_target_that_does_not_sum_to_one_is_pulled_up(self):
        """Not a no-op solve: the answer is nowhere near the starting point.

        Under an identity metric the shortfall is shared equally, because
        moving any name is equally costly.
        """
        target = pd.Series({"A": 0.4, "B": 0.2, "C": 0.2})

        result = solve(FullInvestment(), target=target)

        expected = target + 0.2 / 3
        assert (result.weights - expected).abs().max() < PRECISION

    def test_the_objective_is_zero_when_the_target_is_reachable(self):
        result = solve(FullInvestment())

        assert result.diagnostics.objective < PRECISION
        assert result.tracking_error() < 1e-4

    def test_the_solver_reports_convergence(self):
        result = solve(FullInvestment())

        assert result.diagnostics.converged
        assert result.diagnostics.iterations >= 1


class TestPositionBounds:

    def test_a_cap_binds_and_the_excess_is_shared(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))

        assert result.weights["A"] == pytest.approx(0.4, abs=PRECISION)
        assert result.weights["B"] == pytest.approx(0.35, abs=PRECISION)
        assert result.weights["C"] == pytest.approx(0.25, abs=PRECISION)

    def test_the_solution_still_sums_to_one(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))

        assert result.weights.sum() == pytest.approx(1.0, abs=PRECISION)

    def test_the_active_position_sums_to_zero(self):
        """The acceptance criterion: rearranging weight cannot create any."""
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))

        assert result.active_weights.sum() == pytest.approx(0.0, abs=PRECISION)

    def test_a_floor_lifts_an_underweight_name(self):
        result = solve(FullInvestment(), PositionBounds(0.25, 1.0))

        assert result.weights["C"] == pytest.approx(0.25, abs=PRECISION)

    def test_bounds_on_named_assets_leave_others_alone(self):
        result = solve(FullInvestment(),
                       PositionBounds(0.0, 0.4, assets=["A"]))

        assert result.weights["A"] == pytest.approx(0.4, abs=PRECISION)
        assert result.weights["B"] == pytest.approx(0.35, abs=PRECISION)

    def test_two_bounds_intersect_to_the_tighter(self):
        """A blanket rule plus a tighter one on a single name."""
        result = solve(FullInvestment(),
                       PositionBounds(0.0, 0.45),
                       PositionBounds(0.0, 0.35, assets=["A"]))

        assert result.weights["A"] == pytest.approx(0.35, abs=PRECISION)

    def test_a_bound_on_an_unknown_asset_is_rejected(self):
        with pytest.raises(CalculationError, match="outside the universe"):
            solve(FullInvestment(), PositionBounds(0.0, 0.4, assets=["ZZZ"]))

    def test_a_minimum_above_its_maximum_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            PositionBounds(0.5, 0.2)


class TestGroupBounds:

    def test_a_group_cap_binds(self):
        result = solve(FullInvestment(),
                       GroupBounds("tech", ["A", "B"], maximum=0.6))

        assert result.weights[["A", "B"]].sum() == pytest.approx(0.6, abs=PRECISION)

    def test_the_group_reduction_is_shared_within_the_group(self):
        """0.2 comes out of A and B equally; all of it lands on C."""
        result = solve(FullInvestment(),
                       GroupBounds("tech", ["A", "B"], maximum=0.6))

        assert result.weights["A"] == pytest.approx(0.4, abs=PRECISION)
        assert result.weights["B"] == pytest.approx(0.2, abs=PRECISION)
        assert result.weights["C"] == pytest.approx(0.4, abs=PRECISION)

    def test_a_group_floor_binds(self):
        result = solve(FullInvestment(),
                       GroupBounds("small", ["C"], minimum=0.35))

        assert result.weights["C"] == pytest.approx(0.35, abs=PRECISION)

    def test_a_slack_group_does_not_move_anything(self):
        result = solve(FullInvestment(),
                       GroupBounds("tech", ["A", "B"], maximum=0.95))

        assert (result.weights - TARGET).abs().max() < PRECISION

    def test_members_outside_the_universe_are_ignored(self):
        """A sector map covers a market, not just this index."""
        result = solve(FullInvestment(),
                       GroupBounds("tech", ["A", "B", "NOT_HELD"], maximum=0.6))

        assert result.weights[["A", "B"]].sum() == pytest.approx(0.6, abs=PRECISION)

    def test_a_group_matching_nothing_is_rejected(self):
        with pytest.raises(CalculationError, match="no members in the universe"):
            solve(FullInvestment(), GroupBounds("ghost", ["X", "Y"], maximum=0.5))

    def test_a_minimum_above_its_maximum_is_rejected(self):
        with pytest.raises(ValueError, match="above maximum"):
            GroupBounds("bad", ["A"], minimum=0.6, maximum=0.4)


class TestTurnoverBudget:

    def test_the_budget_binds(self):
        result = solve(FullInvestment(), TurnoverBudget(0.1, CURRENT_HOLDINGS))

        assert result.turnover(CURRENT_HOLDINGS) == pytest.approx(0.1, abs=1e-6)

    def test_the_budget_is_spent_where_it_helps_most(self):
        """A is furthest from target, so the whole budget goes there.

        Worth pinning down rather than only checking the budget: moving 0.1
        into A leaves a squared distance of 0.09, while splitting it between A
        and B leaves 0.105. A solver that merely respected the budget could
        return either.
        """
        result = solve(FullInvestment(), TurnoverBudget(0.1, CURRENT_HOLDINGS))

        assert result.weights["A"] == pytest.approx(0.3, abs=1e-6)
        assert result.weights["B"] == pytest.approx(0.2, abs=1e-6)
        assert result.weights["C"] == pytest.approx(0.5, abs=1e-6)

    def test_a_generous_budget_does_not_bind(self):
        result = solve(FullInvestment(), TurnoverBudget(1.0, CURRENT_HOLDINGS))

        assert (result.weights - TARGET).abs().max() < 1e-6

    def test_names_absent_from_the_current_holdings_count_as_unheld(self):
        result = solve(FullInvestment(), TurnoverBudget(1.0, {"A": 1.0}))

        assert result.weights.sum() == pytest.approx(1.0, abs=PRECISION)

    def test_a_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            TurnoverBudget(-0.1, CURRENT_HOLDINGS)

    def test_the_constraint_measures_turnover_itself(self):
        """Same number the result reports, from the same formula."""
        budget = TurnoverBudget(0.1, CURRENT_HOLDINGS)
        weights = np.array([0.3, 0.2, 0.5])

        assert budget.turnover(weights, UNIVERSE) == pytest.approx(0.1, abs=PRECISION)


class TestCardinality:

    def test_the_holding_limit_is_respected(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.holdings == 2

    def test_the_smallest_name_is_the_one_dropped(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.weights["C"] == pytest.approx(0.0, abs=PRECISION)

    def test_the_survivors_are_re_optimised_not_renormalised(self):
        """Renormalising A and B would give 0.625 and 0.375.

        Re-solving instead shares C's 0.2 equally, because under an identity
        metric that is the closest reachable point — which is a different
        answer, and the reason the second solve exists.
        """
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.weights["A"] == pytest.approx(0.6, abs=PRECISION)
        assert result.weights["B"] == pytest.approx(0.4, abs=PRECISION)

    def test_the_answer_is_flagged_as_heuristic(self):
        """Non-convex: satisfied, but not proven optimal, and it says so."""
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.heuristic

    def test_a_slack_limit_leaves_the_solve_exact(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(5))

        assert not result.heuristic
        assert (result.weights - TARGET).abs().max() < PRECISION

    def test_a_limit_below_the_forced_holdings_is_rejected(self):
        with pytest.raises(CalculationError, match="must be held"):
            solve(FullInvestment(), PositionBounds(0.1, 1.0), Cardinality(2))

    def test_a_limit_that_cannot_reach_full_investment_is_rejected(self):
        with pytest.raises(CalculationError, match="cannot reach"):
            solve(FullInvestment(), PositionBounds(0.0, 0.4), Cardinality(2))

    def test_a_limit_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="at least 1"):
            Cardinality(0)


class TestInfeasibility:

    def test_caps_that_cannot_reach_full_investment(self):
        with pytest.raises(CalculationError, match="cannot reach"):
            solve(FullInvestment(), PositionBounds(0.0, 0.3))

    def test_floors_that_overshoot_full_investment(self):
        with pytest.raises(CalculationError, match="already exceeds"):
            solve(FullInvestment(), PositionBounds(0.4, 1.0))

    def test_conflicting_bounds_on_one_asset(self):
        with pytest.raises(CalculationError, match="no allowed weight"):
            solve(FullInvestment(),
                  PositionBounds(0.5, 1.0, assets=["A"]),
                  PositionBounds(0.0, 0.2, assets=["A"]))

    def test_a_group_floor_its_members_cannot_reach(self):
        with pytest.raises(CalculationError, match="members' maximum weights"):
            solve(FullInvestment(),
                  PositionBounds(0.0, 0.2, assets=["A", "B"]),
                  GroupBounds("tech", ["A", "B"], minimum=0.7))

    def test_a_group_cap_its_members_already_breach(self):
        with pytest.raises(CalculationError, match="minimum weights already total"):
            solve(FullInvestment(),
                  PositionBounds(0.3, 1.0, assets=["A", "B"]),
                  GroupBounds("tech", ["A", "B"], maximum=0.4))

    def test_incompatible_groups_are_refused_rather_than_fudged(self):
        """Neither group is impossible alone; together they cannot both hold.

        The pre-solve checks look at one constraint at a time and cannot see
        this, so it is the verification pass that has to catch it — which is
        exactly what that pass is for.
        """
        with pytest.raises(CalculationError, match=r"no feasible portfolio|did not converge"):
            solve(FullInvestment(),
                  PositionBounds(0.0, 1.0),
                  GroupBounds("low", ["A", "B"], maximum=0.2),
                  GroupBounds("high", ["C"], maximum=0.3))

    def test_an_empty_target_is_rejected(self):
        with pytest.raises(CalculationError, match="empty"):
            minimise_tracking_error(pd.Series(dtype=float))


class TestWithoutFullInvestment:
    """Nothing fixes the total, so the weights are free to sum to anything."""

    def test_bounds_alone_leave_the_target_untouched(self):
        result = solve(PositionBounds(0.0, 1.0))

        assert (result.weights - TARGET).abs().max() < PRECISION

    def test_a_cap_is_applied_without_redistributing_the_excess(self):
        """Nothing requires the weight to be reinvested, so it is simply lost.

        The contrast with the full-investment case is the point: there, capping
        A at 0.4 pushed 0.1 onto B and C. Here it goes nowhere and the
        portfolio ends up 90% invested.
        """
        result = solve(PositionBounds(0.0, 0.4))

        assert result.weights["A"] == pytest.approx(0.4, abs=PRECISION)
        assert result.weights["B"] == pytest.approx(0.3, abs=PRECISION)
        assert result.weights.sum() == pytest.approx(0.9, abs=PRECISION)

    def test_a_holding_limit_still_applies(self):
        result = solve(PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.holdings == 2


class TestNonConvergence:

    def test_a_stalled_solve_is_refused(self,
                                        monkeypatch):
        """A feasible but unconverged point must not come back as an answer.

        Returning it with converged=False would make "this is optimal" and
        "this is merely feasible" look identical to anyone not reading the
        diagnostics, which for a weight vector someone is about to trade is
        the wrong default.
        """
        from beacon.optimise import solver

        real = solver.minimize

        def stalls(*args, **kwargs):
            outcome = real(*args, **kwargs)
            outcome.success = False
            outcome.status = 8
            outcome.message = "Positive directional derivative for linesearch"
            return outcome

        monkeypatch.setattr(solver, "minimize", stalls)

        with pytest.raises(CalculationError, match="did not converge"):
            solve(FullInvestment())

    def test_the_refusal_names_the_solver_message(self,
                                                  monkeypatch):
        from beacon.optimise import solver

        real = solver.minimize

        def stalls(*args, **kwargs):
            outcome = real(*args, **kwargs)
            outcome.success = False
            outcome.status = 8
            outcome.message = "Iteration limit reached"
            return outcome

        monkeypatch.setattr(solver, "minimize", stalls)

        with pytest.raises(CalculationError, match="Iteration limit reached"):
            solve(FullInvestment())


class TestBindingConstraints:

    def test_every_reported_binding_constraint_is_genuinely_tight(self):
        """The acceptance criterion, checked against the weights themselves.

        Each label is recomputed from the solution here rather than trusted
        from the report, so a binding list built from solver multipliers rather
        than from the answer would fail this.
        """
        result = solve(FullInvestment(),
                       PositionBounds(0.0, 0.4),
                       GroupBounds("tech", ["A", "B"], maximum=0.75))

        checks = {
            "maximum weight 40.0000% on A": result.weights["A"] - 0.4,
            "maximum 75.0000% in group 'tech'": result.weights[["A", "B"]].sum() - 0.75,
            "full investment at 100.0000%": result.weights.sum() - 1.0,
        }

        for label in result.binding_labels():
            assert abs(checks[label]) < BINDING_TOLERANCE, f"{label} is not tight"

    def test_a_slack_constraint_is_not_reported(self):
        result = solve(FullInvestment(),
                       PositionBounds(0.0, 0.9),
                       GroupBounds("tech", ["A", "B"], maximum=0.95))

        assert result.binding_labels() == ["full investment at 100.0000%"]

    def test_binding_constraints_come_tightest_first(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))
        slacks = [abs(constraint.slack) for constraint in result.binding]

        assert slacks == sorted(slacks)

    def test_slacks_cover_every_constraint_not_only_the_binding_ones(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))

        assert len(result.slacks) > len(result.binding)
        assert any(not slack.is_binding for slack in result.slacks)


class TestRiskModelObjective:

    def test_a_risk_model_routes_weight_to_the_correlated_substitute(self,
                                                                     correlated_risk_model):
        """The whole point of a risk model, in one assertion.

        With A capped, the freed weight can go to B or to C. C is 99.9%
        correlated with A, so a risk-aware solve puts it there; an identity
        metric has no way to know and splits it evenly.
        """
        rules = [FullInvestment(), PositionBounds(0.0, 0.35)]

        plain = solve(*rules)
        risky = solve(*rules, risk_model=correlated_risk_model)

        assert plain.weights["C"] == pytest.approx(0.30, abs=1e-6)
        assert risky.weights["C"] > 0.34

    def test_the_risk_aware_solve_tracks_better(self,
                                                correlated_risk_model):
        rules = [FullInvestment(), PositionBounds(0.0, 0.35)]

        plain = solve(*rules)
        risky = solve(*rules, risk_model=correlated_risk_model)

        measured = correlated_risk_model.tracking_error(plain.weights.to_dict(),
                                                        TARGET.to_dict())

        assert risky.tracking_error() < measured / 5

    def test_the_reported_tracking_error_matches_the_risk_model(self,
                                                                correlated_risk_model):
        """The objective is the risk model's own tracking error, not a proxy."""
        result = solve(FullInvestment(),
                       PositionBounds(0.0, 0.35),
                       risk_model=correlated_risk_model)

        expected = correlated_risk_model.tracking_error(result.weights.to_dict(),
                                                        TARGET.to_dict())

        assert result.tracking_error() == pytest.approx(expected, rel=1e-6)

    def test_a_risk_model_missing_an_asset_is_rejected(self,
                                                       correlated_risk_model):
        target = pd.Series({"A": 0.5, "B": 0.3, "D": 0.2})

        with pytest.raises(CalculationError, match="does not cover"):
            minimise_tracking_error(target,
                                    [FullInvestment()],
                                    risk_model=correlated_risk_model)

    def test_a_reachable_target_is_still_reproduced_exactly(self,
                                                            correlated_risk_model):
        """A singular metric must not let the solve wander off the target."""
        result = solve(FullInvestment(), risk_model=correlated_risk_model)

        assert (result.weights - TARGET).abs().max() < 1e-6


class TestResultAccessors:

    def test_a_mapping_is_accepted_as_the_target(self):
        result = minimise_tracking_error({"A": 0.6, "B": 0.4})

        assert result.asset_ids == ["A", "B"]

    def test_turnover_defaults_to_measuring_against_the_target(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 0.4))

        assert result.turnover() == pytest.approx(0.1, abs=1e-6)

    def test_turnover_against_explicit_holdings(self):
        result = solve(FullInvestment())

        assert result.turnover({"A": 0.5, "B": 0.3, "C": 0.2}) == pytest.approx(
            0.0, abs=1e-6)

    def test_turnover_treats_absent_holdings_as_unheld(self):
        result = solve(FullInvestment())

        assert result.turnover({"A": 1.0}) == pytest.approx(0.5, abs=1e-6)

    def test_the_frame_carries_all_three_weight_columns(self):
        frame = solve(FullInvestment(), PositionBounds(0.0, 0.4)).to_frame()

        assert list(frame.columns) == ["target_weight", "optimal_weight",
                                       "active_weight"]

    def test_the_frame_is_ordered_by_absolute_active_weight(self):
        frame = solve(FullInvestment(), PositionBounds(0.0, 0.4)).to_frame()

        assert frame.index[0] == "A"

    def test_holdings_counts_only_meaningful_weights(self):
        result = solve(FullInvestment(), PositionBounds(0.0, 1.0), Cardinality(2))

        assert result.holdings == 2
        assert len(result.weights) == 3

    def test_a_risk_model_can_be_bound_after_the_fact(self,
                                                      correlated_risk_model):
        result = solve(FullInvestment()).with_risk_model(correlated_risk_model)

        assert result._risk_model is correlated_risk_model


class TestCountHoldings:

    def test_zero_weights_do_not_count(self):
        assert count_holdings(np.array([0.5, 0.5, 0.0])) == 2

    def test_negligible_weights_do_not_count(self):
        assert count_holdings(np.array([1.0, 1e-12])) == 1

    def test_short_positions_count(self):
        assert count_holdings(np.array([1.2, -0.2])) == 2
