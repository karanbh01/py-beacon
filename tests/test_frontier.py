# tests/test_frontier.py
"""BN-93: the efficient frontier, minimum-variance and tangency points."""
import numpy as np
import pandas as pd
import pytest

from beacon.exceptions import CalculationError
from beacon.optimise import (
    Cardinality,
    FullInvestment,
    GroupBounds,
    PositionBounds,
    efficient_frontier,
    maximum_return_portfolio,
    minimum_variance_portfolio,
)
from beacon.optimise import frontier as frontier_module
from beacon.risk.model import estimate_risk_model

# D earns most and is riskiest; C earns least and is safest. A frontier over
# this universe therefore has somewhere to go.
EXPECTED_RETURNS = {"A": 0.06, "B": 0.10, "C": 0.04, "D": 0.14}
VOLATILITIES = {"A": 0.010, "B": 0.015, "C": 0.008, "D": 0.020}
RISK_FREE = 0.02

PRECISION = 1e-6


@pytest.fixture
def risk_model():
    """A four-asset risk model with genuinely different volatilities."""
    generator = np.random.default_rng(11)
    dates = pd.bdate_range("2024-01-01", periods=500)

    panel = pd.DataFrame(
        {asset_id: generator.normal(0.0, volatility, 500)
         for asset_id, volatility in VOLATILITIES.items()},
        index=dates)

    return estimate_risk_model(panel, intensity=0.1)


@pytest.fixture
def frontier(risk_model):
    return efficient_frontier(risk_model, EXPECTED_RETURNS, points=9,
                              risk_free_rate=RISK_FREE)


class TestShape:

    def test_the_grid_has_the_requested_number_of_points(self,
                                                          frontier):
        assert len(frontier.points) == 9

    def test_risk_rises_with_return(self,
                                    frontier):
        """The defining property: insisting on more return can only cost risk."""
        assert frontier.is_monotonic()

    def test_returns_increase_across_the_grid(self,
                                              frontier):
        returns = [point.expected_return for point in frontier.points]

        assert returns == sorted(returns)

    def test_the_grid_starts_at_the_minimum_variance_portfolio(self,
                                                                frontier):
        assert frontier.points[0].volatility == pytest.approx(
            frontier.minimum_variance.volatility, abs=PRECISION)

    def test_the_grid_ends_at_the_highest_reachable_return(self,
                                                            frontier):
        """Long-only and fully invested, that is the best single asset."""
        assert frontier.points[-1].expected_return == pytest.approx(
            max(EXPECTED_RETURNS.values()), abs=PRECISION)

    def test_every_point_is_fully_invested(self,
                                           frontier):
        for point in frontier.points:
            assert point.weights.sum() == pytest.approx(1.0, abs=PRECISION)

    def test_every_point_is_long_only(self,
                                      frontier):
        for point in frontier.points:
            assert point.weights.min() >= -PRECISION

    def test_the_frame_has_one_row_per_point(self,
                                             frontier):
        frame = frontier.to_frame()

        assert len(frame) == 9
        assert list(frame.columns) == ["expected_return", "volatility",
                                       "sharpe_ratio", "binding"]

    def test_the_weights_frame_has_a_column_per_asset(self,
                                                      frontier):
        frame = frontier.weights_frame()

        assert list(frame.columns) == list(EXPECTED_RETURNS)
        assert len(frame) == 9


class TestMinimumVariance:

    def test_it_is_the_least_risky_point_on_the_frontier(self,
                                                          frontier):
        assert frontier.minimum_variance.volatility == pytest.approx(
            min(frontier.volatilities), abs=PRECISION)

    def test_it_beats_every_equally_weighted_alternative(self,
                                                          risk_model):
        """A real optimisation, not a relabelled equal weighting."""
        point = minimum_variance_portfolio(risk_model)
        equal = dict.fromkeys(VOLATILITIES, 0.25)

        assert point.volatility < risk_model.portfolio_volatility(equal)

    def test_it_tilts_towards_the_least_volatile_asset(self,
                                                        risk_model):
        point = minimum_variance_portfolio(risk_model)

        assert point.weights.idxmax() == "C"

    def test_it_reports_no_return_when_none_was_given(self,
                                                       risk_model):
        """A return cannot be reported if it was never supplied."""
        point = minimum_variance_portfolio(risk_model)

        assert point.expected_return is None
        assert point.sharpe_ratio is None

    def test_it_reports_a_return_when_one_was_given(self,
                                                     risk_model):
        point = minimum_variance_portfolio(risk_model,
                                           expected_returns=EXPECTED_RETURNS,
                                           risk_free_rate=RISK_FREE)

        assert point.expected_return is not None
        assert point.sharpe_ratio == pytest.approx(
            (point.expected_return - RISK_FREE) / point.volatility, rel=1e-9)

    def test_tightening_the_constraints_cannot_lower_the_minimum(self,
                                                                  risk_model):
        """A theorem: a smaller feasible set cannot contain a better optimum."""
        loose = minimum_variance_portfolio(risk_model)
        tight = minimum_variance_portfolio(
            risk_model, [FullInvestment(), PositionBounds(0.0, 0.30)])

        assert tight.volatility >= loose.volatility - PRECISION


class TestMaximumReturn:

    def test_long_only_it_concentrates_in_the_best_asset(self,
                                                          risk_model):
        point = maximum_return_portfolio(risk_model, EXPECTED_RETURNS)

        assert point.weights["D"] == pytest.approx(1.0, abs=PRECISION)

    def test_a_cap_forces_it_to_spread_down_the_ranking(self,
                                                         risk_model):
        """With a 30% cap it fills D, B, A to the cap and C takes the rest.

        Worth pinning the exact weights: a solver that merely respected the cap
        could return many other feasible portfolios.
        """
        point = maximum_return_portfolio(
            risk_model, EXPECTED_RETURNS,
            [FullInvestment(), PositionBounds(0.0, 0.30)])

        assert point.weights["D"] == pytest.approx(0.30, abs=PRECISION)
        assert point.weights["B"] == pytest.approx(0.30, abs=PRECISION)
        assert point.weights["A"] == pytest.approx(0.30, abs=PRECISION)
        assert point.weights["C"] == pytest.approx(0.10, abs=PRECISION)

    def test_the_capped_maximum_is_lower_than_the_uncapped_one(self,
                                                                risk_model):
        uncapped = maximum_return_portfolio(risk_model, EXPECTED_RETURNS)
        capped = maximum_return_portfolio(
            risk_model, EXPECTED_RETURNS,
            [FullInvestment(), PositionBounds(0.0, 0.30)])

        assert capped.expected_return < uncapped.expected_return

    def test_an_unbounded_problem_is_refused(self,
                                             risk_model):
        """Without a cap, shorting one asset to fund another has no limit."""
        with pytest.raises(CalculationError, match="unbounded"):
            maximum_return_portfolio(risk_model, EXPECTED_RETURNS,
                                     [FullInvestment()])


class TestTangency:

    def test_it_has_the_best_sharpe_ratio_on_the_frontier(self,
                                                          frontier):
        best_on_grid = max(point.sharpe_ratio for point in frontier.points)

        assert frontier.tangency.sharpe_ratio >= best_on_grid - 1e-9

    def test_refining_off_the_grid_actually_improves_on_it(self,
                                                            frontier):
        """The refinement stage earns its place.

        The grid is spaced by return, so its best point is only as close to the
        tangency as the spacing allows. If this ever became an equality the
        refinement would be doing nothing.
        """
        best_on_grid = max(point.sharpe_ratio for point in frontier.points)

        assert frontier.tangency.sharpe_ratio > best_on_grid

    def test_it_beats_the_minimum_variance_portfolio_on_sharpe(self,
                                                                frontier):
        assert frontier.tangency.sharpe_ratio > frontier.minimum_variance.sharpe_ratio

    def test_it_is_fully_invested_and_long_only(self,
                                                frontier):
        assert frontier.tangency.weights.sum() == pytest.approx(1.0, abs=PRECISION)
        assert frontier.tangency.weights.min() >= -PRECISION

    def test_constraining_cannot_improve_the_best_sharpe(self,
                                                          risk_model):
        loose = efficient_frontier(risk_model, EXPECTED_RETURNS, points=9,
                                   risk_free_rate=RISK_FREE)
        tight = efficient_frontier(
            risk_model, EXPECTED_RETURNS, points=9, risk_free_rate=RISK_FREE,
            constraints=[FullInvestment(), PositionBounds(0.0, 0.30)])

        assert tight.tangency.sharpe_ratio <= loose.tangency.sharpe_ratio + 1e-9

    def test_a_higher_risk_free_rate_moves_the_tangency_out(self,
                                                             risk_model):
        """Raising the hurdle makes safe, low-return portfolios less attractive."""
        low = efficient_frontier(risk_model, EXPECTED_RETURNS, points=9,
                                 risk_free_rate=0.0)
        high = efficient_frontier(risk_model, EXPECTED_RETURNS, points=9,
                                  risk_free_rate=0.05)

        assert high.tangency.expected_return >= low.tangency.expected_return - 1e-6


class TestConstrainedFrontiers:

    def test_a_group_cap_binds_at_every_point_it_should(self,
                                                         risk_model):
        frontier = efficient_frontier(
            risk_model, EXPECTED_RETURNS, points=7,
            constraints=[FullInvestment(), PositionBounds(0.0, 1.0),
                         GroupBounds("risky", ["B", "D"], maximum=0.5)])

        for point in frontier.points:
            assert point.weights[["B", "D"]].sum() <= 0.5 + PRECISION

    def test_the_constrained_frontier_is_shorter(self,
                                                  risk_model,
                                                  frontier):
        constrained = efficient_frontier(
            risk_model, EXPECTED_RETURNS, points=7,
            constraints=[FullInvestment(), PositionBounds(0.0, 1.0),
                         GroupBounds("risky", ["B", "D"], maximum=0.5)])

        assert (constrained.points[-1].expected_return
                < frontier.points[-1].expected_return)

    def test_a_holding_limit_is_respected_at_every_point(self,
                                                          risk_model):
        """Three of four names is loose enough for the heuristic to work."""
        frontier = efficient_frontier(
            risk_model, EXPECTED_RETURNS, points=5,
            constraints=[FullInvestment(), PositionBounds(0.0, 1.0),
                         Cardinality(3)])

        for point in frontier.points:
            assert (point.weights.abs() > 1e-6).sum() <= 3

    def test_a_holding_limit_the_heuristic_cannot_meet_says_so(self,
                                                               risk_model):
        """Two names cannot generally hit an arbitrary return target.

        The heuristic keeps the largest positions of the unrestricted solve,
        which is blind to whether those two can still earn what was asked. The
        error has to name the heuristic rather than the return constraint — the
        caller did not ask for something impossible, the selection failed.
        """
        with pytest.raises(CalculationError, match="combinatorial problem"):
            efficient_frontier(
                risk_model, EXPECTED_RETURNS, points=5,
                constraints=[FullInvestment(), PositionBounds(0.0, 1.0),
                             Cardinality(2)])

    def test_every_point_reports_which_constraints_bound(self,
                                                          frontier):
        """Full investment binds everywhere by construction."""
        for point in frontier.points:
            assert any("full investment" in label for label in point.binding)


class TestDegenerateCases:

    def test_equal_expected_returns_collapse_the_frontier(self,
                                                           risk_model):
        """Nothing to trade off: every asset earns the same.

        The grid still has the requested number of points, all identical, which
        is honest — the frontier really is one portfolio.
        """
        flat = dict.fromkeys(VOLATILITIES, 0.07)

        frontier = efficient_frontier(risk_model, flat, points=5)
        volatilities = frontier.volatilities

        assert max(volatilities) - min(volatilities) < 1e-6

    def test_the_collapsed_frontier_is_the_minimum_variance_portfolio(self,
                                                                       risk_model):
        flat = dict.fromkeys(VOLATILITIES, 0.07)

        frontier = efficient_frontier(risk_model, flat, points=5)

        assert frontier.points[0].volatility == pytest.approx(
            frontier.minimum_variance.volatility, abs=PRECISION)

    def test_a_single_point_frontier_is_refused(self,
                                                risk_model):
        with pytest.raises(CalculationError, match="at least 2 points"):
            efficient_frontier(risk_model, EXPECTED_RETURNS, points=1)

    def test_a_missing_expected_return_is_refused(self,
                                                   risk_model):
        """Defaulting it to zero would bias the whole frontier."""
        partial = {"A": 0.06, "B": 0.10, "C": 0.04}

        with pytest.raises(CalculationError, match="no expected return"):
            efficient_frontier(risk_model, partial, points=5)

    def test_an_infeasible_constraint_set_is_refused(self,
                                                      risk_model):
        with pytest.raises(CalculationError, match="cannot reach"):
            efficient_frontier(risk_model, EXPECTED_RETURNS, points=5,
                               constraints=[FullInvestment(),
                                            PositionBounds(0.0, 0.2)])


class TestExpectedReturnTarget:

    def test_each_grid_point_earns_exactly_what_it_was_asked_for(self,
                                                                  frontier):
        """The constraint is an equality, so every point hits its target.

        Recomputed from the weights rather than trusted from the solver, since
        a point that missed its target would still look fine in the summary.
        """
        returns = pd.Series(EXPECTED_RETURNS)

        for point in frontier.points:
            realised = float((point.weights * returns.reindex(point.weights.index)).sum())
            assert realised == pytest.approx(point.expected_return, abs=PRECISION)

    def test_the_return_target_shows_up_as_binding(self,
                                                    frontier):
        """Every interior point is held at its return, not resting below it."""
        interior = frontier.points[1:-1]

        for point in interior:
            assert any("expected return" in label for label in point.binding)


class TestRisklessAsset:
    """A zero-variance asset makes the Sharpe ratio's denominator vanish.

    Not a hypothetical: a covariance can be exactly singular, and comparing a
    volatility against zero rather than against a threshold is the mistake this
    codebase has made three times. The guard has to be exercised.
    """

    @pytest.fixture
    def with_cash(self):
        generator = np.random.default_rng(2)
        dates = pd.bdate_range("2024-01-01", periods=300)

        panel = pd.DataFrame({"RISKY": generator.normal(0.0, 0.02, 300),
                              "CASH": np.zeros(300)},
                             index=dates)

        return estimate_risk_model(panel, intensity=0.0)

    def test_the_minimum_variance_portfolio_is_all_cash(self,
                                                        with_cash):
        point = minimum_variance_portfolio(with_cash)

        assert point.weights["CASH"] == pytest.approx(1.0, abs=PRECISION)
        assert point.volatility == pytest.approx(0.0, abs=1e-9)

    def test_a_riskless_portfolio_reports_no_sharpe_ratio(self,
                                                          with_cash):
        """Excess return over zero volatility is not a number worth inventing."""
        point = minimum_variance_portfolio(with_cash,
                                           expected_returns={"RISKY": 0.10,
                                                             "CASH": 0.02},
                                           risk_free_rate=0.02)

        assert point.sharpe_ratio is None

    def test_a_frontier_over_a_riskless_asset_still_traces(self,
                                                           with_cash):
        frontier = efficient_frontier(with_cash,
                                      {"RISKY": 0.10, "CASH": 0.02},
                                      points=5,
                                      risk_free_rate=0.02)

        assert frontier.is_monotonic()
        assert frontier.points[0].volatility == pytest.approx(0.0, abs=1e-9)


class TestFrontierAccessors:

    def test_expected_returns_are_exposed_in_grid_order(self,
                                                        frontier):
        assert frontier.expected_returns == [point.expected_return
                                             for point in frontier.points]
        assert frontier.expected_returns == sorted(frontier.expected_returns)

    def test_the_risk_free_rate_is_recorded(self,
                                            frontier):
        assert frontier.risk_free_rate == RISK_FREE


class TestSharpeGuards:
    """The zero-volatility guards inside the Sharpe objective.

    Exercised directly rather than through a frontier: they protect against an
    iterate the solver may propose mid-search — an all-cash portfolio, once a
    riskless asset is in the universe — and there is no reliable way to force
    the search through that point from outside. Comparing a volatility against
    exact zero instead of a threshold is the mistake this codebase has already
    made three times, so the guard is worth pinning down.
    """

    COVARIANCE = np.diag([0.04, 0.0])
    RETURNS = np.array([0.10, 0.02])

    def test_a_riskless_portfolio_scores_zero_rather_than_dividing_by_zero(self):
        riskless = np.array([0.0, 1.0])

        assert frontier_module._sharpe(riskless, self.COVARIANCE,
                                       self.RETURNS, 0.02) == 0.0

    def test_a_risky_portfolio_scores_the_ratio(self):
        risky = np.array([1.0, 0.0])

        assert frontier_module._sharpe(risky, self.COVARIANCE,
                                       self.RETURNS, 0.02) == pytest.approx(
                                           (0.10 - 0.02) / 0.2, rel=1e-12)

    def test_the_volatility_helper_clamps_float_noise_at_zero(self):
        """A PSD covariance cannot give a negative variance; noise can."""
        assert frontier_module._volatility(np.array([0.0, 1.0]),
                                           self.COVARIANCE) == 0.0
