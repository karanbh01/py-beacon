# src/beacon/optimise/frontier.py
"""
The efficient frontier, and the two points on it worth naming.

A frontier is not one optimisation but a sequence of them: fix the expected
return, minimise the variance, repeat across a grid. Each point answers "if I
insist on earning this much, what is the least risk I can take to do it", and
the curve through them is the boundary of what is achievable.

Two points on that curve are singled out because they answer questions people
actually ask:

* the **minimum-variance** portfolio — the least risky feasible portfolio,
  ignoring return entirely. It is the frontier's left-hand end, and the reason
  the grid starts there: portfolios with lower return than this exist, but each
  is beaten by one on the frontier with the same risk and more return.
* the **tangency** portfolio — the highest Sharpe ratio available, the point
  where a line from the risk-free rate first touches the frontier.

## What constraints do to this

Textbook frontiers are drawn with only a budget constraint, which admits a
closed form. Every real mandate has more than that, and once position bounds
and group limits are in play there is no closed form and the curve has to be
traced numerically. Two consequences worth stating plainly:

The frontier can be **shorter** than the unconstrained one at both ends — a cap
limits how much can be put into the highest-returning asset, so the right-hand
end stops early — and it lies **below** it everywhere in between, because every
constraint removes portfolios and can only make the best remaining one worse.

Maximising the Sharpe ratio is not a convex problem in general. The solve is
warm-started from the best point on the grid, which in the long-only fully
invested case is enough for the answer to be the global one; the grid is
returned alongside so a caller can see the curve the tangency point sits on.
"""
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from ..exceptions import CalculationError
from ..risk.model import RiskModel
from .constraints import (
    Constraint,
    ExpectedReturnTarget,
    FullInvestment,
    PositionBounds,
    Vector,
)
from .solver import Solution, covariance_matrix, solve_constrained, weight_box

logger = logging.getLogger(__name__)

# Below this volatility a Sharpe ratio is not a meaningful number: the
# denominator is float noise and the ratio explodes. A riskless portfolio is a
# real possibility with a singular covariance, so this is a case to handle
# rather than an error to raise.
NEGLIGIBLE_VOLATILITY = 1e-12

# A return range narrower than this makes the grid degenerate — every point
# would be the same portfolio. Happens when every asset has the same expected
# return, in which case there is genuinely only one efficient portfolio.
NEGLIGIBLE_RETURN_RANGE = 1e-12

DEFAULT_POINTS = 20


@dataclass(frozen=True)
class FrontierPoint:
    """One portfolio on the frontier.

    Attributes:
        weights: The portfolio, indexed by asset id.
        volatility: Annualised standard deviation, in the risk model's units.
        expected_return: Portfolio expected return, or None when no expected
            returns were supplied — a return cannot be reported if it was never
            given.
        sharpe_ratio: Excess return over volatility, or None when the expected
            return is unknown or the volatility is negligible.
        binding: Labels of the constraints this point sits on. The interesting
            part of a frontier point: it says which rule is what stops the
            portfolio from doing better.
        heuristic: Whether a non-convex constraint forced a restricted
            re-solve, so this point is feasible but not proven optimal.
    """
    weights: pd.Series
    volatility: float
    expected_return: float | None = None
    sharpe_ratio: float | None = None
    binding: list[str] = field(default_factory=list)
    heuristic: bool = False


@dataclass
class EfficientFrontier:
    """A traced frontier and the points on it worth naming.

    Attributes:
        points: The grid, in increasing order of expected return. The first is
            the minimum-variance portfolio and the last is the highest return
            the constraints allow.
        minimum_variance: The least risky feasible portfolio.
        tangency: The highest Sharpe ratio available.
        risk_free_rate: The rate the Sharpe ratios were computed against.
    """
    points: list[FrontierPoint]
    minimum_variance: FrontierPoint
    tangency: FrontierPoint
    risk_free_rate: float = 0.0

    @property
    def volatilities(self) -> list[float]:
        """Each point's volatility, in grid order."""
        return [point.volatility for point in self.points]

    @property
    def expected_returns(self) -> list[float | None]:
        """Each point's expected return, in grid order."""
        return [point.expected_return for point in self.points]

    def is_monotonic(self,
                     tolerance: float = 1e-7) -> bool:
        """Whether risk rises with return across the grid.

        The defining property of a frontier. It should always hold — insisting
        on more return can only cost risk — so a False here means a point
        failed to solve to optimality rather than that the curve is unusual.
        """
        return all(later >= earlier - tolerance
                   for earlier, later in pairwise(self.volatilities))

    def to_frame(self) -> pd.DataFrame:
        """The frontier as a table, one row per point."""
        return pd.DataFrame([
            {"expected_return": point.expected_return,
             "volatility": point.volatility,
             "sharpe_ratio": point.sharpe_ratio,
             "binding": len(point.binding)}
            for point in self.points
        ])

    def weights_frame(self) -> pd.DataFrame:
        """Every point's weights, points on the index and assets on the columns."""
        return pd.DataFrame([point.weights for point in self.points]).reset_index(
            drop=True)


def default_constraints() -> list[Constraint]:
    """Long-only and fully invested.

    The frontier's default rather than full investment alone, because with
    shorting unbounded the maximum-return portfolio does not exist: short the
    worst asset without limit to fund the best, and the return grows forever.
    A frontier needs a right-hand end.
    """
    return [FullInvestment(), PositionBounds(0.0, 1.0)]


def minimum_variance_portfolio(risk_model: RiskModel,
                               constraints: Sequence[Constraint] | None = None,
                               expected_returns: dict[str, float] | None = None,
                               risk_free_rate: float = 0.0) -> FrontierPoint:
    """The least risky feasible portfolio.

    Args:
        risk_model: The covariance to minimise against.
        constraints: What the answer must satisfy. None means long-only and
            fully invested.
        expected_returns: Used only to report the point's return and Sharpe
            ratio; it has no effect on the weights, since minimum variance
            ignores return by definition.
        risk_free_rate: For the reported Sharpe ratio.

    Returns:
        FrontierPoint: The portfolio, its risk, and which constraints bound.

    Raises:
        CalculationError: If the constraints cannot be satisfied.
    """
    rules = list(constraints) if constraints is not None else default_constraints()
    assets = list(risk_model.covariance.index)
    covariance = covariance_matrix(risk_model, assets)

    objective, gradient = _variance_objective(covariance)
    solved = solve_constrained(objective, gradient, rules, assets)

    return _point(solved, assets, covariance, expected_returns, risk_free_rate)


def maximum_return_portfolio(risk_model: RiskModel,
                             expected_returns: dict[str, float],
                             constraints: Sequence[Constraint] | None = None,
                             risk_free_rate: float = 0.0) -> FrontierPoint:
    """The highest-returning feasible portfolio.

    The frontier's right-hand end. Maximising a linear objective often has many
    optimal solutions — any mix of the top assets, if they tie — so the return
    is found first and the variance minimised subject to achieving it, which
    picks the sensible one out of the tie.

    Args:
        risk_model: The covariance, used to break ties and report risk.
        expected_returns: Expected return per asset.
        constraints: What the answer must satisfy. None means long-only and
            fully invested.
        risk_free_rate: For the reported Sharpe ratio.

    Returns:
        FrontierPoint: The portfolio.
    """
    rules = list(constraints) if constraints is not None else default_constraints()
    assets = list(risk_model.covariance.index)
    _reject_unbounded(rules, assets)

    returns = _return_vector(expected_returns, assets)
    covariance = covariance_matrix(risk_model, assets)

    def objective(weights: Vector) -> float:
        return -float(returns @ weights)

    def gradient(weights: Vector) -> Vector:
        return -returns

    solved = solve_constrained(objective, gradient, rules, assets)
    best = float(returns @ solved.weights)

    return _solve_at_return(best, rules, assets, covariance, returns,
                            expected_returns, risk_free_rate, hint=solved.weights)


def efficient_frontier(risk_model: RiskModel,
                       expected_returns: dict[str, float],
                       points: int = DEFAULT_POINTS,
                       constraints: Sequence[Constraint] | None = None,
                       risk_free_rate: float = 0.0) -> EfficientFrontier:
    """Trace the frontier, and locate its minimum-variance and tangency points.

    Args:
        risk_model: The covariance. Its asset order defines the universe.
        expected_returns: Expected return per asset, in the same units and over
            the same horizon as the covariance — annualised, if it is.
        points: How many portfolios to solve for, minimum 2.
        constraints: What every point must satisfy. None means long-only and
            fully invested.
        risk_free_rate: The rate Sharpe ratios are measured against.

    Returns:
        EfficientFrontier: The grid and the two named points.

    Raises:
        CalculationError: If *points* is below 2, if the constraints cannot be
            satisfied, or if the problem is unbounded above.
    """
    if points < 2:
        raise CalculationError("Frontier",
                               f"a frontier needs at least 2 points, got {points}.")

    rules = list(constraints) if constraints is not None else default_constraints()
    assets = list(risk_model.covariance.index)
    covariance = covariance_matrix(risk_model, assets)
    returns = _return_vector(expected_returns, assets)

    lowest = minimum_variance_portfolio(risk_model, rules, expected_returns,
                                        risk_free_rate)
    highest = maximum_return_portfolio(risk_model, expected_returns, rules,
                                       risk_free_rate)

    if _is_degenerate(lowest, highest):
        return _collapsed(lowest, points, risk_free_rate)

    grid = list(np.linspace(lowest.expected_return or 0.0,
                            highest.expected_return or 0.0,
                            points))
    traced = [lowest] + [
        _solve_at_return(target, rules, assets, covariance, returns,
                         expected_returns, risk_free_rate)
        for target in grid[1:-1]
    ] + [highest]

    tangency = _tangency(traced, rules, assets, covariance, returns,
                         expected_returns, risk_free_rate)

    logger.info(
        f"Traced a {len(traced)}-point frontier over {len(assets)} assets: "
        f"return {grid[0]:.4%} to {grid[-1]:.4%}, minimum volatility "
        f"{lowest.volatility:.4%}, tangency Sharpe "
        f"{tangency.sharpe_ratio or float('nan'):.4f}.")

    return EfficientFrontier(points=traced,
                             minimum_variance=lowest,
                             tangency=tangency,
                             risk_free_rate=risk_free_rate)


def _is_degenerate(lowest: FrontierPoint,
                   highest: FrontierPoint) -> bool:
    """Whether the frontier's two ends earn the same, leaving nothing to trace."""
    return ((highest.expected_return or 0.0) - (lowest.expected_return or 0.0)
            <= NEGLIGIBLE_RETURN_RANGE)


def _collapsed(point: FrontierPoint,
               points: int,
               risk_free_rate: float) -> EfficientFrontier:
    """The frontier when every reachable portfolio earns the same.

    The grid repeats the one portfolio rather than solving for it again at each
    target. That is not a shortcut: with returns flat, the expected-return
    constraint is a multiple of the full-investment constraint, the two
    gradients are parallel, and the solver fails on a singular subproblem. The
    honest answer is that there is one efficient portfolio, so it is returned
    once per requested point.
    """
    logger.info(
        "Expected returns span nothing, so there is a single efficient "
        "portfolio; the frontier repeats it.")

    return EfficientFrontier(points=[point] * points,
                             minimum_variance=point,
                             tangency=point,
                             risk_free_rate=risk_free_rate)


def _solve_at_return(target: float,
                     rules: Sequence[Constraint],
                     assets: Sequence[str],
                     covariance: Vector,
                     returns: Vector,
                     expected_returns: dict[str, float],
                     risk_free_rate: float,
                     hint: Vector | None = None) -> FrontierPoint:
    """Least risk achievable while earning exactly *target*."""
    pinned = list(rules)
    if not _return_target_is_redundant(returns, rules):
        pinned.append(ExpectedReturnTarget(expected_returns, target))

    objective, gradient = _variance_objective(covariance)
    solved = solve_constrained(objective, gradient, pinned, assets, hint=hint)

    return _point(solved, assets, covariance, expected_returns, risk_free_rate)


def _return_target_is_redundant(returns: Vector,
                                rules: Sequence[Constraint]) -> bool:
    """Whether pinning the return would add nothing but a singular row.

    When every asset earns the same, the portfolio return is that number times
    the amount invested. If something already fixes the amount invested, the
    return constraint is a multiple of it — the two gradients are parallel, the
    solver's subproblem goes singular, and it fails with a message about
    matrices that tells the caller nothing. Dropping the redundant row is not a
    workaround: the constraint genuinely carries no information here.
    """
    spread = float(returns.max() - returns.min()) if len(returns) else 0.0
    fixes_the_total = any(isinstance(rule, FullInvestment) for rule in rules)

    return spread <= NEGLIGIBLE_RETURN_RANGE and fixes_the_total


def _tangency(traced: Sequence[FrontierPoint],
              rules: Sequence[Constraint],
              assets: Sequence[str],
              covariance: Vector,
              returns: Vector,
              expected_returns: dict[str, float],
              risk_free_rate: float) -> FrontierPoint:
    """The highest Sharpe ratio available, refined off the best grid point.

    Maximising a ratio is not convex, so the grid does double duty: it is the
    curve a caller wanted anyway, and it is a starting point good enough that
    the local optimum the solver walks to is the global one for the long-only
    fully invested case.
    """
    best = max(traced, key=lambda point: point.sharpe_ratio or -np.inf)

    def objective(weights: Vector) -> float:
        return -_sharpe(weights, covariance, returns, risk_free_rate)

    def gradient(weights: Vector) -> Vector:
        volatility = _volatility(weights, covariance)
        if volatility <= NEGLIGIBLE_VOLATILITY:
            return np.zeros(len(weights))

        excess = float(returns @ weights) - risk_free_rate
        derivative = returns / volatility - excess * (covariance @ weights) / volatility**3

        return np.asarray(-derivative, dtype=np.float64)

    solved = solve_constrained(objective, gradient, rules, assets,
                               hint=best.weights.to_numpy(dtype=float))
    refined = _point(solved, assets, covariance, expected_returns, risk_free_rate)

    # The grid point is kept if refining did not beat it, which can happen when
    # the tangency sits exactly on a grid node.
    if (refined.sharpe_ratio or -np.inf) < (best.sharpe_ratio or -np.inf):
        return best

    return refined


def _variance_objective(
        covariance: Vector) -> tuple[Callable[[Vector], float],
                                     Callable[[Vector], Vector]]:
    """Portfolio variance and its gradient."""
    def objective(weights: Vector) -> float:
        return float(weights @ covariance @ weights)

    def gradient(weights: Vector) -> Vector:
        return np.asarray(2.0 * covariance @ weights, dtype=np.float64)

    return objective, gradient


def _volatility(weights: Vector,
                covariance: Vector) -> float:
    """Portfolio standard deviation, clamped at zero.

    A PSD covariance cannot produce a negative variance, so a negative value is
    float noise on a near-zero result and its square root would be nan.
    """
    variance = float(weights @ covariance @ weights)

    return float(np.sqrt(max(variance, 0.0)))


def _sharpe(weights: Vector,
            covariance: Vector,
            returns: Vector,
            risk_free_rate: float) -> float:
    """Excess return per unit of volatility, zero when there is no volatility."""
    volatility = _volatility(weights, covariance)
    if volatility <= NEGLIGIBLE_VOLATILITY:
        return 0.0

    return (float(returns @ weights) - risk_free_rate) / volatility


def _point(solved: Solution,
           assets: Sequence[str],
           covariance: Vector,
           expected_returns: dict[str, float] | None,
           risk_free_rate: float) -> FrontierPoint:
    """Assemble a frontier point from a verified solution."""
    weights = solved.weights
    volatility = _volatility(weights, covariance)

    expected: float | None = None
    sharpe: float | None = None

    if expected_returns is not None:
        returns = _return_vector(expected_returns, assets)
        expected = float(returns @ weights)
        if volatility > NEGLIGIBLE_VOLATILITY:
            sharpe = (expected - risk_free_rate) / volatility

    return FrontierPoint(
        weights=pd.Series(weights, index=list(assets), name="weight"),
        volatility=volatility,
        expected_return=expected,
        sharpe_ratio=sharpe,
        binding=[slack.label for slack in solved.slacks if slack.is_binding],
        heuristic=solved.heuristic)


def _return_vector(expected_returns: dict[str, float],
                   assets: Sequence[str]) -> Vector:
    """Expected returns aligned to the universe."""
    missing = sorted(set(assets) - set(expected_returns))
    if missing:
        raise CalculationError("Frontier",
                               f"no expected return was given for: {missing}.")

    return np.array([expected_returns[asset_id] for asset_id in assets])


def _reject_unbounded(rules: Sequence[Constraint],
                      assets: Sequence[str]) -> None:
    """Raise if nothing caps the weights, which leaves the return unbounded."""
    _, highs = weight_box(rules, assets)

    if not bool(np.isfinite(highs).all()):
        raise CalculationError(
            "Frontier",
            "the maximum-return portfolio is unbounded: no constraint caps the "
            "weights, so return grows without limit by shorting one asset to "
            "fund another. Add PositionBounds.")
