# src/beacon/strategy/constructions.py
"""
How an active strategy builds its portfolio at each rebalance.

Every construction is long-only and fully invested, and meets the strategy's
constraints (see `beacon.strategy.constraints`). Each takes the rebalance's
problem (its names, the benchmark's weights and their covariance) and the
signal's scores, and returns the weights.

- `MaxAlpha(tracking_error)`: the most exposure to the scores within a
  tracking-error budget.
- `MeanVariance(risk_aversion)`: the best trade-off of expected active return
  against active variance.
- `MinimumVariance()`: the least risky portfolio. No signal.
- `MaximumDiversification()`: the highest ratio of its holdings' average
  volatility to its own. No signal.
- `RiskParity(budgets)`: each name contributing a set share of the risk,
  equal by default. No signal.
- `MaximumSharpe(base_return, alpha_per_score)`: the highest expected excess
  return per unit of risk.
- `BlackLitterman(...)`: the benchmark's implied returns, moved by the scores
  as views.
- `FactorTilts(factors, targets)`: target active exposures to factors, as
  close to the benchmark as they allow. Its own factors, no signal.
- `TopN(n, weighting)`: the top *n* names by score, equal or score weighted.

The benchmark-free and absolute-return constructions (minimum variance,
maximum diversification, risk parity, maximum Sharpe, Black-Litterman) are in
`beacon.strategy.allocations`. The benchmark-free ones ignore the benchmark's
weights, but the run is still measured against it.
"""
# BN-280 built MaxAlpha and MeanVariance into active.py; they moved here with
# BN-281, which added the rest (long-short is BN-306).
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..catalogue import CONSTRUCTION as CONSTRUCTION_KIND
from ..catalogue import COUNT, FRACTION, RATIO, Display, register
from ..catalogue import SIGNAL as SIGNAL_KIND
from ..exceptions import CalculationError
from ..optimise.constraints import Constraint
from ..optimise.solver import solve_constrained
from .base import StrategyContext
from .constraints import ActiveProblem
from .signals import Signal

logger = logging.getLogger(__name__)

EQUAL = "equal"
SIGNAL = "signal"
WEIGHTINGS = (EQUAL, SIGNAL)


class Construction(ABC):
    """How an active portfolio is built from scores, a benchmark and a
    covariance."""

    #: Whether it needs the strategy's signal.
    uses_scores = True

    @property
    def name(self) -> str:
        """How the run's record names this construction."""
        return type(self).__name__

    def inputs(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, dict[str, float]]:
        """Anything else it needs read for the rebalance, by name, which
        reaches it as `problem.inputs`. Nothing by default."""
        return {}

    def report(self,
               problem: ActiveProblem,
               weights: np.ndarray) -> dict[str, float]:
        """What it reports about the portfolio it built, beyond the weights.
        Nothing by default."""
        return {}

    @abstractmethod
    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        """The weights, aligned to the problem's names.

        Args:
            problem: The names, benchmark and covariance.
            alpha: Each name's score; 0 for a name that cannot be held.
            rules: The constraints, including long-only bounds and full
                investment.
            hint: A feasible place to start, invested as much as the
                portfolio is to be.
        """


@register(CONSTRUCTION_KIND, "Maximum alpha within a tracking-error budget", fields={
    "tracking_error": Display("Tracking-error budget", unit=FRACTION, minimum=0.0),
})
class MaxAlpha(Construction):
    """The most exposure to the scores within a tracking-error budget.

    Solved through its equivalent trade-off: maximising exposure within a
    budget has the same answer as maximising exposure less some risk aversion
    times the active variance. The risk aversion is found by bisection so the
    tracking error meets the budget, or falls short of it when the other
    constraints keep the portfolio closer to the benchmark anyway. Each step
    is a quadratic problem the optimiser solves reliably, where the linear
    objective against a quadratic limit did not.

    Args:
        tracking_error: The ex-ante annualised budget, as a decimal.

    Raises:
        ValueError: If *tracking_error* is not positive.
    """

    # The bisection's range of risk aversion, in powers of ten, and steps.
    LOWEST, HIGHEST, STEPS = -3.0, 7.0, 30

    def __init__(self,
                 tracking_error: float = 0.03):
        if tracking_error <= 0:
            raise ValueError(f"the tracking-error budget must be positive, got "
                             f"{tracking_error!r}.")

        self.tracking_error = tracking_error

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        boldest = self._within(problem, alpha, self.LOWEST, rules, hint)

        if boldest is not None:
            return boldest

        # The most cautious end must solve: it stays close to the benchmark,
        # and if it cannot, the constraints are what is wrong.
        low, high = self.LOWEST, self.HIGHEST
        best = traded_off(problem, alpha, 10.0 ** high, rules, hint)

        for _ in range(self.STEPS):
            middle = (low + high) / 2.0
            weights = self._within(problem, alpha, middle, rules, best)

            if weights is not None:
                high, best = middle, weights
            else:
                low = middle

        return best

    def _within(self,
                problem: ActiveProblem,
                alpha: np.ndarray,
                exponent: float,
                rules: list[Constraint],
                hint: np.ndarray) -> np.ndarray | None:
        """The trade-off at a risk aversion of ten to *exponent*, if it keeps
        within the budget. None when it does not, or when the solve fails:
        bold trade-offs against a kinked constraint such as turnover can stop
        just outside it, and a bolder step that fails is one too far."""
        try:
            weights = traded_off(problem, alpha, 10.0 ** exponent, rules, hint)
        except CalculationError:
            return None

        return weights if problem.tracking_error(weights) <= self.tracking_error else None

    def __repr__(self) -> str:
        return f"MaxAlpha(tracking_error={self.tracking_error!r})"


@register(CONSTRUCTION_KIND, "Mean-variance", fields={
    "risk_aversion": Display("Risk aversion", unit=RATIO, minimum=0.0),
    "alpha_per_score": Display("Expected return per score", unit=FRACTION, minimum=0.0),
})
class MeanVariance(Construction):
    """The best trade-off of expected active return against active variance.

    Expected active return is each name's score times *alpha_per_score*; the
    portfolio maximises it less *risk_aversion* times the active variance.

    Args:
        risk_aversion: How heavily active variance counts: higher holds
            closer to the benchmark.
        alpha_per_score: The expected annual return of one score, as a
            decimal.

    Raises:
        ValueError: If either is not positive.
    """

    def __init__(self,
                 risk_aversion: float = 10.0,
                 alpha_per_score: float = 0.02):
        if risk_aversion <= 0 or alpha_per_score <= 0:
            raise ValueError("MeanVariance needs a positive risk aversion and "
                             "alpha per score.")

        self.risk_aversion = risk_aversion
        self.alpha_per_score = alpha_per_score

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        return traded_off(problem, self.alpha_per_score * alpha,
                          self.risk_aversion, rules, hint)

    def __repr__(self) -> str:
        return (f"MeanVariance(risk_aversion={self.risk_aversion!r}, "
                f"alpha_per_score={self.alpha_per_score!r})")


@register(CONSTRUCTION_KIND, "Factor tilts", fields={
    "factors": Display("Factors", help="Each factor by name, as a signal whose scores "
                       "are the names' exposures."),
    "targets": Display("Target active exposures", help="Each factor's target active "
                       "exposure, in scores, such as {\"value\": 0.3}."),
    "strength": Display("How hard the targets pull", unit=RATIO, minimum=0.0),
})
class FactorTilts(Construction):
    """Target active exposures to factors, as close to the benchmark as they
    allow.

    Each factor is a signal: its standardised scores are the names'
    exposures to it, so a value factor is a value signal. The portfolio's
    active exposure to a factor is its active weights times the exposures.
    It minimises its active variance plus *strength* times the squared
    misses from the targets, so the targets are met as nearly as the
    constraints and the risk allow.

    Args:
        factors: Each factor by name, as a `Signal`.
        targets: Each factor's target active exposure, in scores; a factor
            with no target is aimed at zero.
        strength: How hard the targets pull against the tracking error.

    Raises:
        ValueError: If there are no factors, a target names no factor, or
            *strength* is not positive.
    """

    uses_scores = False
    # The shapes the editor renders these in (BN-330).
    PARAM_SCHEMAS: ClassVar[dict[str, str]] = {"factors": "FactorSignals",
                                                "targets": "FactorTargets"}
    PARAM_KINDS: ClassVar[dict[str, str]] = {"factors": SIGNAL_KIND}

    def __init__(self,
                 factors: Mapping[str, Signal],
                 targets: Mapping[str, float],
                 strength: float = 1.0):
        if not factors:
            raise ValueError("FactorTilts needs at least one factor.")

        unknown = sorted(set(targets) - set(factors))

        if unknown:
            raise ValueError(f"targets name no factor: {', '.join(unknown)}.")

        if strength <= 0:
            raise ValueError(f"strength must be positive, got {strength!r}.")

        self.factors = dict(factors)
        self.targets = dict(targets)
        self.strength = strength

    def inputs(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, dict[str, float]]:
        return {factor: signal.scores(names, date, context)
                for factor, signal in self.factors.items()}

    def exposures(self,
                  problem: ActiveProblem) -> np.ndarray:
        """The names' exposures, one column per factor."""
        return np.array([[problem.inputs.get(factor, {}).get(name, 0.0)
                          for factor in self.factors] for name in problem.assets])

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        loadings = self.exposures(problem)
        targets = np.array([self.targets.get(factor, 0.0) for factor in self.factors])
        size = average_variance(sigma)

        def objective(w: np.ndarray) -> float:
            active = problem.active(w)
            missed = loadings.T @ active - targets
            return float(active @ sigma @ active + self.strength * size * missed @ missed) / size

        def gradient(w: np.ndarray) -> np.ndarray:
            active = problem.active(w)
            missed = loadings.T @ active - targets
            return np.asarray(2.0 * sigma @ active
                              + 2.0 * self.strength * size * loadings @ missed) / size

        return solve(objective, gradient, rules, problem, hint)

    def report(self,
               problem: ActiveProblem,
               weights: np.ndarray) -> dict[str, float]:
        """The active exposure to each factor."""
        reached = self.exposures(problem).T @ problem.active(weights)

        return {factor: float(value) for factor, value in zip(self.factors, reached,
                                                               strict=True)}

    def __repr__(self) -> str:
        return (f"FactorTilts(factors={list(self.factors)!r}, targets={self.targets!r}, "
                f"strength={self.strength!r})")


@register(CONSTRUCTION_KIND, "Top names by score", fields={
    "n": Display("Names held", unit=COUNT, minimum=1),
    "weighting": Display("Weighting", choices=WEIGHTINGS),
})
class TopN(Construction):
    """The top *n* names by score, equal weighted or weighted by score: a
    simple rule the optimised constructions can be judged against.

    Names with a score of zero or below are weighted equally among
    themselves under ``"signal"`` when every chosen score is. The weights are
    then moved as little as the strategy's constraints require.

    Args:
        n: How many names to hold.
        weighting: ``"equal"`` or ``"signal"`` (in proportion to each
            name's score above the lowest chosen).

    Raises:
        ValueError: If *n* is below 1 or the weighting is not one of the two.
    """

    def __init__(self,
                 n: int = 20,
                 weighting: str = EQUAL):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n!r}.")

        if weighting not in WEIGHTINGS:
            raise ValueError(f"Unknown weighting {weighting!r}. Supported: "
                             f"{', '.join(WEIGHTINGS)}.")

        self.n = n
        self.weighting = weighting

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        holdable = set(problem.holdable) if problem.holdable else set(problem.assets)
        ranked = sorted((index for index, name in enumerate(problem.assets) if name in holdable),
                        key=lambda index: alpha[index], reverse=True)[:self.n]
        target = np.zeros(len(problem.assets))

        if self.weighting == SIGNAL:
            lifted = alpha[ranked] - alpha[ranked].min()
            target[ranked] = lifted if lifted.sum() > 0 else 1.0
        else:
            target[ranked] = 1.0

        target *= float(hint.sum()) / float(target.sum())

        # Moved as little as the constraints need: the rule's own weights
        # when nothing else binds.
        return solve(lambda w: float(np.sum((w - target) ** 2)),
                      lambda w: np.asarray(2.0 * (w - target)),
                      rules, problem, target)

    def __repr__(self) -> str:
        return f"TopN(n={self.n!r}, weighting={self.weighting!r})"


def traded_off(problem: ActiveProblem,
               expected: np.ndarray,
               aversion: float,
               rules: list[Constraint],
               hint: np.ndarray) -> np.ndarray:
    """The weights maximising *expected* active return less *aversion*
    times the active variance."""
    sigma = problem.covariance

    # Divided by a constant of its own size, which moves no optimum: at a
    # large risk aversion the raw objective is too big for SLSQP's line
    # search to make progress.
    scale = max(1.0, aversion * float(np.trace(sigma)) / max(len(problem.assets), 1))

    def objective(w: np.ndarray) -> float:
        active = problem.active(w)

        return float(-expected @ active + aversion * active @ sigma @ active) / scale

    def gradient(w: np.ndarray) -> np.ndarray:
        return np.asarray(-expected + 2.0 * aversion * sigma @ problem.active(w)) / scale

    return solve_constrained(objective, gradient, rules, problem.assets,
                             hint=hint).weights


def risk_shares(problem: ActiveProblem,
                weights: np.ndarray) -> dict[str, float]:
    """Each held name's share of the portfolio's variance: its weight times
    its covariance with the portfolio, over the variance."""
    sigma = problem.covariance
    variance = float(weights @ sigma @ weights)

    if variance <= 0.0:
        return {}

    shares = weights * (sigma @ weights) / variance

    return {name: float(share) for name, share, weight
            in zip(problem.assets, shares, weights, strict=True) if weight > 1e-9}


def solve(objective: Any,
           gradient: Any,
           rules: list[Constraint],
           problem: ActiveProblem,
           hint: np.ndarray) -> np.ndarray:
    """Minimise *objective* over the problem's names within *rules*."""
    return solve_constrained(objective, gradient, rules, problem.assets, hint=hint).weights


def average_variance(sigma: np.ndarray) -> float:
    """The covariance's average variance: a constant to divide an objective
    by, so its size suits the solver whatever the returns' scale."""
    return max(float(np.trace(sigma)) / max(len(sigma), 1), 1e-12)


def volatility(weights: np.ndarray,
                sigma: np.ndarray) -> float:
    """The portfolio's volatility, kept above zero."""
    return max(float(np.sqrt(max(float(weights @ sigma @ weights), 0.0))), 1e-12)


def inverse_volatility(problem: ActiveProblem,
                        hint: np.ndarray) -> np.ndarray:
    """A start in proportion to each holdable name's inverse volatility,
    invested as the hint is: closer to a risk-balanced answer than the
    benchmark is."""
    holdable = set(problem.holdable) if problem.holdable else set(problem.assets)
    inverse = np.array([1.0 / np.sqrt(variance) if name in holdable and variance > 0 else 0.0
                        for name, variance in zip(problem.assets, np.diag(problem.covariance),
                                                  strict=True)])

    if inverse.sum() <= 0.0:
        return hint

    return np.asarray(inverse * float(hint.sum()) / float(inverse.sum()))
