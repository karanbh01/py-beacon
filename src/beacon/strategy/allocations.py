# src/beacon/strategy/allocations.py
"""
Constructions that build a portfolio on its own terms rather than around the
benchmark: minimum variance, maximum diversification and risk parity, which
ignore it, and maximum Sharpe and Black-Litterman, which build from expected
returns. See `beacon.strategy.constructions` for the interface and the rest.
"""
# BN-281: split from constructions.py, which would otherwise run well past
# the size a module should be.
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from ..catalogue import CONSTRUCTION as CONSTRUCTION_KIND
from ..catalogue import FRACTION, RATIO, Display, register
from ..exceptions import CalculationError
from ..optimise.constraints import Constraint
from .constraints import ActiveProblem
from .constructions import (
    Construction,
    average_variance,
    inverse_volatility,
    solve,
    volatility,
)


@register(CONSTRUCTION_KIND, "Minimum variance")
class MinimumVariance(Construction):
    """The least risky portfolio: the lowest variance the constraints allow.
    Benchmark-free, and needs no signal."""

    uses_scores = False

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        size = average_variance(sigma)

        return solve(lambda w: float(w @ sigma @ w) / size,
                     lambda w: np.asarray(2.0 * sigma @ w) / size,
                     rules, problem, hint)

    def __repr__(self) -> str:
        return "MinimumVariance()"


@register(CONSTRUCTION_KIND, "Maximum diversification")
class MaximumDiversification(Construction):
    """The highest diversification ratio: the holdings' weighted average
    volatility over the portfolio's own. Benchmark-free, and needs no
    signal."""

    uses_scores = False

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        vols = np.sqrt(np.diag(sigma))

        def objective(w: np.ndarray) -> float:
            return -float(vols @ w) / volatility(w, sigma)

        def gradient(w: np.ndarray) -> np.ndarray:
            v = volatility(w, sigma)
            return np.asarray(-vols / v + float(vols @ w) * (sigma @ w) / v ** 3)

        return solve(objective, gradient, rules, problem, inverse_volatility(problem, hint))

    def __repr__(self) -> str:
        return "MaximumDiversification()"


@register(CONSTRUCTION_KIND, "Risk parity", fields={
    "budgets": Display("Risk budgets", help="Each name's share of the risk, such as "
                       "{\"AAA\": 0.2}; equal when left blank."),
})
class RiskParity(Construction):
    """Each name contributing a set share of the portfolio's risk: equal
    shares by default (equal risk contribution), or *budgets*. A name's
    contribution is its weight times its covariance with the portfolio, over
    the portfolio's variance. Benchmark-free, and needs no signal.

    Args:
        budgets: Each name's share of the risk, scaled to add up to one over
            the names that may be held; a name left out has none. None for
            equal shares.

    Raises:
        ValueError: If a budget is negative or none is positive.
    """

    uses_scores = False
    # The shapes the editor renders these in (BN-330).
    PARAM_SCHEMAS: ClassVar[dict[str, str]] = {"budgets": "IdentifierValues"}

    def __init__(self,
                 budgets: Mapping[str, float] | None = None):
        if budgets is not None:
            if any(budget < 0 for budget in budgets.values()):
                raise ValueError("risk budgets cannot be negative.")

            if not any(budget > 0 for budget in budgets.values()):
                raise ValueError("at least one risk budget must be positive.")

        self.budgets = dict(budgets) if budgets is not None else None

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        budget = self._targets(problem)

        def objective(w: np.ndarray) -> float:
            p = sigma @ w
            s = float(w @ p)
            return float(np.sum((w * p / s - budget) ** 2))

        def gradient(w: np.ndarray) -> np.ndarray:
            p = sigma @ w
            s = float(w @ p)
            e = w * p / s - budget
            return np.asarray(2.0 * (e * p / s + sigma @ (e * w) / s
                                     - 2.0 * p * float(e @ (w * p)) / s ** 2))

        return solve(objective, gradient, rules, problem, inverse_volatility(problem, hint))

    def _targets(self,
                 problem: ActiveProblem) -> np.ndarray:
        """Each name's share of the risk, over the names that may be held."""
        holdable = set(problem.holdable) if problem.holdable else set(problem.assets)
        raw = np.array([(self.budgets.get(name, 0.0) if self.budgets is not None else 1.0)
                        if name in holdable else 0.0 for name in problem.assets])
        total = float(raw.sum())

        if total <= 0.0:
            raise CalculationError("RiskParity", "none of the budgeted names may be held.")

        return raw / total

    def __repr__(self) -> str:
        return f"RiskParity(budgets={self.budgets!r})"


@register(CONSTRUCTION_KIND, "Maximum Sharpe ratio", fields={
    "base_return": Display("Expected excess return of an average name", unit=FRACTION),
    "alpha_per_score": Display("Expected return per score", unit=FRACTION, minimum=0.0),
})
class MaximumSharpe(Construction):
    """The highest expected excess return per unit of risk.

    Each name's expected excess return is *base_return* plus its score times
    *alpha_per_score*: the signal says which names should earn more than
    the average, and the base return how much an average name earns over
    cash.

    Args:
        base_return: The expected annual excess return of a name with an
            average score, as a decimal.
        alpha_per_score: The expected annual return of one score, as a
            decimal.

    Raises:
        ValueError: If *base_return* is not positive or *alpha_per_score*
            is negative.
    """

    def __init__(self,
                 base_return: float = 0.05,
                 alpha_per_score: float = 0.02):
        if base_return <= 0 or alpha_per_score < 0:
            raise ValueError("MaximumSharpe needs a positive base return and an "
                             "alpha per score of zero or more.")

        self.base_return = base_return
        self.alpha_per_score = alpha_per_score

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        expected = self.base_return + self.alpha_per_score * alpha

        def objective(w: np.ndarray) -> float:
            return -float(expected @ w) / volatility(w, sigma)

        def gradient(w: np.ndarray) -> np.ndarray:
            v = volatility(w, sigma)
            return np.asarray(-expected / v + float(expected @ w) * (sigma @ w) / v ** 3)

        return solve(objective, gradient, rules, problem, hint)

    def __repr__(self) -> str:
        return (f"MaximumSharpe(base_return={self.base_return!r}, "
                f"alpha_per_score={self.alpha_per_score!r})")


@register(CONSTRUCTION_KIND, "Black-Litterman", fields={
    "risk_aversion": Display("Risk aversion", unit=RATIO, minimum=0.0),
    "tau": Display("Uncertainty of the prior", unit=RATIO, minimum=0.0),
    "alpha_per_score": Display("View per score", unit=FRACTION, minimum=0.0),
    "confidence": Display("Confidence in the views", unit=RATIO, minimum=0.0),
})
class BlackLitterman(Construction):
    """The benchmark's implied returns, moved by the scores as views.

    The prior is the return each name must be expected to earn for the
    benchmark to be the best portfolio: ``risk_aversion x Σ x benchmark``.
    Each scored name is a view that it will earn its prior plus its score
    times *alpha_per_score*, held with an uncertainty in proportion to the
    prior's (``tau x Σ``) over *confidence*. The two are blended in the
    usual way, and the portfolio is the best trade-off of the blended returns
    against variance at the same risk aversion. With no scores it is the
    benchmark.

    Args:
        risk_aversion: The market's risk aversion, which sets the prior and
            the trade-off.
        tau: How uncertain the prior is, as a share of the covariance.
        alpha_per_score: How far one score moves a view, as a decimal a year.
        confidence: How much the views are believed against the prior:
            higher moves the blend further toward them.

    Raises:
        ValueError: If any setting is not positive.
    """

    def __init__(self,
                 risk_aversion: float = 2.5,
                 tau: float = 0.05,
                 alpha_per_score: float = 0.02,
                 confidence: float = 1.0):
        if min(risk_aversion, tau, alpha_per_score, confidence) <= 0:
            raise ValueError("BlackLitterman's settings must all be positive.")

        self.risk_aversion = risk_aversion
        self.tau = tau
        self.alpha_per_score = alpha_per_score
        self.confidence = confidence

    def blended(self,
                problem: ActiveProblem,
                alpha: np.ndarray) -> np.ndarray:
        """The blended expected returns, aligned to the problem's names."""
        sigma = problem.covariance
        total = float(problem.benchmark.sum())
        benchmark = problem.benchmark / total if total > 0 else problem.benchmark
        prior = self.risk_aversion * sigma @ benchmark
        viewed = np.flatnonzero(alpha)

        if not len(viewed):
            return np.asarray(prior)

        picks = np.eye(len(alpha))[viewed]
        views = prior[viewed] + self.alpha_per_score * alpha[viewed]
        prior_uncertainty = self.tau * sigma
        omega = np.diag(np.diag(picks @ prior_uncertainty @ picks.T)) / self.confidence
        middle = picks @ prior_uncertainty @ picks.T + omega

        return np.asarray(prior + prior_uncertainty @ picks.T
                          @ np.linalg.solve(middle, views - picks @ prior))

    def build(self,
              problem: ActiveProblem,
              alpha: np.ndarray,
              rules: list[Constraint],
              hint: np.ndarray) -> np.ndarray:
        sigma = problem.covariance
        expected = self.blended(problem, alpha)
        half = self.risk_aversion / 2.0
        scale = max(1.0, half * average_variance(sigma))

        return solve(lambda w: float(-expected @ w + half * w @ sigma @ w) / scale,
                      lambda w: np.asarray(-expected + 2.0 * half * sigma @ w) / scale,
                      rules, problem, hint)

    def __repr__(self) -> str:
        return (f"BlackLitterman(risk_aversion={self.risk_aversion!r}, tau={self.tau!r}, "
                f"alpha_per_score={self.alpha_per_score!r}, "
                f"confidence={self.confidence!r})")
