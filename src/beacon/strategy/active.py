# src/beacon/strategy/active.py
"""
Active strategies: a universe, a signal and a way of building a portfolio
from it, measured against a benchmark.

    strategy = ActiveStrategy(
        benchmark=my_index,
        signal=FieldSignal(data.features.fundamentals.earnings_yield),
        construction=MaxAlpha(tracking_error=0.03),
        constraints=[RelativeSectorBounds(within=0.05), HoldingsLimit(60)])

    Backtest(initial_capital=1e8).run(strategy, start="2023-01-03", end="2024-12-31")

At each rebalance the strategy:

1. Takes the **benchmark's** weights in force that day. The benchmark is an
   index, calculated as any index is, and the run is measured against it.
2. Picks its **candidates**: the `universe` (the benchmark's constituents by
   default), less any failing the `screen`, a condition such as
   ``data.market.market_cap > 1e9``. A benchmark name that is not a
   candidate cannot be held.
3. Scores the candidates with its **signal** (see `beacon.strategy.signals`).
4. Estimates the **covariance** from a year of returns, shrunk toward constant
   correlation. A candidate with too little history is held at its benchmark
   weight, outside the solve.
5. **Constructs** the long-only, fully invested portfolio, within its
   constraints (see `beacon.strategy.constraints`):

   - `MaxAlpha(tracking_error)`: the most exposure to the scores that a
     tracking-error budget allows.
   - `MeanVariance(risk_aversion)`: the best trade-off of expected active
     return (the scores times `alpha_per_score`) against active variance
     times `risk_aversion`; the tracking error is whatever results.

The constructions share one interface, `Construction`, so others can be added.

Rebalances fall on the first session of each period (``"MONTHLY"`` by
default). Each records the weights, the benchmark's, the ex-ante tracking
error and the active share in `result.active`.
"""
# BN-280, phase 7 of decisions/0006. Karan asked for flexibility: both
# constructions now, and more (absolute return among them) in BN-281.
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..exceptions import CalculationError
from ..expressions.core import Expression
from ..expressions.resolve import resolve
from ..index.derived import AnyIndexDefinition
from ..index.result import IndexResult
from ..index.schedule import sessions
from ..optimise.constraints import Constraint, FullInvestment, PositionBounds
from ..optimise.solver import solve_constrained
from .base import (
    DEFAULT_LOOKBACK_DAYS,
    MINIMUM_OBSERVATIONS,
    StrategyContext,
    covariance_of,
    estimated_risk,
    measured,
    trailing_returns,
)
from .constraints import (
    ActiveConstraint,
    ActiveProblem,
    RelativeSectorBounds,
)
from .signals import Signal

logger = logging.getLogger(__name__)

FREQUENCIES = ("WEEKLY", "MONTHLY", "QUARTERLY", "ANNUALLY")


@dataclass(frozen=True)
class ActiveStep:
    """What an active strategy did at one rebalance.

    Attributes:
        date: The rebalance date.
        weights: The weights the portfolio traded to.
        benchmark: The benchmark's weights that day.
        tracking_error: The ex-ante annualised tracking error to the
            benchmark, from the day's covariance.
        active_share: Half the summed absolute differences from the
            benchmark.
        scores: Each candidate's signal score.
        unmeasured: Candidates held at their benchmark weight because their
            history was too short to measure.
    """
    date: pd.Timestamp
    weights: dict[str, float]
    benchmark: dict[str, float]
    tracking_error: float
    active_share: float
    scores: dict[str, float] = field(default_factory=dict)
    unmeasured: tuple[str, ...] = field(default_factory=tuple)


class Construction(ABC):
    """How an active portfolio is built from scores, a benchmark and a
    covariance."""

    @property
    def name(self) -> str:
        """How the run's record names this construction."""
        return type(self).__name__

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
            hint: A feasible place to start.
        """


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
        best = _traded_off(problem, alpha, 10.0 ** high, rules, hint)

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
            weights = _traded_off(problem, alpha, 10.0 ** exponent, rules, hint)
        except CalculationError:
            return None

        return weights if problem.tracking_error(weights) <= self.tracking_error else None

    def __repr__(self) -> str:
        return f"MaxAlpha(tracking_error={self.tracking_error!r})"


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
        return _traded_off(problem, self.alpha_per_score * alpha,
                           self.risk_aversion, rules, hint)

    def __repr__(self) -> str:
        return (f"MeanVariance(risk_aversion={self.risk_aversion!r}, "
                f"alpha_per_score={self.alpha_per_score!r})")


class ActiveStrategy:
    """A strategy that builds its own portfolio from a signal, against a
    benchmark.

    Args:
        benchmark: The index it is measured against.
        signal: What it believes about each name.
        construction: How it builds the portfolio; `MaxAlpha()` by default.
        constraints: What the portfolio must also satisfy.
        universe: The names it may hold; the benchmark's constituents when
            None.
        screen: A condition every holding must pass, such as
            ``data.market.market_cap > 1e9``.
        rebalancing: ``"WEEKLY"``, ``"MONTHLY"`` (the default),
            ``"QUARTERLY"`` or ``"ANNUALLY"``.
        lookback_days: Trading days of returns the covariance is estimated
            over.
        minimum_observations: The fewest returns a name needs to be measured.
        name: What the strategy is called.

    Raises:
        ValueError: If *rebalancing* is not one of the four.
    """

    def __init__(self,
                 benchmark: AnyIndexDefinition,
                 signal: Signal,
                 construction: Construction | None = None,
                 constraints: Sequence[ActiveConstraint] = (),
                 universe: Sequence[str] | None = None,
                 screen: Expression | None = None,
                 rebalancing: str = "MONTHLY",
                 lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                 minimum_observations: int = MINIMUM_OBSERVATIONS,
                 name: str = "Active strategy"):
        if rebalancing not in FREQUENCIES:
            raise ValueError(f"Unknown rebalancing {rebalancing!r}. Supported: "
                             f"{', '.join(FREQUENCIES)}.")

        self.benchmark = benchmark
        self.signal = signal
        self.construction: Construction = construction or MaxAlpha()
        self.constraints: tuple[ActiveConstraint, ...] = tuple(constraints)
        self.universe = list(universe) if universe is not None else None
        self.screen = screen
        self.rebalancing = rebalancing
        self.lookback_days = lookback_days
        self.minimum_observations = minimum_observations
        self.name = name

    @property
    def index(self) -> AnyIndexDefinition:
        """The index the run calculates and measures against."""
        return self.benchmark

    def rebalance_dates(self,
                        start: str,
                        end: str) -> list[pd.Timestamp]:
        """The first session of each period from *start* to *end*."""
        days = sessions(pd.Timestamp(start), pd.Timestamp(end), self.benchmark.calendar)

        return [day for position, day in enumerate(days)
                if position == 0 or _period(day, self.rebalancing)
                != _period(days[position - 1], self.rebalancing)]

    def steps(self,
              benchmark: IndexResult,
              start: str,
              end: str,
              context: StrategyContext) -> list[ActiveStep]:
        """The portfolio at each rebalance from *start* to *end*."""
        built: list[ActiveStep] = []

        for date in self.rebalance_dates(start, end):
            weights = _in_force(benchmark.weight_snapshots, date)

            if not weights:
                continue

            previous = built[-1].weights if built else {}
            built.append(self.rebalanced(date, weights, previous, context))

        return built

    def rebalanced(self,
                   date: pd.Timestamp,
                   benchmark: dict[str, float],
                   previous: dict[str, float],
                   context: StrategyContext) -> ActiveStep:
        """One rebalance: the portfolio for *benchmark* on *date*."""
        candidates = self._candidates(benchmark, date, context)
        names = sorted(set(candidates) | set(benchmark))
        returns = trailing_returns(names, date, context, self.lookback_days)
        known = measured(names, returns, self.minimum_observations)
        fixed = {name: benchmark.get(name, 0.0) for name in candidates
                 if name not in known and benchmark.get(name, 0.0) > 0.0}
        scores = self.signal.scores(candidates, date, context)

        if len(known) < 2:
            logger.warning("[%s] Too few names with enough history; the "
                           "benchmark is held.", date.date())

            return ActiveStep(date=date, weights=dict(benchmark), benchmark=benchmark,
                              tracking_error=0.0, active_share=0.0, scores=scores)

        problem = ActiveProblem(
            assets=known,
            benchmark=np.array([benchmark.get(name, 0.0) for name in known]),
            covariance=covariance_of(estimated_risk(returns, known), known),
            sectors=_sectors(known, date, context, self.constraints),
            previous=previous)
        holdable = set(candidates)
        alpha = np.array([scores.get(name, 0.0) if name in holdable else 0.0
                          for name in known])
        invested = 1.0 - sum(fixed.values())
        rules = self._rules(problem, holdable, invested)
        hint = _start(problem, holdable, invested, previous)

        solved = self.construction.build(problem, alpha, rules, hint)
        weights = {**{name: float(weight) for name, weight in zip(known, solved, strict=True)
                      if weight > 1e-9},
                   **fixed}

        return ActiveStep(date=date, weights=weights, benchmark=dict(benchmark),
                          tracking_error=problem.tracking_error(solved),
                          active_share=_active_share(weights, benchmark),
                          scores=scores, unmeasured=tuple(sorted(fixed)))

    def _candidates(self,
                    benchmark: dict[str, float],
                    date: pd.Timestamp,
                    context: StrategyContext) -> list[str]:
        """The names that may be held on *date*."""
        names = self.universe if self.universe is not None else list(benchmark)

        if self.screen is None:
            return list(names)

        return [name for name in names
                if resolve(self.screen, name, date, context.fetcher,
                           currency=context.currency)]

    def _rules(self,
               problem: ActiveProblem,
               holdable: set[str],
               invested: float) -> list[Constraint]:
        """Long-only, fully invested, nothing in a name that may not be held,
        and the strategy's own constraints."""
        barred = [name for name in problem.assets if name not in holdable]
        rules: list[Constraint] = [FullInvestment(invested),
                                   PositionBounds(minimum=0.0, maximum=1.0)]

        if barred:
            rules.append(PositionBounds(minimum=0.0, maximum=0.0, assets=barred))

        for constraint in self.constraints:
            rules += constraint.build(problem)

        return rules

    def __repr__(self) -> str:
        return (f"ActiveStrategy(name={self.name!r}, benchmark="
                f"{self.benchmark.index_id!r}, signal={self.signal.name}, "
                f"construction={self.construction!r})")


def _traded_off(problem: ActiveProblem,
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


def _in_force(snapshots: dict[pd.Timestamp, dict[str, float]],
              date: pd.Timestamp) -> dict[str, float]:
    """The benchmark's weights in force on *date*: its latest snapshot on or
    before it."""
    dates = [snapshot for snapshot in snapshots if snapshot <= date]

    return dict(snapshots[max(dates)]) if dates else {}


def _start(problem: ActiveProblem,
           holdable: set[str],
           invested: float,
           previous: dict[str, float]) -> np.ndarray:
    """Where the solve starts: the last rebalance's weights, or the benchmark
    at the first, held only where it may be and scaled to what is invested.
    Starting from the last weights keeps a turnover limit within reach."""
    source = (previous if previous
              else dict(zip(problem.assets, problem.benchmark, strict=True)))
    weights = np.array([source.get(name, 0.0) if name in holdable else 0.0
                        for name in problem.assets])
    total = float(weights.sum())

    if total <= 0.0:
        held = np.array([name in holdable for name in problem.assets], dtype=float)

        return held * invested / max(float(held.sum()), 1.0)

    return np.asarray(weights * invested / total)


def _sectors(names: list[str],
             date: pd.Timestamp,
             context: StrategyContext,
             constraints: tuple[ActiveConstraint, ...]) -> dict[str, str]:
    """Each name's sector, read only when a constraint needs it."""
    schemes = sorted({str(constraint.scheme) for constraint in constraints
                      if isinstance(constraint, RelativeSectorBounds)})

    if not schemes:
        return {}

    scheme = schemes[0]
    found = context.fetcher.fetch_classifications(names, date, scheme)

    return {name: sector or "Unclassified" for name, sector in found.items()}


def _active_share(weights: dict[str, float],
                  benchmark: dict[str, float]) -> float:
    """Half the summed absolute differences, over every name in either."""
    names = set(weights) | set(benchmark)

    return sum(abs(weights.get(name, 0.0) - benchmark.get(name, 0.0))
               for name in names) / 2.0


def _period(date: pd.Timestamp,
            frequency: str) -> tuple[int, ...]:
    """The period *date* falls in, at *frequency*."""
    if frequency == "WEEKLY":
        iso = date.isocalendar()

        return (iso.year, iso.week)

    if frequency == "MONTHLY":
        return (date.year, date.month)

    if frequency == "QUARTERLY":
        return (date.year, (date.month - 1) // 3)

    return (date.year,)
