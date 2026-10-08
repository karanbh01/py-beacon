# src/beacon/strategy/constraints.py
"""
What an active portfolio must satisfy, mostly measured against its
benchmark.

    ActiveStrategy(..., constraints=[ActiveShare(minimum=0.3),
                                     RelativeSectorBounds(within=0.05),
                                     HoldingsLimit(60)])

- `TrackingErrorBudget(maximum)`: an ex-ante tracking error to the
  benchmark of at most *maximum* a year.
- `ActiveShare(minimum, maximum)`: an active share in the range.
- `RelativeSectorBounds(within)`: each sector within *within* of the
  benchmark's weight in it.
- `RelativePositionBounds(within)`: each name within *within* of its
  benchmark weight.
- `HoldingsLimit(maximum)`: no more than *maximum* names.
- `TurnoverLimit(maximum)`: at most *maximum* traded one way from the last
  rebalance's weights.

Active share is half the summed absolute differences from the benchmark's
weights. Every portfolio is also long-only and fully invested.
"""
# BN-280, phase 7 of decisions/0006. Each constraint becomes optimiser
# conditions at a rebalance, once the benchmark, covariance and sectors there
# are known; the optimiser's own constraints are absolute and serialisable,
# these are neither.
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..optimise.constraints import (
    INEQUALITY,
    Cardinality,
    Condition,
    Constraint,
    TurnoverBudget,
)


@dataclass(frozen=True)
class ActiveProblem:
    """One rebalance's problem, aligned to its names.

    Attributes:
        assets: The names the solve allocates over, in order.
        benchmark: Their benchmark weights.
        covariance: Their annualised covariance.
        sectors: Each name's sector, for sector bounds.
        previous: The last rebalance's weights, for a turnover limit.
    """
    assets: list[str]
    benchmark: np.ndarray
    covariance: np.ndarray
    sectors: dict[str, str] = field(default_factory=dict)
    previous: dict[str, float] = field(default_factory=dict)

    def active(self,
               weights: np.ndarray) -> np.ndarray:
        """Weights less the benchmark's."""
        return np.asarray(weights - self.benchmark, dtype=float)

    def tracking_error(self,
                       weights: np.ndarray) -> float:
        """Ex-ante annualised tracking error of *weights*."""
        active = self.active(weights)

        return float(np.sqrt(max(float(active @ self.covariance @ active), 0.0)))

    def active_share(self,
                     weights: np.ndarray) -> float:
        """Half the summed absolute differences from the benchmark."""
        return float(np.abs(self.active(weights)).sum() / 2.0)


class ActiveConstraint(ABC):
    """A constraint on an active portfolio."""

    @abstractmethod
    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        """The optimiser constraints this becomes for *problem*."""


class TrackingErrorBudget(ActiveConstraint):
    """An ex-ante tracking error to the benchmark of at most *maximum*.

    Args:
        maximum: The annualised limit, as a decimal: 0.03 for 3%.

    Raises:
        ValueError: If *maximum* is not positive.
    """

    def __init__(self,
                 maximum: float):
        if maximum <= 0:
            raise ValueError(f"the tracking-error budget must be positive, got "
                             f"{maximum!r}.")

        self.maximum = maximum

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        # As a share of the budget's variance, so the condition is of order
        # one like the objectives it sits beside: stated in raw variance
        # (around 1e-3) it left SLSQP's line search unable to make progress.
        limit = self.maximum ** 2
        sigma = problem.covariance

        def room(w: np.ndarray) -> float:
            active = problem.active(w)

            return float(1.0 - active @ sigma @ active / limit)

        return [_Conditions([Condition(
            label=f"tracking error at most {self.maximum:.2%}", kind=INEQUALITY,
            evaluate=room,
            gradient=lambda w: -2.0 * sigma @ problem.active(w) / limit)])]


class ActiveShare(ActiveConstraint):
    """An active share in a range.

    Args:
        minimum: The least, as a decimal, or None.
        maximum: The most, as a decimal, or None.

    Raises:
        ValueError: If neither is given, or they are outside 0 to 1 or cross.
    """

    def __init__(self,
                 minimum: float | None = None,
                 maximum: float | None = None):
        if minimum is None and maximum is None:
            raise ValueError("ActiveShare needs a minimum, a maximum or both.")

        low = 0.0 if minimum is None else minimum
        high = 1.0 if maximum is None else maximum

        if not 0.0 <= low <= high <= 1.0:
            raise ValueError("ActiveShare's range must lie within 0 to 1.")

        self.minimum = minimum
        self.maximum = maximum

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        # The absolute value has kinks; the sign vector is a subgradient, as
        # the optimiser's own turnover budget uses.
        def slope(w: np.ndarray) -> np.ndarray:
            return np.asarray(np.sign(problem.active(w)) / 2.0, dtype=float)

        conditions = []

        if self.minimum is not None:
            low = self.minimum
            conditions.append(Condition(
                label=f"active share at least {low:.2%}", kind=INEQUALITY,
                evaluate=lambda w: problem.active_share(w) - low, gradient=slope))

        if self.maximum is not None:
            high = self.maximum
            conditions.append(Condition(
                label=f"active share at most {high:.2%}", kind=INEQUALITY,
                evaluate=lambda w: high - problem.active_share(w),
                gradient=lambda w: -slope(w)))

        return [_Conditions(conditions)]


class RelativeSectorBounds(ActiveConstraint):
    """Each sector's weight within *within* of the benchmark's.

    Args:
        within: The largest difference, as a decimal: 0.05 for 5 points.
        scheme: The classification sectors are read from.

    Raises:
        ValueError: If *within* is negative.
    """

    def __init__(self,
                 within: float,
                 scheme: str = "SECTOR"):
        if within < 0:
            raise ValueError(f"within cannot be negative, got {within!r}.")

        self.within = within
        self.scheme = scheme

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        conditions = []

        for sector in sorted(set(problem.sectors.values())):
            members = np.array([problem.sectors.get(name) == sector
                                for name in problem.assets], dtype=float)
            conditions += _band(members, float(members @ problem.benchmark),
                                self.within, f"{sector} weight")

        return [_Conditions(conditions)]


class RelativePositionBounds(ActiveConstraint):
    """Each name within *within* of its benchmark weight.

    Args:
        within: The largest difference, as a decimal.

    Raises:
        ValueError: If *within* is negative.
    """

    def __init__(self,
                 within: float):
        if within < 0:
            raise ValueError(f"within cannot be negative, got {within!r}.")

        self.within = within

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        box = [(max(weight - self.within, 0.0), min(weight + self.within, 1.0))
               for weight in problem.benchmark]

        return [_Conditions([], box)]


class HoldingsLimit(ActiveConstraint):
    """No more than *maximum* names held.

    Args:
        maximum: The most names.
    """

    def __init__(self,
                 maximum: int):
        if maximum < 1:
            raise ValueError(f"maximum must be at least 1, got {maximum!r}.")

        self.maximum = maximum

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        return [Cardinality(self.maximum)]


class TurnoverLimit(ActiveConstraint):
    """At most *maximum* traded one way from the last rebalance's weights.
    The first rebalance, with nothing held, is not limited.

    Args:
        maximum: The one-way limit, as a decimal.
    """

    def __init__(self,
                 maximum: float):
        if maximum < 0:
            raise ValueError(f"maximum cannot be negative, got {maximum!r}.")

        self.maximum = maximum

    def build(self,
              problem: ActiveProblem) -> list[Constraint]:
        if not problem.previous:
            return []

        return [TurnoverBudget(self.maximum, problem.previous)]


class _Conditions(Constraint):
    """Conditions and a box already worked out for one problem."""

    def __init__(self,
                 conditions: list[Condition],
                 box: list[tuple[float, float]] | None = None):
        self._conditions = conditions
        self._box = box

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        return list(self._conditions)

    def bounds(self,
               assets: Sequence[str]) -> list[tuple[float, float]] | None:
        return self._box


def _band(members: np.ndarray,
          benchmark: float,
          within: float,
          what: str) -> list[Condition]:
    """A group's weight within *within* of *benchmark*, as two inequalities."""
    return [
        Condition(label=f"{what} at most {within:.2%} over the benchmark",
                  kind=INEQUALITY,
                  evaluate=lambda w: benchmark + within - float(members @ w),
                  gradient=lambda w: -members),
        Condition(label=f"{what} at most {within:.2%} under the benchmark",
                  kind=INEQUALITY,
                  evaluate=lambda w: float(members @ w) - benchmark + within,
                  gradient=lambda w: members),
    ]
