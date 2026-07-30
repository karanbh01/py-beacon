# src/beacon/optimise/constraints.py
"""
Constraints an optimisation must respect.

Each class maps to one row a user adds in a constraint editor — position
bounds, sector bounds, a turnover budget, a holding count, full investment — so
the client can build a problem without translating between two vocabularies.

Every constraint states itself once, as :class:`Condition` objects in scipy's
own convention: an equality holds when its function is zero, an inequality when
its function is non-negative. That single statement is then used three ways —
it is handed to the solver, it decides which constraints came out binding, and
it verifies the returned weights actually satisfy what was asked. The three can
therefore never drift apart, which is the point: the verification pass exists
to catch a solver that returned something it should not have, and it would be
worthless if it checked a second, separately-written copy of the rules.
"""
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

Vector = NDArray[np.float64]

# scipy's own constraint kinds, used verbatim so a Condition can be handed to
# the solver without translation.
EQUALITY = "eq"
INEQUALITY = "ineq"

# Within this distance of its boundary a constraint counts as binding. Weights
# are fractions of a portfolio, so a millionth is far below any allocation a
# caller would act on, and the solver's own convergence leaves residuals a few
# orders of magnitude smaller than this.
BINDING_TOLERANCE = 1e-6

# Beyond this a constraint counts as violated and the answer is refused. Held
# equal to the binding tolerance deliberately: a constraint that is not
# satisfied to the accuracy at which we would call it binding is one we cannot
# honestly report on either way.
FEASIBILITY_TOLERANCE = 1e-6

# Below this weight a name is not held. An equal-weighted 500-name index puts
# 0.2% in each, so this sits four orders of magnitude beneath the smallest
# position any real universe produces.
HOLDING_THRESHOLD = 1e-6


@dataclass(frozen=True)
class Condition:
    """One scalar condition on a weight vector.

    Attributes:
        label: Human-readable description, carried through to the binding
            report so a client can show which rule bit.
        kind: EQUALITY or INEQUALITY.
        evaluate: The condition function. Zero when an equality holds,
            non-negative when an inequality holds.
        gradient: Derivative with respect to the weights, when one can be
            written down. Supplying it keeps the solver off finite differences,
            which on a problem this small is most of the accuracy.
    """
    label: str
    kind: str
    evaluate: Callable[[Vector], float]
    gradient: Callable[[Vector], Vector] | None = None


@dataclass(frozen=True)
class Slack:
    """How much room a condition has left at a given solution.

    Attributes:
        label: The condition's description.
        kind: EQUALITY or INEQUALITY.
        slack: Signed room. For an inequality this is the condition function
            itself, so zero means the solution sits exactly on the boundary and
            negative means it has crossed. For an equality it is the negated
            absolute residual, which makes an exactly-satisfied equality read
            as zero slack — correctly, since an equality is always binding.
    """
    label: str
    kind: str
    slack: float

    @property
    def is_binding(self) -> bool:
        """Whether the solution sits on this condition's boundary."""
        return abs(self.slack) <= BINDING_TOLERANCE

    @property
    def is_violated(self) -> bool:
        """Whether the solution breaks this condition."""
        return self.slack < -FEASIBILITY_TOLERANCE


class Constraint(ABC):
    """Something an optimal weight vector must satisfy.

    Subclasses state their rules as conditions, and everything else — solving,
    binding detection, verification — is derived from those.
    """

    @abstractmethod
    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        """The conditions this constraint imposes over *assets*, in order.

        Args:
            assets: The universe, fixing the meaning of each weight position.

        Returns:
            list: Conditions in scipy's convention.
        """

    def solver_conditions(self,
                          assets: Sequence[str]) -> list[Condition]:
        """The subset of the conditions that should go to the solver.

        Everything by default. A constraint overrides this to nothing when it
        reaches the solver some other way — as a box, or by restricting the
        problem — while still reporting its conditions for verification.
        """
        return self.conditions(assets)

    def bounds(self,
               assets: Sequence[str]) -> list[tuple[float, float]] | None:
        """Per-asset box limits, when this constraint is one.

        A box handed to the solver as a bound is enforced at every iterate,
        whereas the same box as an inequality is only approached from outside;
        for position limits that difference is worth the special case.

        Returns:
            list or None: One ``(low, high)`` per asset, or None if this
            constraint is not a box.
        """
        return None

    def report(self,
               weights: Vector,
               assets: Sequence[str]) -> list[Slack]:
        """How much room this constraint has left at *weights*.

        Args:
            weights: A candidate solution, aligned to *assets*.
            assets: The universe.

        Returns:
            list: One Slack per condition.
        """
        return [_slack_of(condition, weights)
                for condition in self.conditions(assets)]

    def validate(self,  # noqa: B027 — an optional hook, not part of the interface
                 assets: Sequence[str]) -> None:
        """Raise if this constraint cannot be applied to *assets* at all.

        Separate from feasibility: this catches a constraint that is malformed
        or refers to names that are not there, which is a caller mistake rather
        than an over-tight problem. Deliberately concrete and empty rather than
        abstract — most constraints have nothing to check, and forcing them all
        to write an empty override would hide the two that do.
        """


def _slack_of(condition: Condition,
              weights: Vector) -> Slack:
    """Measure one condition at a solution."""
    value = float(condition.evaluate(weights))

    return Slack(label=condition.label,
                 kind=condition.kind,
                 slack=-abs(value) if condition.kind == EQUALITY else value)


def _positions(assets: Sequence[str],
               wanted: Iterable[str]) -> list[int]:
    """Index positions of *wanted* within *assets*, in universe order."""
    lookup = {asset_id: position for position, asset_id in enumerate(assets)}

    return sorted(lookup[asset_id] for asset_id in wanted if asset_id in lookup)


class FullInvestment(Constraint):
    """The weights must sum to a fixed total, normally one.

    Attributes:
        target: The total. One means fully invested with no leverage and no
            cash; below one leaves cash, above one is levered.
    """

    def __init__(self,
                 target: float = 1.0):
        self.target = float(target)

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        count = len(assets)
        target = self.target

        return [Condition(label=f"full investment at {target:.4%}",
                          kind=EQUALITY,
                          evaluate=lambda w: float(w.sum()) - target,
                          gradient=lambda w: np.ones(count))]


class PositionBounds(Constraint):
    """Lower and upper limits on individual weights.

    Attributes:
        minimum: Smallest weight any covered asset may take. Zero forbids
            short positions, which is the usual index-tracking case.
        maximum: Largest weight any covered asset may take.
        assets: Names this applies to, or None for every name. Several of these
            compose — a blanket rule plus a tighter one on a few names — and
            the tightest limit on each name wins.
    """

    def __init__(self,
                 minimum: float = 0.0,
                 maximum: float = 1.0,
                 assets: Sequence[str] | None = None):
        if minimum > maximum:
            raise ValueError(
                f"minimum weight {minimum} exceeds maximum weight {maximum}.")

        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.assets = None if assets is None else tuple(assets)

    def validate(self,
                 assets: Sequence[str]) -> None:
        """Reject a bound on a name that is not in the universe.

        Unlike a group, a position bound names one asset explicitly, so a name
        that is not there is a typo rather than a broader definition being
        reused — and silently dropping it would return an answer that ignored
        a limit the caller asked for.
        """
        if self.assets is None:
            return

        unknown = sorted(set(self.assets) - set(assets))
        if unknown:
            raise CalculationError(
                "Optimiser",
                f"position bounds name assets outside the universe: {unknown}.")

    def _covered(self,
                 assets: Sequence[str]) -> list[int]:
        """Positions this bound applies to."""
        if self.assets is None:
            return list(range(len(assets)))

        return _positions(assets, self.assets)

    def bounds(self,
               assets: Sequence[str]) -> list[tuple[float, float]]:
        """The box, unlimited on any asset this constraint does not cover."""
        box = [(-np.inf, np.inf)] * len(assets)

        for position in self._covered(assets):
            box[position] = (self.minimum, self.maximum)

        return box

    def solver_conditions(self,
                          assets: Sequence[str]) -> list[Condition]:
        """None: the solver receives this constraint as a box instead."""
        return []

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        """The same box, written as inequalities.

        Not given to the solver — :meth:`bounds` covers that — but used to
        report which positions came out at a limit.
        """
        conditions = []

        for position in self._covered(assets):
            asset_id = assets[position]
            conditions.append(_floor_condition(position, asset_id, self.minimum))
            conditions.append(_cap_condition(position, asset_id, self.maximum))

        return conditions


def _floor_condition(position: int,
                     asset_id: str,
                     minimum: float) -> Condition:
    """One asset's lower limit, as an inequality."""
    def evaluate(weights: Vector) -> float:
        return float(weights[position]) - minimum

    return Condition(label=f"minimum weight {minimum:.4%} on {asset_id}",
                     kind=INEQUALITY,
                     evaluate=evaluate)


def _cap_condition(position: int,
                   asset_id: str,
                   maximum: float) -> Condition:
    """One asset's upper limit, as an inequality."""
    def evaluate(weights: Vector) -> float:
        return maximum - float(weights[position])

    return Condition(label=f"maximum weight {maximum:.4%} on {asset_id}",
                     kind=INEQUALITY,
                     evaluate=evaluate)


class GroupBounds(Constraint):
    """Limits on the combined weight of a set of names.

    A sector, a country, a liquidity bucket — anything the client groups by.

    Attributes:
        name: What the group is, used in the binding report.
        members: The names in it.
        minimum: Smallest combined weight the group may take.
        maximum: Largest combined weight the group may take.
    """

    def __init__(self,
                 name: str,
                 members: Sequence[str],
                 minimum: float = 0.0,
                 maximum: float = 1.0):
        if minimum > maximum:
            raise ValueError(
                f"group '{name}' has minimum {minimum} above maximum {maximum}.")

        self.name = name
        self.members = tuple(members)
        self.minimum = float(minimum)
        self.maximum = float(maximum)

    def validate(self,
                 assets: Sequence[str]) -> None:
        """Reject a group with no members in the universe.

        Members outside the universe are dropped rather than rejected: a sector
        map is defined over a whole market and is meant to be reused across
        indices, so naming companies this index does not hold is normal. A
        group that matches *nothing*, though, is a mistake — it would silently
        constrain an empty sum, which is always satisfied.
        """
        present = _positions(assets, self.members)
        if not present:
            raise CalculationError(
                "Optimiser",
                f"group '{self.name}' has no members in the universe.")

        dropped = len(self.members) - len(present)
        if dropped:
            logger.debug(
                f"Group '{self.name}': {dropped} member(s) are outside the "
                f"universe and are ignored.")

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        positions = _positions(assets, self.members)
        selector = np.zeros(len(assets))
        selector[positions] = 1.0

        return [
            Condition(label=f"minimum {self.minimum:.4%} in group '{self.name}'",
                      kind=INEQUALITY,
                      evaluate=lambda w: float(selector @ w) - self.minimum,
                      gradient=lambda w: selector),
            Condition(label=f"maximum {self.maximum:.4%} in group '{self.name}'",
                      kind=INEQUALITY,
                      evaluate=lambda w: self.maximum - float(selector @ w),
                      gradient=lambda w: -selector),
        ]


class TurnoverBudget(Constraint):
    """A limit on how far the solution may move from the current holdings.

    Turnover here is **one-way**: half the sum of absolute weight changes,
    which under full investment is the amount bought, and equally the amount
    sold. A budget of 5% therefore means what an index methodology means by
    "5% turnover", not a 2.5% round trip.

    Attributes:
        maximum: The one-way budget, as a fraction of the portfolio.
        current_weights: Where the portfolio is now. Names absent from it are
            treated as currently unheld.
    """

    def __init__(self,
                 maximum: float,
                 current_weights: dict[str, float]):
        if maximum < 0.0:
            raise ValueError(f"turnover budget must be non-negative, got {maximum}.")

        self.maximum = float(maximum)
        self.current_weights = dict(current_weights)

    def _current_vector(self,
                        assets: Sequence[str]) -> Vector:
        """Current weights aligned to the universe, unheld names at zero."""
        return np.array([self.current_weights.get(asset_id, 0.0)
                         for asset_id in assets])

    def turnover(self,
                 weights: Vector,
                 assets: Sequence[str]) -> float:
        """One-way turnover of *weights* against the current holdings."""
        return one_way_turnover(weights, self._current_vector(assets))

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        """The budget as one inequality.

        The absolute value has a kink wherever an asset's weight equals its
        current weight, so this function is not differentiable everywhere and
        the gradient below is a subgradient — the sign vector, which is the
        true derivative away from those kinks and an arbitrary choice of one at
        them. SLSQP assumes smoothness and can in principle stall on a solution
        that sits exactly on many kinks at once. In practice it converges, and
        the verification pass refuses the answer if it does not: an exact
        formulation needs auxiliary variables for the positive and negative
        parts, which doubles the problem and is not worth it until a real case
        demands it.
        """
        current = self._current_vector(assets)

        return [Condition(
            label=f"turnover budget of {self.maximum:.4%}",
            kind=INEQUALITY,
            evaluate=lambda w: self.maximum - one_way_turnover(w, current),
            gradient=lambda w: -np.sign(w - current) / 2.0)]


class ExpectedReturnTarget(Constraint):
    """The portfolio must be expected to return exactly this much.

    The constraint that traces out a frontier: fix the return, minimise the
    variance, repeat. It is an equality rather than a floor because a frontier
    point is a specific point, and a floor would let every solve collapse onto
    the minimum-variance portfolio whenever that portfolio happened to clear
    the bar.

    Attributes:
        expected_returns: Expected return per asset, in the same units and over
            the same horizon as the risk model's covariance. Annualised, if the
            risk model is.
        target: The portfolio return being asked for.
    """

    def __init__(self,
                 expected_returns: dict[str, float],
                 target: float):
        self.expected_returns = dict(expected_returns)
        self.target = float(target)

    def validate(self,
                 assets: Sequence[str]) -> None:
        """Every asset must have an expected return.

        Treating an absent one as zero would quietly bias the whole frontier
        towards holding it, since a zero-return asset looks like the safest way
        to satisfy a low return target.
        """
        missing = sorted(set(assets) - set(self.expected_returns))
        if missing:
            raise CalculationError(
                "Optimiser",
                f"no expected return was given for: {missing}.")

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        returns = np.array([self.expected_returns[asset_id] for asset_id in assets])
        target = self.target

        return [Condition(label=f"expected return of {target:.4%}",
                          kind=EQUALITY,
                          evaluate=lambda w: float(returns @ w) - target,
                          gradient=lambda w: returns)]


class Cardinality(Constraint):
    """A limit on how many names may be held.

    Unlike every other constraint here this one is not convex — it counts
    non-zero positions, and no continuous solver can express that. It is
    honoured by a two-stage heuristic (see
    :func:`beacon.optimise.solver.minimise_tracking_error`): solve, keep the
    largest positions, then solve again with the rest pinned at zero. The
    answer satisfies the limit but is not proven optimal, and the exact problem
    is a mixed-integer program that would need a different solver entirely.

    Attributes:
        maximum: The largest number of names that may carry weight.
    """

    def __init__(self,
                 maximum: int):
        if maximum < 1:
            raise ValueError(f"cardinality must be at least 1, got {maximum}.")

        self.maximum = int(maximum)

    def solver_conditions(self,
                          assets: Sequence[str]) -> list[Condition]:
        """None: counting holdings is a step function.

        Its gradient is zero everywhere it is defined, so handing it to SLSQP
        would say that dropping a name costs nothing and changes nothing. The
        limit is enforced by restricting the problem instead.
        """
        return []

    def conditions(self,
                   assets: Sequence[str]) -> list[Condition]:
        """The count, as an inequality that is only ever *checked*."""
        return [Condition(
            label=f"at most {self.maximum} holdings",
            kind=INEQUALITY,
            evaluate=lambda w: float(self.maximum - count_holdings(w)))]


def count_holdings(weights: Vector) -> int:
    """How many positions carry meaningful weight."""
    return int((np.abs(weights) > HOLDING_THRESHOLD).sum())


def one_way_turnover(weights: Vector,
                     current: Vector) -> float:
    """Half the summed absolute weight change between two portfolios.

    The halving is what makes this one-way: under full investment every unit
    bought is a unit sold, so the undivided sum counts each trade twice.
    """
    return float(np.abs(weights - current).sum()) / 2.0
