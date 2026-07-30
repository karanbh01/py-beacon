# src/beacon/optimise/solver.py
"""
Constrained minimisation, and the tracking-error objective built on it.

:func:`solve_constrained` is the objective-agnostic core: give it something to
minimise and a list of constraints and it handles the box, the infeasibility
messages, the non-convex cardinality stage and the verification pass. The
frontier objectives in `beacon.optimise.frontier` use the same core, so the
guarantees below hold for all of them.

The first objective an index business needs is not mean-variance: it is "get me
as close as possible to this target, subject to what I am allowed to hold".
That is what this solves — minimise

    (w - b)ᵀ Σ (w - b)

over the weights ``w``, given a target ``b`` and whatever constraints were
supplied. With a risk model, Σ is its covariance and the objective is squared
tracking error. Without one, Σ is the identity and the objective is the squared
distance between the two weight vectors, which is the same problem with every
asset treated as equally risky and uncorrelated. The two share a code path
because they are the same problem; only the metric differs.

The objective is convex — Σ is positive semi-definite by construction — and
every constraint but cardinality is linear, so a local optimum is the global
one and SLSQP is an appropriate solver.

## Refusing rather than fudging

A solve either returns an answer that satisfies every constraint or it raises.
There is no third outcome where a violating vector comes back with a warning
attached, because a weight vector is the kind of thing a caller will act on and
an ignored warning becomes a breached mandate. Two checks enforce that: cheap
provable-infeasibility tests before solving, which produce a message naming
what is impossible rather than an opaque solver code, and a verification pass
afterwards that re-evaluates every constraint against the returned weights.

Non-convergence raises for the same reason. A stalled solve leaves a feasible
but not-necessarily-optimal point, and returning it with `converged=False` in
the diagnostics would make "this is the best answer" and "this is merely an
answer" look identical to anyone who does not check the flag.
"""
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .._optional import require
from ..exceptions import CalculationError
from ..risk.model import RiskModel
from .constraints import (
    Cardinality,
    Constraint,
    FullInvestment,
    GroupBounds,
    Slack,
    Vector,
    count_holdings,
)
from .result import BindingConstraint, OptimisationResult, SolverDiagnostics

require("scipy", "Portfolio optimisation")

from scipy.optimize import minimize  # noqa: E402

logger = logging.getLogger(__name__)

# Sequential Least Squares Programming. It handles a smooth objective with
# mixed equality and inequality constraints, which is exactly this problem.
METHOD = "SLSQP"

# Tighter than scipy's 1e-6 default. The default leaves constraint residuals
# around the tolerance at which this module calls a constraint violated, so a
# clean solve could be refused by its own verification pass.
FUNCTION_TOLERANCE = 1e-12

MAX_ITERATIONS = 500

# Feasibility slack for the pre-solve checks, which compare sums of bounds.
BOUND_TOLERANCE = 1e-9


def minimise_tracking_error(target_weights: pd.Series | dict[str, float],
                            constraints: Sequence[Constraint] | None = None,
                            risk_model: RiskModel | None = None) -> OptimisationResult:
    """Find the closest feasible portfolio to a target.

    Args:
        target_weights: What to track, by asset id. Defines the universe: the
            optimiser allocates over exactly these names, in this order.
        constraints: What the answer must satisfy. None means full investment
            alone, which is the smallest problem that has a unique answer.
        risk_model: Covariance to measure distance with. None treats every
            asset as equally risky and uncorrelated, which minimises plain
            squared weight distance.

    Returns:
        OptimisationResult: Optimal weights, the active position, which
        constraints bound, and how the solve went.

    Raises:
        CalculationError: If the constraints cannot all be satisfied, if the
            solver fails to converge, or if the returned weights violate a
            constraint. Also if a constraint or the risk model refers to assets
            outside the universe.
    """
    target = _as_series(target_weights)
    assets = list(target.index)
    rules = list(constraints) if constraints is not None else [FullInvestment()]

    if not assets:
        raise CalculationError("Optimiser", "the target weights are empty.")

    covariance = covariance_matrix(risk_model, assets)
    vector = target.to_numpy(dtype=float)

    def objective(weights: Vector) -> float:
        active = weights - vector

        return float(active @ covariance @ active)

    def gradient(weights: Vector) -> Vector:
        return np.asarray(2.0 * covariance @ (weights - vector), dtype=np.float64)

    solved = solve_constrained(objective, gradient, rules, assets, hint=vector)
    result = _build_result(target, assets, solved)

    logger.info(
        f"Optimised {len(assets)} assets: tracking error "
        f"{result.tracking_error():.6f}, turnover {result.turnover():.4%}, "
        f"{len(result.binding)} binding constraint(s), "
        f"{solved.outcome.nit} iteration(s).")

    return result


@dataclass(frozen=True)
class Solution:
    """A verified answer to a constrained problem.

    Attributes:
        weights: The solution vector, aligned to the universe.
        outcome: The solver's own result object.
        slacks: Every constraint's room at the solution.
        heuristic: Whether a non-convex constraint forced a restricted
            re-solve, in which case the answer is feasible but not proven
            optimal.
    """
    weights: Vector
    outcome: Any
    slacks: list[Slack]
    heuristic: bool


def solve_constrained(objective: Callable[[Vector], float],
                      gradient: Callable[[Vector], Vector],
                      rules: Sequence[Constraint],
                      assets: Sequence[str],
                      hint: Vector | None = None) -> Solution:
    """Minimise *objective* over the weights, subject to *rules*.

    The objective-agnostic core. Tracking error is one objective; portfolio
    variance, expected return and the Sharpe ratio are others, and all of them
    want the same constraint handling, the same infeasibility messages and the
    same refusal to return a violating answer.

    Args:
        objective: What to minimise, as a function of the weight vector.
        gradient: Its derivative. Required rather than optional — finite
            differences on a problem this small cost more accuracy than they
            save effort.
        rules: The constraints.
        assets: The universe, fixing the meaning of each weight position.
        hint: Where to start the search. None starts from equal weights.

    Returns:
        Solution: The verified answer.

    Raises:
        CalculationError: If the constraints cannot all be satisfied, if the
            solver fails to converge, or if the weights violate a constraint.
    """
    for rule in rules:
        rule.validate(assets)

    lows, highs = weight_box(rules, assets)
    _reject_infeasible(rules, assets, lows, highs)

    forced = _forced_by_bounds(rules, lows, highs)
    if forced is not None:
        return _forced_solution(objective, forced, rules, assets)

    start = hint if hint is not None else np.full(len(assets), 1.0 / len(assets))
    weights, outcome, heuristic = _solve_with_cardinality(
        objective, gradient, rules, assets, start, lows, highs)

    slacks = _all_slacks(rules, weights, assets)
    _reject_violations(slacks, outcome)

    return Solution(weights=weights,
                    outcome=outcome,
                    slacks=slacks,
                    heuristic=heuristic)


@dataclass(frozen=True)
class _DeterminedOutcome:
    """Stands in for a solver result when no solve was needed.

    Carries the same attributes the rest of this module reads off scipy's
    result object, so a determined answer flows through the reporting and
    verification path unchanged.
    """
    x: Vector
    fun: float
    success: bool = True
    status: int = 0
    message: str = "The bounds leave exactly one feasible portfolio."
    nit: int = 0
    nfev: int = 0


def _forced_by_bounds(rules: Sequence[Constraint],
                      lows: Vector,
                      highs: Vector) -> Vector | None:
    """The only feasible point, when the box and the budget leave exactly one.

    If the upper bounds sum to precisely the amount that must be invested, then
    every weight has to sit at its upper bound: there is no choice left to
    make. The same holds at the lower bounds.

    Worth detecting rather than handing to the solver. The feasible set is a
    single point, so no descent direction exists, and whether SLSQP calls that
    success or "positive directional derivative for linesearch" turns out to
    vary between scipy builds — the same input passed on eight CI cells and
    failed on the ninth. Computing the answer directly removes the question:
    it is arithmetic, not a search.

    A cap of exactly 1/n on n assets is not a contrived case, either. It is
    what anyone gets who caps an equal-weighted index at its natural weight.

    Returns:
        Vector or None: The determined weights, or None if the bounds leave
        room to optimise.
    """
    investment = _investment_target(rules)
    if investment is None:
        return None

    # Infinite bounds sum to infinity and never match a finite budget, so the
    # comparisons below are safe without a separate finiteness check.
    if abs(float(highs.sum()) - investment) <= BOUND_TOLERANCE:
        return np.asarray(highs, dtype=np.float64)

    if abs(float(lows.sum()) - investment) <= BOUND_TOLERANCE:
        return np.asarray(lows, dtype=np.float64)

    return None


def _forced_solution(objective: Callable[[Vector], float],
                     weights: Vector,
                     rules: Sequence[Constraint],
                     assets: Sequence[str]) -> Solution:
    """Package a determined answer, still checking it against every constraint.

    The bounds fix the weights, but nothing says a group limit or a turnover
    budget agrees with them. If one does not, the problem really is infeasible
    and it is refused exactly as any other violating answer would be.
    """
    slacks = _all_slacks(rules, weights, assets)
    outcome = _DeterminedOutcome(x=weights, fun=float(objective(weights)))
    _reject_violations(slacks, outcome)

    logger.info(
        "The position bounds leave a single feasible portfolio; returning it "
        "without a search.")

    return Solution(weights=weights,
                    outcome=outcome,
                    slacks=slacks,
                    heuristic=False)


def _as_series(weights: pd.Series | dict[str, float]) -> pd.Series:
    """Accept either a Series or a plain mapping of weights."""
    if isinstance(weights, pd.Series):
        return weights.astype(float)

    return pd.Series(weights, dtype=float)


def covariance_matrix(risk_model: RiskModel | None,
                       assets: Sequence[str]) -> Vector:
    """The metric the objective measures distance in.

    The identity is not a placeholder standing in for a missing model: it is
    the honest statement that without one, every unit of active weight is
    equally costly wherever it is taken.
    """
    if risk_model is None:
        return np.eye(len(assets))

    missing = sorted(set(assets) - set(risk_model.covariance.index))
    if missing:
        raise CalculationError(
            "Optimiser",
            f"the risk model does not cover every asset in the universe: {missing}.")

    ordered = risk_model.covariance.loc[list(assets), list(assets)]

    return np.asarray(ordered.to_numpy(), dtype=np.float64)


def weight_box(rules: Sequence[Constraint],
         assets: Sequence[str]) -> tuple[Vector, Vector]:
    """Intersect every position-bound constraint into one box.

    Several bounds may cover the same asset — a blanket rule plus a tighter one
    on a few names — and the answer must satisfy all of them, so the tightest
    limit on each side wins.
    """
    lows = np.full(len(assets), -np.inf)
    highs = np.full(len(assets), np.inf)

    for rule in rules:
        contribution = rule.bounds(assets)
        if contribution is None:
            continue

        lows = np.maximum(lows, [low for low, _ in contribution])
        highs = np.minimum(highs, [high for _, high in contribution])

    return lows, highs


def _reject_infeasible(rules: Sequence[Constraint],
                       assets: Sequence[str],
                       lows: Vector,
                       highs: Vector) -> None:
    """Raise on the infeasibilities that can be proven without solving.

    These are the cases where arithmetic on the bounds alone settles it. They
    are worth catching here because the alternative is an SLSQP exit code,
    which tells a user that something was incompatible but not what.
    """
    _reject_crossed_bounds(assets, lows, highs)

    investment = _investment_target(rules)
    if investment is not None:
        _reject_unreachable_total(investment, lows, highs)

    for rule in rules:
        if isinstance(rule, GroupBounds):
            _reject_impossible_group(rule, assets, lows, highs)

    _reject_impossible_cardinality(rules, assets, lows, highs, investment)


def _reject_crossed_bounds(assets: Sequence[str],
                           lows: Vector,
                           highs: Vector) -> None:
    """Raise if intersecting the position bounds left an empty range."""
    crossed = [assets[position] for position in range(len(assets))
               if lows[position] > highs[position] + BOUND_TOLERANCE]

    if crossed:
        raise CalculationError(
            "Optimiser",
            f"position bounds leave no allowed weight for: {crossed}. Two "
            f"bounds on the same asset conflict.")


def _investment_target(rules: Sequence[Constraint]) -> float | None:
    """The total that must be invested, if any constraint fixes one."""
    for rule in rules:
        if isinstance(rule, FullInvestment):
            return rule.target

    return None


def _reject_unreachable_total(investment: float,
                              lows: Vector,
                              highs: Vector) -> None:
    """Raise if the box cannot sum to the required total.

    The most that can be invested is the sum of the upper bounds and the least
    is the sum of the lower bounds; a target outside that range is impossible
    however the weight is arranged.
    """
    ceiling = float(highs.sum())
    floor = float(lows.sum())

    if ceiling < investment - BOUND_TOLERANCE:
        raise CalculationError(
            "Optimiser",
            f"the maximum weights total {ceiling:.4%}, which cannot reach the "
            f"{investment:.4%} that must be invested.")

    if floor > investment + BOUND_TOLERANCE:
        raise CalculationError(
            "Optimiser",
            f"the minimum weights total {floor:.4%}, which already exceeds the "
            f"{investment:.4%} that may be invested.")


def _reject_impossible_group(rule: GroupBounds,
                             assets: Sequence[str],
                             lows: Vector,
                             highs: Vector) -> None:
    """Raise if a group's limits contradict its members' position bounds."""
    positions = [position for position, asset_id in enumerate(assets)
                 if asset_id in set(rule.members)]

    ceiling = float(highs[positions].sum())
    floor = float(lows[positions].sum())

    if ceiling < rule.minimum - BOUND_TOLERANCE:
        raise CalculationError(
            "Optimiser",
            f"group '{rule.name}' must hold at least {rule.minimum:.4%} but its "
            f"members' maximum weights total only {ceiling:.4%}.")

    if floor > rule.maximum + BOUND_TOLERANCE:
        raise CalculationError(
            "Optimiser",
            f"group '{rule.name}' may hold at most {rule.maximum:.4%} but its "
            f"members' minimum weights already total {floor:.4%}.")


def _reject_impossible_cardinality(rules: Sequence[Constraint],
                                   assets: Sequence[str],
                                   lows: Vector,
                                   highs: Vector,
                                   investment: float | None) -> None:
    """Raise if a holding limit cannot coexist with the other constraints."""
    limit = _cardinality_limit(rules)
    if limit is None:
        return

    forced = _must_hold(lows, highs)
    if len(forced) > limit:
        held = [assets[position] for position in forced]
        raise CalculationError(
            "Optimiser",
            f"at most {limit} names may be held, but {len(forced)} have bounds "
            f"that exclude zero and so must be held: {held}.")

    if investment is None:
        return

    # The k largest upper bounds are the most that k names can carry.
    reachable = float(np.sort(highs)[::-1][:limit].sum())
    if reachable < investment - BOUND_TOLERANCE:
        raise CalculationError(
            "Optimiser",
            f"{limit} names can carry at most {reachable:.4%} between them, "
            f"which cannot reach the {investment:.4%} that must be invested.")


def _cardinality_limit(rules: Sequence[Constraint]) -> int | None:
    """The tightest holding limit across the constraints, if any."""
    limits = [rule.maximum for rule in rules if isinstance(rule, Cardinality)]

    return min(limits) if limits else None


def _must_hold(lows: Vector,
               highs: Vector) -> list[int]:
    """Positions whose bounds exclude zero, so they cannot be dropped."""
    return [position for position in range(len(lows))
            if lows[position] > 0.0 or highs[position] < 0.0]


def _solve_with_cardinality(objective: Callable[[Vector], float],
                            gradient: Callable[[Vector], Vector],
                            rules: Sequence[Constraint],
                            assets: Sequence[str],
                            hint: Vector,
                            lows: Vector,
                            highs: Vector) -> tuple[Vector, Any, bool]:
    """Solve, then enforce any holding limit by a second restricted solve.

    Cardinality is not convex and cannot be given to a continuous solver, so it
    is honoured the way practitioners do: solve freely, keep the largest
    positions, and re-solve with the rest pinned at zero. The second solve
    matters — simply deleting the small positions and renormalising would leave
    a portfolio nobody optimised.

    The result satisfies the limit but is not proven optimal. Choosing which
    names to keep is a combinatorial problem, and the largest positions of the
    unrestricted solution are a good guess rather than a provably right one.
    """
    outcome = _solve(objective, gradient, rules, assets, hint, lows, highs)
    limit = _cardinality_limit(rules)

    if limit is None or count_holdings(outcome.x) <= limit:
        return outcome.x, outcome, False

    keep = _names_to_keep(outcome.x, lows, highs, limit)
    dropped = [position for position in range(len(assets)) if position not in keep]

    restricted_lows = lows.copy()
    restricted_highs = highs.copy()
    restricted_lows[dropped] = 0.0
    restricted_highs[dropped] = 0.0

    logger.info(
        f"Cardinality limit of {limit} bound: re-solving with "
        f"{len(dropped)} name(s) pinned at zero.")

    second = _solve(objective, gradient, rules, assets, hint,
                    restricted_lows, restricted_highs)
    _reject_infeasible_restriction(rules, assets, second.x, limit)

    return second.x, second, True


def _reject_infeasible_restriction(rules: Sequence[Constraint],
                                   assets: Sequence[str],
                                   weights: Vector,
                                   limit: int) -> None:
    """Raise when pinning names at zero broke a constraint the full set met.

    Worth its own message rather than falling through to the generic violation
    error, which would name the constraint that broke and imply the caller
    asked for something impossible. They did not: the constraint was satisfiable
    before the heuristic chose a subset, and it is the choice that failed.
    """
    violated = [slack.label for slack in _all_slacks(rules, weights, assets)
                if slack.is_violated]
    if not violated:
        return

    raise CalculationError(
        "Optimiser",
        f"a holding limit of {limit} could not be met without breaking "
        f"{'; '.join(violated)}. Which names to keep is a combinatorial "
        f"problem, and the heuristic here keeps the largest positions of the "
        f"unrestricted solution — it cannot see that a different subset of the "
        f"same size would have satisfied everything.")


def _names_to_keep(weights: Vector,
                   lows: Vector,
                   highs: Vector,
                   limit: int) -> set[int]:
    """Choose which positions survive a holding limit.

    Names whose bounds exclude zero come first because they cannot be dropped
    at all; the rest are ranked by absolute weight, on the reasoning that the
    largest positions are the ones doing the tracking. The pre-solve check has
    already established that the forced names fit within the limit.
    """
    forced = set(_must_hold(lows, highs))
    ranked = sorted(range(len(weights)),
                    key=lambda position: (position in forced, abs(weights[position])),
                    reverse=True)

    return set(ranked[:limit]) | forced


def _solve(objective: Callable[[Vector], float],
           gradient: Callable[[Vector], Vector],
           rules: Sequence[Constraint],
           assets: Sequence[str],
           hint: Vector,
           lows: Vector,
           highs: Vector) -> Any:
    """Run SLSQP once over the given box."""
    return minimize(objective,
                    x0=_starting_point(rules, hint, lows, highs),
                    jac=gradient,
                    method=METHOD,
                    bounds=list(zip(lows, highs, strict=True)),
                    constraints=_scipy_constraints(rules, assets),
                    options={"maxiter": MAX_ITERATIONS, "ftol": FUNCTION_TOLERANCE})


def _scipy_constraints(rules: Sequence[Constraint],
                       assets: Sequence[str]) -> list[dict[str, Any]]:
    """Translate the constraints the solver can use into its own dicts."""
    dicts: list[dict[str, Any]] = []

    for rule in rules:
        for condition in rule.solver_conditions(assets):
            entry: dict[str, Any] = {"type": condition.kind, "fun": condition.evaluate}
            if condition.gradient is not None:
                entry["jac"] = condition.gradient

            dicts.append(entry)

    return dicts


def _starting_point(rules: Sequence[Constraint],
                    hint: Vector,
                    lows: Vector,
                    highs: Vector) -> Vector:
    """Where to start the search.

    The hint, pulled inside the box and then nudged to invest the right total.
    For a tracking solve the hint is the target itself, so when the target is
    already feasible the search starts on the answer — the correct behaviour
    rather than a shortcut, since with nothing else binding the closest
    feasible portfolio to the target *is* the target.
    """
    start = np.clip(hint, lows, highs)

    investment = _investment_target(rules)
    if investment is None:
        return start

    deficit = investment - float(start.sum())
    room = (highs - start) if deficit > 0 else (start - lows)
    available = float(np.sum(np.where(np.isfinite(room), room, 0.0)))

    if abs(deficit) <= BOUND_TOLERANCE or available <= 0.0:
        return start

    share = np.where(np.isfinite(room), room, 0.0) / available

    return start + deficit * share


def _all_slacks(rules: Sequence[Constraint],
                weights: Vector,
                assets: Sequence[str]) -> list[Slack]:
    """Measure every constraint against the solution.

    Every constraint, not only the ones the solver saw: position bounds were
    enforced as a box and cardinality was enforced by restriction, and both
    still have to be checked against the vector that actually came back.
    """
    return [slack for rule in rules for slack in rule.report(weights, assets)]


def _reject_violations(slacks: Sequence[Slack],
                       outcome: Any) -> None:
    """Refuse an answer that breaks a constraint or a solve that stalled."""
    violated = [slack for slack in slacks if slack.is_violated]

    if violated:
        detail = "; ".join(f"{slack.label} (short by {-slack.slack:.6f})"
                           for slack in violated)
        raise CalculationError(
            "Optimiser",
            f"no feasible portfolio was found: the best available point still "
            f"violates {len(violated)} constraint(s) — {detail}.")

    if not outcome.success:
        raise CalculationError(
            "Optimiser",
            f"the solver did not converge, so the weights it reached satisfy "
            f"every constraint but are not optimal: {outcome.message} "
            f"(status {outcome.status}).")


def _build_result(target: pd.Series,
                  assets: Sequence[str],
                  solved: Solution) -> OptimisationResult:
    """Assemble the result object from a verified solution."""
    outcome = solved.outcome
    binding = [BindingConstraint(label=slack.label, kind=slack.kind, slack=slack.slack)
               for slack in solved.slacks if slack.is_binding]
    binding.sort(key=lambda constraint: abs(constraint.slack))

    diagnostics = SolverDiagnostics(converged=bool(outcome.success),
                                    iterations=int(outcome.nit),
                                    evaluations=int(outcome.nfev),
                                    objective=float(outcome.fun),
                                    status=int(outcome.status),
                                    message=str(outcome.message))

    return OptimisationResult(
        weights=pd.Series(solved.weights, index=list(assets), name="optimal_weight"),
        target_weights=target.rename("target_weight"),
        binding=binding,
        diagnostics=diagnostics,
        slacks=list(solved.slacks),
        heuristic=solved.heuristic)
