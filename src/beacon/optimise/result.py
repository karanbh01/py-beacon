# src/beacon/optimise/result.py
"""
OptimisationResult — the output of a solve.

Same shape as `IndexResult`, `BacktestResult` and `RiskModel`: a dataclass of
pandas structures with accessors that answer the questions a caller actually
has, rather than handing back a bare weight vector and leaving everyone to
recompute the active position and the turnover themselves.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..risk.model import RiskModel
from .constraints import Slack, count_holdings, one_way_turnover

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BindingConstraint:
    """A constraint the solution sits exactly on.

    Binding constraints are the interesting part of an answer: they are the
    rules that actually cost something, and relaxing one of them is the only
    way to improve the objective.

    Attributes:
        label: What the constraint was.
        kind: EQUALITY or INEQUALITY.
        slack: Room left, at or near zero by definition of binding. Carried so
            a caller can see how tight "tight" was.
    """
    label: str
    kind: str
    slack: float


@dataclass(frozen=True)
class SolverDiagnostics:
    """What the solver did.

    Attributes:
        converged: Whether the solver reported success. An answer is only
            returned when this is True and the weights pass verification, so a
            caller seeing this is seeing a solve that worked.
        iterations: Major iterations taken.
        evaluations: Objective evaluations.
        objective: Final objective value — a variance, so the tracking error is
            its square root.
        status: The solver's numeric exit code.
        message: The solver's own description of how it exited.
    """
    converged: bool
    iterations: int
    evaluations: int
    objective: float
    status: int
    message: str


@dataclass
class OptimisationResult:
    """Optimal weights, what they cost, and which rules bound.

    Attributes:
        weights: The solution, indexed by asset id.
        target_weights: What the solve was tracking, on the same index.
        binding: Constraints the solution sits on, tightest first.
        diagnostics: How the solve went.
        slacks: Every constraint's room at the solution, binding or not. Kept
            so a caller can see what nearly bound as well as what did.
        heuristic: Whether a non-convex constraint forced a heuristic stage, in
            which case the answer satisfies every constraint but is not proven
            optimal.
    """
    weights: pd.Series
    target_weights: pd.Series
    binding: list[BindingConstraint]
    diagnostics: SolverDiagnostics
    slacks: list[Slack] = field(default_factory=list)
    heuristic: bool = False
    _risk_model: RiskModel | None = field(default=None, repr=False, compare=False)

    def with_risk_model(self,
                        risk_model: RiskModel) -> "OptimisationResult":
        """Bind a risk model for risk-based accessors. Returns self."""
        self._risk_model = risk_model
        return self

    @property
    def asset_ids(self) -> list[str]:
        """Assets in the solution, in weight-vector order."""
        return list(self.weights.index)

    @property
    def active_weights(self) -> pd.Series:
        """Solution minus target — the active position.

        Sums to zero whenever both sides are fully invested to the same total,
        which is the usual case and worth checking: a non-zero sum means the
        solve was allowed to change how much is invested, not just where.
        """
        return (self.weights - self.target_weights).rename("active_weight")

    @property
    def holdings(self) -> int:
        """How many names carry meaningful weight."""
        return count_holdings(self.weights.to_numpy(dtype=float))

    def tracking_error(self) -> float:
        """Distance from the target, in the metric the solve minimised.

        With a risk model this is annualised tracking error, the quantity a
        tracking mandate is measured on. Without one the objective's identity
        covariance makes it the Euclidean distance between the two weight
        vectors — a sensible thing to minimise, but not a volatility, and not
        comparable to a number produced with a risk model.
        """
        return float(np.sqrt(max(self.diagnostics.objective, 0.0)))

    def turnover(self,
                 current_weights: dict[str, float] | None = None) -> float:
        """One-way turnover from *current_weights* to the solution.

        Half the summed absolute weight change, matching
        :class:`~beacon.optimise.constraints.TurnoverBudget`. Defaults to
        measuring against the target, which answers "how far did the optimiser
        move me off the index".

        Args:
            current_weights: Where the portfolio is now. None measures against
                the target weights.

        Returns:
            float: One-way turnover as a fraction of the portfolio.
        """
        if current_weights is None:
            reference = self.target_weights
        else:
            reference = pd.Series(current_weights).reindex(self.weights.index).fillna(0.0)

        return one_way_turnover(self.weights.to_numpy(dtype=float),
                                reference.to_numpy(dtype=float))

    def to_frame(self) -> pd.DataFrame:
        """Target, optimal and active weights side by side, largest active first."""
        frame = pd.DataFrame({"target_weight": self.target_weights,
                              "optimal_weight": self.weights,
                              "active_weight": self.active_weights})

        return frame.reindex(frame["active_weight"].abs()
                             .sort_values(ascending=False).index)

    def binding_labels(self) -> list[str]:
        """Just the descriptions of the binding constraints."""
        return [constraint.label for constraint in self.binding]
