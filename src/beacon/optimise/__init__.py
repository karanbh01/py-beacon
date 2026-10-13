# src/beacon/optimise/__init__.py
"""
Portfolio optimisation.

Constraint classes describing what a portfolio is allowed to be, and a solver
that finds the closest feasible portfolio to a target: the tracking problem an
index business meets first.

Solving uses scipy, imported when a problem is solved rather than when this
package is. Constraints, configs and results are descriptions: `Backtest.run`
takes an :class:`OptimisationConfig`, and the catalogue serialises constraint
configurations, neither of which should cost scipy's import time.
"""
# The import/solve split above was introduced in BN-166.
from .config import OptimisationConfig, constraint_from_payload, constraint_payload
from .constraints import (
    BINDING_TOLERANCE,
    EQUALITY,
    FEASIBILITY_TOLERANCE,
    HOLDING_THRESHOLD,
    INEQUALITY,
    Cardinality,
    Condition,
    Constraint,
    ExpectedReturnTarget,
    FullInvestment,
    GroupBounds,
    PositionBounds,
    Slack,
    TurnoverBudget,
    count_holdings,
    one_way_turnover,
)
from .frontier import (
    DEFAULT_POINTS,
    EfficientFrontier,
    FrontierPoint,
    default_constraints,
    efficient_frontier,
    maximum_return_portfolio,
    minimum_variance_portfolio,
)
from .result import BindingConstraint, OptimisationResult, SolverDiagnostics
from .solver import Solution, minimise_tracking_error, solve_constrained

__all__ = [
    "BINDING_TOLERANCE",
    "DEFAULT_POINTS",
    "EQUALITY",
    "FEASIBILITY_TOLERANCE",
    "HOLDING_THRESHOLD",
    "INEQUALITY",
    "BindingConstraint",
    "Cardinality",
    "Condition",
    "Constraint",
    "EfficientFrontier",
    "ExpectedReturnTarget",
    "FrontierPoint",
    "FullInvestment",
    "GroupBounds",
    "OptimisationConfig",
    "OptimisationResult",
    "PositionBounds",
    "Slack",
    "Solution",
    "SolverDiagnostics",
    "TurnoverBudget",
    "constraint_from_payload",
    "constraint_payload",
    "count_holdings",
    "default_constraints",
    "efficient_frontier",
    "maximum_return_portfolio",
    "minimise_tracking_error",
    "minimum_variance_portfolio",
    "one_way_turnover",
    "solve_constrained",
]
