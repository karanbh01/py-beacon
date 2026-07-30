# src/beacon/optimise/__init__.py
"""
Portfolio optimisation.

Constraint classes describing what a portfolio is allowed to be, and a solver
that finds the closest feasible portfolio to a target — the tracking problem an
index business meets first.

This package needs scipy, which ships in the ``optimise`` extra:

    pip install "py-beacon[optimise]"

Importing it without scipy raises MissingDependencyError naming that extra, so
`beacon` itself stays importable on pandas and numpy alone.
"""
from .constraints import (
    BINDING_TOLERANCE,
    EQUALITY,
    FEASIBILITY_TOLERANCE,
    HOLDING_THRESHOLD,
    INEQUALITY,
    Cardinality,
    Condition,
    Constraint,
    FullInvestment,
    GroupBounds,
    PositionBounds,
    Slack,
    TurnoverBudget,
    count_holdings,
    one_way_turnover,
)
from .result import BindingConstraint, OptimisationResult, SolverDiagnostics
from .solver import minimise_tracking_error

__all__ = [
    "BINDING_TOLERANCE",
    "EQUALITY",
    "FEASIBILITY_TOLERANCE",
    "HOLDING_THRESHOLD",
    "INEQUALITY",
    "BindingConstraint",
    "Cardinality",
    "Condition",
    "Constraint",
    "FullInvestment",
    "GroupBounds",
    "OptimisationResult",
    "PositionBounds",
    "Slack",
    "SolverDiagnostics",
    "TurnoverBudget",
    "count_holdings",
    "minimise_tracking_error",
    "one_way_turnover",
]
