# src/beacon/strategy/__init__.py
"""
Strategies: what a backtest holds, whatever the vehicle that holds it.

A bare index definition passed to `Backtest.run` is tracked in full.
`IndexTracking` holds an index another way: through an optimised subset or a
stratified sample (see `beacon.strategy.tracking`). `ActiveStrategy` builds
its own portfolio from a signal, against a benchmark (see
`beacon.strategy.active`).
"""
from .active import ActiveStep, ActiveStrategy, Construction, MaxAlpha, MeanVariance
from .base import StrategyContext
from .constraints import (
    ActiveConstraint,
    ActiveProblem,
    ActiveShare,
    HoldingsLimit,
    RelativePositionBounds,
    RelativeSectorBounds,
    TrackingErrorBudget,
    TurnoverLimit,
)
from .signals import FieldSignal, FunctionSignal, Momentum, Signal
from .tracking import (
    FullReplication,
    IndexTracking,
    OptimisedReplication,
    Replication,
    ReplicationStep,
    SampledReplication,
)

__all__ = [
    "ActiveConstraint",
    "ActiveProblem",
    "ActiveShare",
    "ActiveStep",
    "ActiveStrategy",
    "Construction",
    "FieldSignal",
    "FullReplication",
    "FunctionSignal",
    "HoldingsLimit",
    "IndexTracking",
    "MaxAlpha",
    "MeanVariance",
    "Momentum",
    "OptimisedReplication",
    "RelativePositionBounds",
    "RelativeSectorBounds",
    "Replication",
    "ReplicationStep",
    "SampledReplication",
    "Signal",
    "StrategyContext",
    "TrackingErrorBudget",
    "TurnoverLimit",
]
