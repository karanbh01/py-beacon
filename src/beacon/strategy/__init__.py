# src/beacon/strategy/__init__.py
"""
Strategies: what a backtest holds, whatever the vehicle that holds it.

A bare index definition passed to `Backtest.run` is tracked in full.
`IndexTracking` holds an index another way: through an optimised subset or a
stratified sample (see `beacon.strategy.tracking`), or synthetically, through
a substitute basket and a total return swap (see `beacon.strategy.synthetic`).
`ActiveStrategy` builds its own portfolio from a signal, against a benchmark
(see `beacon.strategy.active`).
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
from .synthetic import UCITS_EXPOSURE_LIMIT, SwapTerms, SyntheticReplication
from .tracking import (
    FullReplication,
    IndexTracking,
    OptimisedReplication,
    Replication,
    ReplicationStep,
    SampledReplication,
)

__all__ = [
    "UCITS_EXPOSURE_LIMIT",
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
    "SwapTerms",
    "SyntheticReplication",
    "TrackingErrorBudget",
    "TurnoverLimit",
]
