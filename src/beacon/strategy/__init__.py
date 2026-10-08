# src/beacon/strategy/__init__.py
"""
Strategies: what a backtest holds, whatever the vehicle that holds it.

A bare index definition passed to `Backtest.run` is tracked in full.
`IndexTracking` holds an index another way: through an optimised subset or a
stratified sample. See `beacon.strategy.tracking`.
"""
from .tracking import (
    FullReplication,
    IndexTracking,
    OptimisedReplication,
    Replication,
    ReplicationContext,
    ReplicationStep,
    SampledReplication,
)

__all__ = [
    "FullReplication",
    "IndexTracking",
    "OptimisedReplication",
    "Replication",
    "ReplicationContext",
    "ReplicationStep",
    "SampledReplication",
]
