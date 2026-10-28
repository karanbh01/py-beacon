# src/beacon/analysis/attribution/__init__.py
"""
Performance attribution: which names, groups and decisions produced a return.

`beacon.analysis.attribution.contribution` decomposes one return into its
names' contributions, linked over many periods (`attribute`), with the cap and
cost drags beside it. `Attribution` does it for a run: by name or by group, for
the fund or a comparator, and against the index or the benchmark.
"""
from typing import Any

from .contribution import (
    AttributionResult,
    Contribution,
    attribute,
    cap_drag,
    carino_factor,
    cost_drag,
    drifted_weights,
    link_contributions,
    simple_performance_attribution,
)

__all__ = [
    "Attribution",
    "AttributionResult",
    "Contribution",
    "attribute",
    "cap_drag",
    "carino_factor",
    "cost_drag",
    "drifted_weights",
    "link_contributions",
    "simple_performance_attribution",
]


def __getattr__(name: str) -> Any:
    # BN-393: Attribution reads a RunData, whose module imports the backtest
    # result, which imports this package; so it is imported when first asked
    # for.
    if name == "Attribution":
        from .analytics import Attribution  # noqa: PLC0415

        return Attribution

    raise AttributeError(f"module 'beacon.analysis.attribution' has no attribute '{name}'")
