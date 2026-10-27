# src/beacon/analysis/performance/__init__.py
"""
A run's performance: its returns, risk and figures against its comparators.

`PerformanceAnalytics` measures a run's figures from its `RunData`, with the
`AnalysisSettings` they are measured with; `RelativeAnalytics` its figures
against one comparator; `CrisisAnalytics` how it held up in crises.
`beacon.analysis.performance.formulas` holds the formulas, one definition
each, that every figure is computed with.
"""
from typing import Any

from .settings import AnalysisSettings

__all__ = ["AnalysisSettings", "CrisisAnalytics", "PerformanceAnalytics", "RelativeAnalytics"]


def __getattr__(name: str) -> Any:
    # BN-371: `PerformanceAnalytics` reads a `RunData`, whose module imports
    # the backtest's result, which imports the formulas here, so it is
    # imported when first asked for.
    if name == "PerformanceAnalytics":
        from .analytics import PerformanceAnalytics  # noqa: PLC0415

        return PerformanceAnalytics

    if name == "CrisisAnalytics":
        from .crises import CrisisAnalytics  # noqa: PLC0415

        return CrisisAnalytics

    if name == "RelativeAnalytics":
        from .relative_analytics import RelativeAnalytics  # noqa: PLC0415

        return RelativeAnalytics

    raise AttributeError(f"module 'beacon.analysis.performance' has no attribute '{name}'")
