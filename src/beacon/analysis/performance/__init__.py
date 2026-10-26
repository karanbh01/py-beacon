# src/beacon/analysis/performance/__init__.py
"""
A run's performance: its returns, risk and figures against its comparators.

`PerformanceAnalytics` measures a run's figures from its `RunData`, with the
`AnalysisSettings` they are measured with; `beacon.analysis.performance.formulas`
holds the formulas, one definition each, that every figure is computed with.
"""
from typing import Any

from .settings import AnalysisSettings

__all__ = ["AnalysisSettings", "PerformanceAnalytics"]


def __getattr__(name: str) -> Any:
    # BN-371: `PerformanceAnalytics` reads a `RunData`, whose module imports
    # the backtest's result, which imports the formulas here, so it is
    # imported when first asked for.
    if name == "PerformanceAnalytics":
        from .analytics import PerformanceAnalytics  # noqa: PLC0415

        return PerformanceAnalytics

    raise AttributeError(f"module 'beacon.analysis.performance' has no attribute '{name}'")
