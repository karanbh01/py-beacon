# src/beacon/analysis/risk/__init__.py
"""
Risk analysis of a run, and the single-series risk metrics.

`beacon.analysis.risk.model` chooses and sets up the risk model an analysis
uses, and estimates it once at each rebalance. The single-series metrics
(volatility, Sharpe ratio, maximum drawdown) are in
`beacon.analysis.risk.single_series`, and importable from here as before.
"""
from .single_series import (
    RiskMetricsCalculator,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)

__all__ = [
    "RiskMetricsCalculator",
    "calculate_max_drawdown",
    "calculate_sharpe_ratio",
    "calculate_volatility",
]
