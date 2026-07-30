# src/beacon/analysis/__init__.py
"""
The __init__.py for the 'analysis' module.

This module provides tools for analyzing the performance and
risk characteristics of indices, ETFs, and portfolios.
"""
from .attribution import Attribution, simple_performance_attribution
from .concentration import (
    ConcentrationMetrics,
    DriftMetrics,
    concentration,
    drift_from_target,
    drift_history,
    effective_number_of_assets,
    herfindahl_index,
    top_n_weight,
)
from .etf.analytics import (
    ETFAnalytics,
    calculate_premium_discount,
    calculate_tracking_difference,
    calculate_tracking_error,
)
from .risk import (
    RiskMetricsCalculator,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)

__all__ = [
    "Attribution",
    "ConcentrationMetrics",
    "DriftMetrics",
    "ETFAnalytics",
    "RiskMetricsCalculator",
    "calculate_max_drawdown",
    "calculate_premium_discount",
    "calculate_sharpe_ratio",
    "calculate_tracking_difference",
    "calculate_tracking_error",
    "calculate_volatility",
    "concentration",
    "drift_from_target",
    "drift_history",
    "effective_number_of_assets",
    "herfindahl_index",
    "simple_performance_attribution",
    "top_n_weight",
]
