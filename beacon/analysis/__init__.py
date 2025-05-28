# beacon/analysis/__init__.py
"""
The __init__.py for the 'analysis' module.

This module provides tools for analyzing the performance and
risk characteristics of indices, ETFs, and portfolios.
"""
from .risk import RiskMetricsCalculator, calculate_volatility, calculate_sharpe_ratio, calculate_max_drawdown
from .etf.analytics import ETFAnalytics, calculate_tracking_difference, calculate_tracking_error, calculate_premium_discount
from .attribution import Attribution, simple_performance_attribution

__all__ = [
    "RiskMetricsCalculator",
    "calculate_volatility",
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    "ETFAnalytics",
    "calculate_tracking_difference",
    "calculate_tracking_error",
    "calculate_premium_discount",
    "Attribution",
    "simple_performance_attribution",
]