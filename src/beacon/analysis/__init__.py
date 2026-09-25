# src/beacon/analysis/__init__.py
"""
Analytics for indices, ETFs and portfolios.

Covers performance attribution (`attribute`, `Attribution`, cost and cap
drag), concentration and drift from target weights, liquidity, risk metrics
(volatility, Sharpe ratio, maximum drawdown) and ETF tracking analytics
(tracking difference, tracking error, premium or discount to NAV).
"""
from .attribution import (
    Attribution,
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
from .liquidity import average_daily_volume
from .risk import (
    RiskMetricsCalculator,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_volatility,
)

__all__ = [
    "Attribution",
    "AttributionResult",
    "ConcentrationMetrics",
    "Contribution",
    "DriftMetrics",
    "ETFAnalytics",
    "RiskMetricsCalculator",
    "attribute",
    "average_daily_volume",
    "calculate_max_drawdown",
    "calculate_premium_discount",
    "calculate_sharpe_ratio",
    "calculate_tracking_difference",
    "calculate_tracking_error",
    "calculate_volatility",
    "cap_drag",
    "carino_factor",
    "concentration",
    "cost_drag",
    "drift_from_target",
    "drift_history",
    "drifted_weights",
    "effective_number_of_assets",
    "herfindahl_index",
    "link_contributions",
    "simple_performance_attribution",
    "top_n_weight",
]
