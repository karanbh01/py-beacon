# src/beacon/analysis/reports/catalogue/__init__.py
"""
Every statistic a run's analysis gives, in the order the categories are read.

    from beacon.analysis.reports.catalogue import CATALOGUE

    CATALOGUE["sharpe_ratio"].label  # "Sharpe ratio"
"""
# BN-398/BN-400.
from dataclasses import replace

from ...performance.crises import AFTER_MONTHS
from ..statistics import Catalogue
from . import attribution, performance, portfolio, relative, risk

EVERY = ("*",)
CONTRIBUTIONS = ("contribution", "total_return", "residual", "cap_drag", "cost_drag")

# Which figures are returns or differences, whose sign says better or worse,
# so a client colours those and never a volatility or a weight (beacon-ui,
# 2026-10-29).
SIGNED = {
    "total_return": EVERY, "annualised_return": EVERY, "money_weighted_return": EVERY,
    "trailing_returns": EVERY, "period_returns": ("total_return",),
    "best_and_worst": ("value",), "gains_and_losses": ("average_gain", "average_loss"),
    "streaks": ("total",), "mean_return": EVERY, "month_of_year": EVERY,
    "tracking_difference": EVERY, "annualised_relative_return": EVERY, "alpha": ("alpha", "t"),
    "tracking_difference_by_year": EVERY, "rolling_outcomes": ("worst", "median", "best"),
    "crisis_returns": ("fund", "comparator", "outperformance", "alpha",
                       *(f"{book} {months}m" for months in AFTER_MONTHS
                         for book in ("fund", "comparator"))),
    "crisis_summary": ("average_outperformance", "pooled_alpha"),
    "regimes": ("fund", "comparator", "fund_annual", "comparator_annual", "difference", "alpha"),
    "stress_windows": ("fund_return", "index_return", "optimised_return", "benchmark_return"),
    "factor_returns": EVERY,
    "contribution": CONTRIBUTIONS, "active_attribution": CONTRIBUTIONS,
    "brinson": ("fund_return", "comparator_return", "allocation", "selection", "interaction",
                "total", "active_return", "explained", "residual"),
    "currency_attribution": EVERY, "factor_attribution": EVERY,
    "implementation_breakdown": EVERY,
}

CATALOGUE = Catalogue({statistic.id: replace(statistic, signed=SIGNED.get(statistic.id, ()))
                       for module in (performance, relative, portfolio, risk, attribution)
                       for statistic in module.statistics()})

# The categories, in the order the formulas page and the reports read them.
CATEGORIES = {
    "returns": "Returns", "risk": "Risk", "drawdown": "Drawdown",
    "distribution": "Distribution", "risk_adjusted": "Risk-adjusted", "relative": "Relative",
    "rolling": "Rolling", "crises_and_regimes": "Crises and regimes",
    "portfolio": "Portfolio", "trading_and_costs": "Trading and costs",
    "cash_and_flows": "Cash and flows", "income": "Income", "etf": "ETF", "futures": "Futures",
    "factors": "Factors", "value_at_risk": "Value at risk",
    "risk_contributions": "Risk contributions", "concentration": "Concentration",
    "liquidity": "Liquidity", "attribution": "Attribution",
}

__all__ = ["CATALOGUE", "CATEGORIES", "SIGNED"]
