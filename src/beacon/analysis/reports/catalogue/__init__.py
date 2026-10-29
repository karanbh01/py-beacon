# src/beacon/analysis/reports/catalogue/__init__.py
"""
Every statistic a run's analysis gives, in the order the categories are read.

    from beacon.analysis.reports.catalogue import CATALOGUE

    CATALOGUE["sharpe_ratio"].label  # "Sharpe ratio"
"""
# BN-398/BN-400.
from ..statistics import Catalogue
from . import attribution, performance, portfolio, relative, risk

CATALOGUE = Catalogue({statistic.id: statistic
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

__all__ = ["CATALOGUE", "CATEGORIES"]
