# src/beacon/analysis/analytics/risk.py
"""
A run's risk analytics: volatility and drawdown, the distribution of returns,
value at risk, factors, the risk model, concentration and liquidity.
"""
# BN-433. Its statistics are wired in their own issue.
from .category import Category


class RiskCategory(Category):
    """A run's risk analytics: volatility and drawdown, the distribution of
    returns, value at risk, factors, the risk model, concentration and
    liquidity."""

    name = "risk"
