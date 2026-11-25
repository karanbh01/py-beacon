# src/beacon/analysis/analytics/portfolio.py
"""
A run's analytics of a whole portfolio, the fund's by default, in nine
categories.

    result.analytics.portfolio.performance
    result.analytics.portfolio.risk
"""
# BN-433 (Karan, 2026-11-24).
from functools import cached_property
from typing import Any

from .attribution import AttributionCategory
from .category import Category
from .flows import FlowsCategory
from .holdings import HoldingsCategory
from .income import IncomeCategory
from .performance import PerformanceCategory
from .relative import RelativeCategory
from .risk import RiskCategory
from .trading import TradingCategory
from .vehicle import VehicleCategory


class PortfolioLevel:
    """A run's analytics of a whole portfolio, in nine categories, each
    built on first read and kept."""

    def __init__(self,
                 analytics: Any):
        self._analytics = analytics

    @cached_property
    def performance(self) -> PerformanceCategory:
        """Returns, risk-adjusted ratios, rolling outcomes, and crises and
        regimes."""
        return PerformanceCategory(self._analytics)

    @cached_property
    def risk(self) -> RiskCategory:
        """Volatility and drawdown, the distribution of returns, value at
        risk, factors, the risk model, concentration and liquidity."""
        return RiskCategory(self._analytics)

    @cached_property
    def relative(self) -> RelativeCategory:
        """The fund against its index or benchmark: tracking, regression and
        capture."""
        return RelativeCategory(self._analytics)

    @cached_property
    def holdings(self) -> HoldingsCategory:
        """The names held, their groups and their style."""
        return HoldingsCategory(self._analytics)

    @cached_property
    def trading(self) -> TradingCategory:
        """Turnover, trades and their costs."""
        return TradingCategory(self._analytics)

    @cached_property
    def flows(self) -> FlowsCategory:
        """Money arriving and leaving, and what dealing in it cost."""
        return FlowsCategory(self._analytics)

    @cached_property
    def income(self) -> IncomeCategory:
        """The dividends the portfolio earned."""
        return IncomeCategory(self._analytics)

    @cached_property
    def vehicle(self) -> VehicleCategory:
        """The fund as an ETF and its futures."""
        return VehicleCategory(self._analytics)

    @cached_property
    def attribution(self) -> AttributionCategory:
        """What the return and the active return came from."""
        return AttributionCategory(self._analytics)

    def categories(self) -> list[Category]:
        """The nine categories, in order."""
        return [self.performance, self.risk, self.relative, self.holdings, self.trading,
                self.flows, self.income, self.vehicle, self.attribution]

    def __repr__(self) -> str:
        return f"PortfolioLevel({', '.join(category.name for category in self.categories())})"
