# src/beacon/analysis/analytics/income.py
"""
A run's income analytics: the yields the portfolio paid and earned, and how
its dividends grew.

    income = result.analytics.portfolio.income
    income.income_yields(as_of="2024-12-31")
    income.dividend_growth()
"""
# BN-441 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from .category import Category, Day


class IncomeCategory(Category):
    """A run's income analytics: the yields the portfolio paid and earned,
    and how its dividends grew. Each method is a statistic, named by its id;
    its keywords pick the form it is given in
    (`beacon.analysis.analytics.forms`)."""

    name = "income"
    statistics = ("income_yields", "dividend_growth")

    # -- dividends ------------------------------------------------------------

    def income_yields(self,
                      *,
                      as_of: Day = None) -> Any:
        """Distribution yield and trailing dividend yield over 12 months,
        each day."""
        return self._get("income_yields", as_of=as_of)

    def dividend_growth(self) -> Any:
        """Each year's dividends and their growth."""
        return self._get("dividend_growth")
