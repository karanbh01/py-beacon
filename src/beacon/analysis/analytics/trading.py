# src/beacon/analysis/analytics/trading.py
"""
A run's trading analytics: turnover, trades and their costs.

    trading = result.analytics.portfolio.trading
    trading.one_way_turnover()
    trading.costs()
"""
# BN-439 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from .category import Category, Day


class TradingCategory(Category):
    """A run's trading analytics: turnover, trades and their costs. Each
    method is a statistic, named by its id; its keywords pick the form it is
    given in (`beacon.analysis.analytics.forms`)."""

    name = "trading"
    statistics = ("one_way_turnover", "two_way_turnover", "trade_counts", "costs",
                  "cost_per_traded", "cash_trading")

    # -- turnover and costs ---------------------------------------------------

    def one_way_turnover(self,
                         *,
                         start: Day = None,
                         end: Day = None) -> Any:
        """The lesser of buys and sells over the NAV, each year, and a year
        on average."""
        return self._get("one_way_turnover", start=start, end=end)

    def two_way_turnover(self,
                         *,
                         start: Day = None,
                         end: Day = None) -> Any:
        """All buys and sells over the NAV, each year, and a year on
        average."""
        return self._get("two_way_turnover", start=start, end=end)

    def trade_counts(self,
                     *,
                     start: Day = None,
                     end: Day = None,
                     by: str | None = None) -> Any:
        """Counts and values of the trading, each year.

        ``by="reason"`` gives them by why each trade was made, with each
        reason's share of the value traded."""
        return self._get("trade_counts", start=start, end=end, by=by)

    def costs(self,
              *,
              start: Day = None,
              end: Day = None,
              by: str | None = None) -> Any:
        """Fixed costs, market impact, futures rolls and trading, creation
        fees received, and the net, in money and in bp of assets a year.

        ``by="year"`` gives each year's, with the net in bp of the year's
        mean assets; ``by="reason"`` the trading costs by why each trade was
        made, with the value traded and the cost in bp of it."""
        return self._get("costs", start=start, end=end, by=by)

    def cost_per_traded(self,
                        *,
                        start: Day = None,
                        end: Day = None) -> Any:
        """Trading costs in bp of the value traded."""
        return self._get("cost_per_traded", start=start, end=end)

    def cash_trading(self) -> Any:
        """Every trade made to raise or spend cash, by what it was for."""
        return self._get("cash_trading")
