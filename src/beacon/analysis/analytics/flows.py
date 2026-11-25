# src/beacon/analysis/analytics/flows.py
"""
A run's flows analytics: money arriving and leaving, and what dealing in it
cost.

    flows = result.analytics.portfolio.flows
    flows.net_flows()
    flows.dealing_costs()
"""
# BN-440 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from .category import Category, Day


class FlowsCategory(Category):
    """A run's flows analytics: money arriving and leaving, and what dealing
    in it cost. Each method is a statistic, named by its id; its keywords
    pick the form it is given in (`beacon.analysis.analytics.forms`)."""

    name = "flows"
    statistics = ("net_flows", "flow_costs", "dealing_costs", "class_costs")

    # -- money in and out -----------------------------------------------------

    def net_flows(self) -> Any:
        """Each month's flows and their share of starting AUM."""
        return self._get("net_flows")

    def flow_costs(self,
                   *,
                   as_of: Day = None) -> Any:
        """The cost of trading each day's flows, less creation fees
        received."""
        return self._get("flow_costs", as_of=as_of)

    def dealing_costs(self,
                      *,
                      by: str | None = None) -> Any:
        """Each subscription or redemption: trading caused, what dealers
        paid in, net cost to the fund and in bp, and for an ETF the AP's
        cost.

        ``by="kind"`` adds them up by kind of flow over the run, and
        ``by="year"`` by year and kind."""
        return self._get("dealing_costs", by=by)

    def class_costs(self,
                    *,
                    start: Day = None,
                    end: Day = None) -> Any:
        """Each class's fees, hedge costs less carry earned, its flows'
        trading, its share of the rest, and the net in bp."""
        return self._get("class_costs", start=start, end=end)
