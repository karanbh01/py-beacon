# src/beacon/analysis/analytics/attribution.py
"""
A run's attribution analytics: what the return and the active return came
from: names, groups, currencies, factors and the implementation.

    attribution = result.analytics.portfolio.attribution
    attribution.contribution()
    attribution.active_attribution()
"""
# BN-443 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from .category import Category, Day


class AttributionCategory(Category):
    """A run's attribution analytics: what the return and the active return
    came from: names, groups, currencies, factors and the implementation.
    Each method is a statistic, named by its id; its keywords pick the form
    it is given in (`beacon.analysis.analytics.forms`)."""

    name = "attribution"
    statistics = ("contribution", "active_attribution", "brinson", "currency_attribution",
                  "factor_attribution", "implementation_breakdown")

    # -- attribution ----------------------------------------------------------

    def contribution(self,
                     *,
                     start: Day = None,
                     end: Day = None) -> Any:
        """Each name's and group's contribution to the return, Carino-
        linked."""
        return self._get("contribution", start=start, end=end)

    def active_attribution(self,
                           *,
                           start: Day = None,
                           end: Day = None) -> Any:
        """Each name's and group's contribution to the return over the
        tracked comparator, from its active weight."""
        return self._get("active_attribution", start=start, end=end)

    def brinson(self,
                *,
                start: Day = None,
                end: Day = None) -> Any:
        """Allocation, selection and interaction by group, cash its own
        group, and each group's selection and interaction by name, linked
        over the run."""
        return self._get("brinson", start=start, end=end)

    def currency_attribution(self,
                             *,
                             start: Day = None,
                             end: Day = None) -> Any:
        """Local return, currency effect and cross term, by currency."""
        return self._get("currency_attribution", start=start, end=end)

    def factor_attribution(self,
                           *,
                           start: Day = None,
                           end: Day = None) -> Any:
        """The return explained by factor exposures, holdings-based and
        returns-based."""
        return self._get("factor_attribution", start=start, end=end)

    def implementation_breakdown(self,
                                 *,
                                 start: Day = None,
                                 end: Day = None) -> Any:
        """The tracking difference split into cap and screens, drift, cash
        drag, equitisation, each cost, fees, withholding, creation fees and
        dividend treatment."""
        return self._get("implementation_breakdown", start=start, end=end)
