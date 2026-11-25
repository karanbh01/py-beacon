# src/beacon/analysis/analytics/holdings.py
"""
A run's holdings analytics: what each portfolio held: how many names, their
groups and their style.

    holdings = result.analytics.portfolio.holdings
    holdings.holdings_count(as_of="2024-06-28", of="all")
    holdings.style_box()
"""
# BN-438 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

from ..reports.context import FUND
from .category import Category, Day


class HoldingsCategory(Category):
    """A run's holdings analytics: what each portfolio held: how many names,
    their groups and their style. Each method is a statistic, named by its
    id; its keywords pick the form it is given in
    (`beacon.analysis.analytics.forms`)."""

    name = "holdings"
    statistics = ("holdings_count", "holdings_range", "holdings_distribution", "characteristics",
                  "group_weights", "group_exposures", "style_box",
                  "style_classification")

    # -- names ----------------------------------------------------------------

    def holdings_count(self,
                       *,
                       of: str = FUND,
                       as_of: Day = None) -> Any:
        """How many names the fund and its comparators held each day."""
        return self._get("holdings_count", of=of, as_of=as_of)

    def holdings_range(self,
                       *,
                       start: Day = None,
                       end: Day = None) -> Any:
        """The fewest and most names the fund and its comparators held on
        any day, and how many they hold now."""
        return self._get("holdings_range", start=start, end=end)

    def holdings_distribution(self,
                              *,
                              start: Day = None,
                              end: Day = None) -> Any:
        """How many days the fund held each number of names."""
        return self._get("holdings_distribution", start=start, end=end)

    def characteristics(self,
                        *,
                        of: str = FUND,
                        as_of: Day = None) -> Any:
        """Weighted market cap, P/E, P/B, dividend yield and earnings and
        sales growth of the fund and its comparators at each rebalance."""
        return self._get("characteristics", of=of, as_of=as_of)

    # -- groups ---------------------------------------------------------------

    def group_weights(self,
                      *,
                      as_of: Day = None) -> Any:
        """The fund's weight in each group, and less the tracked
        comparator's."""
        return self._get("group_weights", as_of=as_of)

    def group_exposures(self) -> Any:
        """The fund's latest weight in each sector and country, against its
        index's and benchmark's."""
        return self._get("group_exposures")

    # -- style ----------------------------------------------------------------

    def style_box(self) -> Any:
        """Each portfolio's weight in each of the nine size and style
        cells."""
        return self._get("style_box")

    def style_classification(self) -> Any:
        """The size and style each portfolio's weight is centred on, such as
        large blend."""
        return self._get("style_classification")
