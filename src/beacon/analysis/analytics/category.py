# src/beacon/analysis/analytics/category.py
"""
What every category of a run's analytics shares: the run's analytics it reads
from, and the statistics it gives, each a method named by its id.
"""
# BN-433.
from typing import Any


class Category:
    """One category of a run's analytics, such as its performance.

    Attributes:
        name: The category, as `result.analytics.portfolio` names it.
        statistics: The ids of the statistics it gives, in order, each a
            method of the same name.
    """

    name = ""
    statistics: tuple[str, ...] = ()

    def __init__(self,
                 analytics: Any):
        self._analytics = analytics

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self.statistics)})"
