# src/beacon/analysis/analytics/category.py
"""
What every category of a run's analytics shares: the run's analytics it reads
from, and the statistics it gives, each a method named by its id.
"""
# BN-433.
from collections.abc import Sequence
from typing import Any

import pandas as pd

# A day, as a date's text or a timestamp; None leaves it to the run.
Day = str | pd.Timestamp | None

# Trailing windows, by name (``1Y``, ``YTD``, ``inception``), one or several.
Windows = str | Sequence[str] | None


class Category:
    """One category of a run's analytics, such as its performance.

    Attributes:
        name: The category, as `result.analytics.portfolio` names it.
        statistics: The ids of the statistics it gives, in order, each a
            method of the same name.
        extras: Its methods that are not a statistic of the catalogue.
    """

    name = ""
    statistics: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()

    def __init__(self,
                 analytics: Any):
        self._analytics = analytics

    def _get(self,
             statistic_id: str,
             **form: Any) -> Any:
        """Statistic *statistic_id* in the form *form*'s keywords ask for;
        those left as None are not given."""
        return self._analytics._value(statistic_id, form)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join((*self.statistics, *self.extras))})"
