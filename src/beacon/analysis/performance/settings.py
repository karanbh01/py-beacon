# src/beacon/analysis/performance/settings.py
"""
What a run's analysis is measured over and against.
"""
# BN-371.
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AnalysisSettings:
    """How a run's figures are measured. Each setting left as None takes the
    run's own.

    Attributes:
        start: The first day measured; the run's first when None. A later
            start measures from the previous day's close.
        end: The last day measured; the run's last when None.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate the Sharpe ratio is in excess of.
        tracked: The comparator tracking is measured against, by name
            (``index``, ``optimised`` or ``benchmark``): the index the run
            traded toward when None, which is the optimised index when it
            solved one.
    """
    start: pd.Timestamp | str | None = None
    end: pd.Timestamp | str | None = None
    periods_per_year: int | None = None
    risk_free_rate: float | None = None
    tracked: str | None = None
