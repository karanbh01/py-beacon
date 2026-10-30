# src/beacon/analysis/performance/levels.py
"""
Level series as they are shown: rebased to 100, their daily returns, their
returns by calendar year, and several on one axis over the dates they share.
"""
# BN-401: moved from the server's views, which now call these.
from dataclasses import dataclass

import pandas as pd

from ...exceptions import DataNotFoundError
from . import formulas

# Every series is rebased to this so a book and its comparators start
# together and can be read off the same axis.
BASE_LEVEL = 100.0


def rebased(series: pd.Series,
            base: float = BASE_LEVEL) -> pd.Series:
    """*series* rebased to *base* at its first value; as it is when empty or
    when that value is 0."""
    if series.empty:
        return series

    first = series.iloc[0]

    if first == 0:
        return series

    return series / first * base


def daily_returns(levels: pd.Series) -> pd.Series:
    """Each day's return of *levels*, from the day before; none for the first."""
    return levels.pct_change().dropna()


def total_return(levels: pd.Series) -> float:
    """The last level over the first, less one; 0 when there is none or the
    first is 0."""
    if not len(levels) or not levels.iloc[0]:
        return 0.0

    return float(levels.iloc[-1] / levels.iloc[0] - 1)


def headline(levels: pd.Series,
             periods_per_year: int,
             risk_free_rate: float) -> dict[str, float]:
    """The five headline figures of a level series alone; see
    `formulas.headline_figures`."""
    return formulas.headline_figures(daily_returns(levels), total_return(levels), levels,
                                     periods_per_year, risk_free_rate)


def annual_returns(level: pd.Series) -> dict[str, float]:
    """Return per calendar year, keyed by the year.

    Each year runs from the previous year's closing level to its own, so
    the product of (1 + r) telescopes to ``last / first - 1``. The first
    value is the starting point, not a year of its own: a run that starts on
    2 January begins from the capital held on the eve, which may fall in the
    year before.

    Args:
        level: The level series, indexed by date.

    Returns:
        dict: Year (as a string) -> return for that year.
    """
    if level.empty:
        return {}

    later = level.iloc[1:]
    closes = later.groupby(later.index.year).last()
    returns: dict[str, float] = {}
    previous = level.iloc[0]

    for year, close in closes.items():
        returns[str(year)] = float(close / previous - 1.0)
        previous = close

    return returns


@dataclass(frozen=True)
class LevelComparison:
    """Several level series over the dates they all have, each rebased to
    100 on the first.

    Attributes:
        window: The shared dates.
        levels: Each series over them, rebased.
        total_returns: Each series' return over them.
    """
    window: pd.DatetimeIndex
    levels: dict[str, pd.Series]
    total_returns: dict[str, float]


def compared(levels: dict[str, pd.Series]) -> LevelComparison:
    """*levels* on one axis, over the dates every one of them has.

    Intersected rather than clipped to the latest start and earliest end, so
    a gap in one series drops that date for all of them instead of leaving a
    hole only one line has.

    Raises:
        DataNotFoundError: If they share no date.
    """
    window = _shared_dates(levels)
    shown = {name: rebased(series.loc[window]) for name, series in levels.items()}

    return LevelComparison(window=window, levels=shown,
                           total_returns={name: float(series.iloc[-1] / series.iloc[0] - 1.0)
                                          for name, series in shown.items()})


def _shared_dates(levels: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """The dates every series has, in order."""
    shared: pd.DatetimeIndex | None = None

    for series in levels.values():
        index = pd.DatetimeIndex(series.index)
        shared = index if shared is None else shared.intersection(index)

    if shared is None or shared.empty:
        raise DataNotFoundError(
            "a window these indices share",
            source="their level series do not overlap on any date")

    return shared.sort_values()
