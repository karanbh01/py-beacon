# src/beacon/analysis/analytics/profile/contribution.py
"""A name's profile's contribution to the return, by period."""
# BN-469 (Karan, 2026-11-26): a share of the window by default, each period
# on its own as an option.
from typing import Any

import pandas as pd

from ...attribution.contribution import linking_factors
from ...reports.context import FUND
from ...reports.options import WINDOW
from ...reports.statistics import TOO_SHORT, Unavailable
from ..groupings import FREQUENCIES
from .columns import ACTIVE_CONTRIBUTION, CONTRIBUTION, REBALANCE, TO_DATE


def attribution(name: str,
                context: Any,
                by: str,
                against: str | None,
                rebalances: list[pd.Timestamp]) -> pd.DataFrame:
    """Each month, quarter, year or rebalance period (*by*) of *context*'s
    window: *name*'s contribution to the fund's return and to its return
    over *against*, and each's total to date, linked as the analysis
    options' ``attribution.linking`` says. A rebalance period runs from the
    day after one of *rebalances* to the next inclusive.

    Raises:
        Unavailable: If the window attributes no day.
    """
    start, end = context.window()
    linking = context.options.attribution.linking
    daily = context.attribution.by_day(name, None, start, end)

    if daily.empty:
        raise Unavailable(TOO_SHORT)

    periods = _periods(daily.index, by, rebalances)
    found = _linked(daily, periods, linking).add_prefix(CONTRIBUTION)

    if against is not None:
        theirs = context.attribution.by_day(name, against, start, end)
        active = _linked(theirs, _periods(theirs.index, by, rebalances), linking)
        found = found.join(active.add_prefix(ACTIVE_CONTRIBUTION))

    # Each period dated by its last day.
    found.index = pd.DatetimeIndex(daily.index.to_series().groupby(periods).max()
                                   .reindex(found.index))
    found.attrs = {"against": against, "by": by, "linking": linking}

    return found.rename_axis("DATE")


def _periods(days: pd.DatetimeIndex,
             by: str,
             rebalances: list[pd.Timestamp]) -> pd.Index:
    """The period each of *days* is in, in order."""
    if by != REBALANCE:
        return days.to_period(FREQUENCIES[by])

    # BN-487: as `period_statistics.cut` cuts them, each day in the period
    # ending at the first rebalance on or after it; the window's first day,
    # a rebalance or not, is not attributed (`Attribution.by_day`).
    return pd.Index(pd.DatetimeIndex(sorted(rebalances)).searchsorted(days, side="left"))


def _linked(daily: pd.DataFrame,
            periods: pd.Index,
            linking: str) -> pd.DataFrame:
    """A name's contribution each period (``""``) and in all to its end
    (``"_to_date"``), from its days (`Attribution.by_day`), linked over the
    whole window or each period on its own."""
    if linking == WINDOW:
        each = (daily[CONTRIBUTION] * day_factors(daily)).groupby(periods).sum()

        return pd.DataFrame({"": each, TO_DATE: each.cumsum()})

    found = {}

    for period in periods.unique():
        found[period] = {"": linked(daily[periods == period]),
                         TO_DATE: linked(daily[periods <= period])}

    return pd.DataFrame.from_dict(found, orient="index")


def linked(daily: pd.DataFrame,
           of: str = FUND) -> float:
    """The days' contribution (`Attribution.by_day`) to portfolio *of*'s
    return, linked over them."""
    return float((daily[CONTRIBUTION] * day_factors(daily, of)).sum())


def day_factors(daily: pd.DataFrame,
                of: str = FUND) -> pd.Series:
    """The Carino factor each of the days (`Attribution.by_day`) is
    scaled by, linked over them on portfolio *of*'s returns."""
    # BN-477: of, for the comparators' contributions in period_statistics.
    return linking_factors(daily[of], daily.get("comparator"))
