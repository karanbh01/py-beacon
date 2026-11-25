# src/beacon/analysis/analytics/groupings.py
"""
Tables split by groups of their own, which ``by`` asks for beside the
calendar periods every window statistic takes:

    trading.costs(by="reason")                 # a row a reason trades were made
    trading.trade_counts(by="reason")
    flows.dealing_costs(by="year")             # a row a year and kind of flow
    attribution.brinson(by="month")            # the effects each month, linked within it
    risk.herfindahl(by="year")                 # each year's mean daily figures
    risk.risk_contributions(by="sector")       # added up by sector at the last rebalance
    performance.total_return(by="month", grid=True)   # years by months

What ``by`` takes for each is the catalogue's (`GROUPINGS`).
"""
# BN-446 (Karan, 2026-11-24): the calculator methods no statistic called.
from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd

from ..performance.tables import DIFFERENCE
from ..reports.catalogue.forms import BY, FIELD, GROUPINGS, OF
from ..reports.context import FUND, Context
from ..reports.statistics import NO_FLOWS, NO_GROUPING, StatisticSpec, Unavailable
from .shape import ALL

# The calendar periods, as each calculation spells them.
FREQUENCIES = {"month": "M", "quarter": "Q", "year": "Y"}

# The portfolio a grid of the fund less its tracked comparator is of.
ACTIVE = "active"


def grouped(statistic: StatisticSpec,
            form: Mapping[str, Any],
            context: Context) -> Any:
    """*statistic* split by ``by``'s groups.

    Raises:
        ValueError: If ``by`` is not one the statistic is split by.
        Unavailable: If the run cannot give it.
    """
    group = str(form[BY])
    takes = GROUPINGS[statistic.id]

    if FIELD in takes:
        return CALCULATIONS[statistic.id](context, group.upper())

    if group not in takes:
        raise ValueError(f"{statistic.id} by is one of {', '.join(takes)}, got {group!r}.")

    return CALCULATIONS[statistic.id](context, group)


def grid(form: Mapping[str, Any],
         context: Context) -> pd.DataFrame:
    """The monthly returns of ``of`` as a calendar grid, a row a year, with
    each year's: the fund's, a comparator's, or the fund's less its tracked
    comparator's (``active``).

    Raises:
        ValueError: If not by month, or of every portfolio.
    """
    if form.get(BY) != "month":
        raise ValueError("grid goes with by='month'.")

    of = form.get(OF, FUND)

    if of == ALL:
        raise ValueError("A grid is one portfolio's: give of='fund', a comparator or 'active'.")

    if of == ACTIVE:
        return context.performance.monthly_grid(f"{DIFFERENCE} {context.require_tracked()}")

    if of != FUND:
        context.require(of)

    return context.performance.monthly_grid(of)


def _costs(context: Context,
           group: str) -> pd.DataFrame:
    if group == "reason":
        return context.portfolio.costs_by_reason()

    return context.portfolio.costs("year")


def _trade_counts(context: Context,
                  group: str) -> pd.DataFrame:
    return context.portfolio.trading_by_reason()


def _dealing_costs(context: Context,
                   group: str) -> pd.DataFrame:
    if context.run.investor_flows is None:
        raise Unavailable(NO_FLOWS)

    return context.portfolio.dealing_summary("year" if group == "year" else "run")


def _brinson(context: Context,
             group: str) -> pd.DataFrame:
    try:
        return context.attribution.brinson_by_period(context.options.attribution.grouping,
                                                     context.require_tracked(),
                                                     FREQUENCIES[group])
    except KeyError as error:
        raise Unavailable(NO_GROUPING) from error


def _herfindahl(context: Context,
                group: str) -> pd.DataFrame:
    columns = ["names", "herfindahl", "normalised_herfindahl", "effective_number"]
    table = context.concentration.by_period(FREQUENCIES[group])

    return table[[column for column in columns if column in table.columns]]


def _risk_contributions(context: Context,
                        field: str) -> pd.DataFrame:
    over_time = context.risk_over_time
    dates = over_time.dates()

    if not len(dates):
        raise ValueError("The run has no rebalance the risk model was fitted for.")

    found = over_time.group_contributions(dates[-1], field, against=context.tracked)

    if found.index.isin(["Unknown"]).all():
        raise ValueError(f"The run's names were not kept with a {field} field.")

    return found


# Each statistic's split, given the group (or, for a field, its column name).
CALCULATIONS: dict[str, Callable[[Context, str], Any]] = {
    "costs": _costs,
    "trade_counts": _trade_counts,
    "dealing_costs": _dealing_costs,
    "brinson": _brinson,
    "herfindahl": _herfindahl,
    "risk_contributions": _risk_contributions,
}


__all__ = ["ACTIVE", "CALCULATIONS", "grid", "grouped"]
