# src/beacon/analysis/performance/tables.py
"""
A run's figures laid out as tables: the calendar grid of monthly returns with
each year's, the figures for each quarter or year, and the trailing returns,
each for the fund, its comparators and the differences.

Differences are the fund's figure less the comparator's, in percentage
points, as a period table shows them. A period's return compounds its days,
so a year's months compound to the year, and the years to the whole run.
"""
# BN-379.
import calendar
from collections.abc import Mapping

import pandas as pd

from . import formulas
from .comparisons import aligned, periods_of
from .return_figures import WindowReturn

# The grid's columns: the months, then the year.
MONTHS = [calendar.month_abbr[month] for month in range(1, 13)]
YEAR = "Year"

# The figures a period table gives for the fund and for each comparator.
PERIOD_FIGURES = ("total_return", "volatility", "sharpe_ratio", "max_drawdown")

# Who a table's columns are for: the fund and the differences from a
# comparator.
FUND, DIFFERENCE = "fund", "difference"


def monthly_grid(returns: pd.Series) -> pd.DataFrame:
    """Each month's compounded return, a row a year and a column a month,
    with the year's own return (to date, for the last) in ``Year``. A month
    the run did not cover is blank."""
    if returns.empty:
        return pd.DataFrame(columns=[*MONTHS, YEAR])

    months = formulas.period_returns(returns, formulas.MONTHLY)
    years = formulas.period_returns(returns, formulas.ANNUAL)
    grid = pd.DataFrame(index=years.index, columns=[*MONTHS, YEAR], dtype=float)

    for label, value in months.items():
        year, month = str(label).split("-")
        grid.loc[year, MONTHS[int(month) - 1]] = value

    grid[YEAR] = years
    grid.index.name = "year"

    return grid


def difference_grid(ours: pd.Series,
                    theirs: pd.Series) -> pd.DataFrame:
    """The fund's monthly grid less a comparator's, cell by cell, the two on
    the same days."""
    return monthly_grid(ours) - monthly_grid(theirs)


def period_table(returns: pd.Series,
                 frequency: str,
                 periods_per_year: int,
                 risk_free_rate: float,
                 comparators: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Each quarter's or year's figures (return, volatility, Sharpe ratio,
    maximum drawdown) for the fund, with its share of positive days, and for
    each comparator (given by its daily returns, its first day's nothing),
    and the fund's less each comparator's: a row a period, columns by whom
    and figure."""
    rows: dict[str, dict[tuple[str, str], float | None]] = {}

    for period in periods_of(returns, frequency, periods_per_year, risk_free_rate,
                             comparators):
        row: dict[tuple[str, str], float | None] = {
            (FUND, name): period.figures[name] for name in (*PERIOD_FIGURES, "positive_days")}

        for name, against in period.against.items():
            for figure in PERIOD_FIGURES:
                theirs = against["own"][figure] if against else None
                row[(name, figure)] = theirs
                row[(f"{DIFFERENCE} {name}", figure)] = (
                    period.figures[figure] - theirs if theirs is not None else None)

        rows[period.period] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["of", "figure"])
    table.index.name = frequency

    return table


def trailing_table(fund: Mapping[str, WindowReturn | None],
                   comparators: Mapping[str, Mapping[str, WindowReturn | None]]) -> pd.DataFrame:
    """The trailing returns as a table: a row a window, the total and
    annualised return for the fund, each comparator and the fund's less each
    comparator's. A difference is given only when the two cover the same
    days: a comparator that starts after the run has a shorter inception."""
    rows: dict[str, dict[tuple[str, str], float | None]] = {}

    for window, ours in fund.items():
        row: dict[tuple[str, str], float | None] = {(FUND, "total"): _total(ours),
                                                     (FUND, "annualised"): _annualised(ours)}

        for name, trailing in comparators.items():
            theirs = trailing.get(window)
            same = (ours is not None and theirs is not None
                    and (ours.start, ours.end) == (theirs.start, theirs.end))
            row[(name, "total")] = _total(theirs)
            row[(name, "annualised")] = _annualised(theirs)
            row[(f"{DIFFERENCE} {name}", "total")] = (_less(_total(ours), _total(theirs))
                                                      if same else None)
            row[(f"{DIFFERENCE} {name}", "annualised")] = (
                _less(_annualised(ours), _annualised(theirs)) if same else None)

        rows[window] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.columns = pd.MultiIndex.from_tuples(table.columns, names=["of", "return"])
    table.index.name = "window"

    return table


def month_of_year(returns: pd.Series) -> pd.Series:
    """The mean return of each calendar month over the years, January to
    December; blank for a month the run never covered."""
    grid = monthly_grid(returns)

    return grid[MONTHS].astype(float).mean() if not grid.empty else pd.Series(
        index=MONTHS, dtype=float)


def comparator_returns(returns: pd.Series,
                       levels: Mapping[str, pd.Series]) -> dict[str, tuple[pd.Series, pd.Series]]:
    """For each comparator, the fund's and its daily returns on the days both
    have (`beacon.analysis.performance.comparisons.aligned`); a comparator
    sharing fewer than two days is left out."""
    found = {}

    for name, series in levels.items():
        ours, theirs = aligned(returns, series)

        if not ours.empty:
            found[name] = (ours, theirs)

    return found


def _total(window: WindowReturn | None) -> float | None:
    return window.total if window is not None else None


def _annualised(window: WindowReturn | None) -> float | None:
    return window.annualised if window is not None else None


def _less(ours: float | None,
          theirs: float | None) -> float | None:
    return ours - theirs if ours is not None and theirs is not None else None
