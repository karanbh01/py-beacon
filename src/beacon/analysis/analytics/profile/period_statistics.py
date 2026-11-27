"""A name's profile's return and risk: over the window, each rebalance
period, and each year, quarter and month."""
# BN-477 (Karan, 2026-11-27): every cut in one table, told apart by period;
# no tracking error, which is the fund's against a comparator, not a name's.
from typing import Any

import numpy as np
import pandas as pd

from ...panels import NAME_RETURNS
from ...performance import name_figures
from ...reports.context import FUND
from ...reports.options import WINDOW
from ...reports.statistics import REASONS, RUN_PREDATES, TOO_SHORT, Unavailable
from .columns import CONTRIBUTION, MONTH, QUARTER, REBALANCE, RUN, YEAR
from .contribution import day_factors, linked

CALENDAR = {YEAR: "Y", QUARTER: "Q", MONTH: "M"}
# The figures against each portfolio, in their order.
AGAINST = ("excess_return", CONTRIBUTION, "beta", "correlation", "volatility_share")
TOTAL = "total"

# Why a figure is null, beside a period too short for it.
NO_RETURN = "Some periods have no return for the name."
NO_SHARED = "Some periods have no day with a return for both the name and the portfolio."
FLAT = "The portfolio's return did not vary over some periods."

# A period's cut, its first day and its last.
Period = tuple[str, pd.Timestamp, pd.Timestamp]


def period_statistics(name: str,
                context: Any,
                days: pd.DatetimeIndex,
                rebalances: list[pd.Timestamp],
                portfolios: list[str]) -> pd.DataFrame:
    """*name*'s return and risk over *context*'s window (*days*), each
    period between *rebalances*, and each year, quarter and month, against
    each of *portfolios* the run kept returns for.

    Raises:
        Unavailable: If the run was kept before its names' returns were.
    """
    tables = context.run.tables

    if NAME_RETURNS not in tables:
        raise Unavailable(RUN_PREDATES)

    returns = tables[NAME_RETURNS]
    returns = returns.set_axis(returns.columns.astype(str), axis=1)
    mine = returns[name].reindex(days) if name in returns.columns else pd.Series(np.nan, index=days)
    periods = cut(days, rebalances)
    fewest = context.performance.min_risk_days
    linking = context.options.attribution.linking
    rows = _Rows(periods, context.performance.periods_per_year, fewest)
    start, end = context.window()

    found = pd.DataFrame({"period": [period for period, _, _ in periods],
                          "start": [start for _, start, _ in periods]},
                         index=pd.DatetimeIndex([end for _, _, end in periods], name="DATE"))
    found = found.assign(**rows.own(mine))

    for portfolio in portfolios:
        if portfolio != FUND and portfolio not in context.comparators:
            continue

        theirs = context.performance_of(portfolio).returns
        daily = context.attribution.by_day(name, None, start, end, of=portfolio, total=True)
        figures = rows.against(mine, theirs, daily, portfolio, linking)
        found = found.assign(**{f"{figure}/{portfolio}": figures[figure] for figure in AGAINST})

    found.attrs = {"linking": linking, "min_risk_days": fewest, "unavailable": rows.reasons}

    return found


def cut(days: pd.DatetimeIndex,
        rebalances: list[pd.Timestamp]) -> list[Period]:
    """The periods of *days*, in order: all of them, then each period
    between *rebalances*, then each year, quarter and month, each from its
    first day in *days* to its last.

    A rebalance period runs from the day after a rebalance to the next
    rebalance inclusive, as the fund trades at a rebalance's close; the days
    before the first rebalance in *days* end at it, except the first day
    alone, when it is a rebalance.
    """
    found: list[Period] = [(RUN, days[0], days[-1])]
    after = pd.DatetimeIndex(sorted(rebalances)).searchsorted(days, side="left")
    groups = _grouped(days, pd.Index(after))

    # The window starting on a rebalance: that day's return was earned on
    # what was held before the window, and its contribution is not
    # attributed (`Attribution.by_day`), so it is no period of its own.
    if len(groups) > 1 and groups[0][0] == groups[0][1] and days[0] in rebalances:
        groups = groups[1:]

    found += [(REBALANCE, first, last) for first, last in groups]

    for period, frequency in CALENDAR.items():
        found += [(period, first, last)
                  for first, last in _grouped(days, days.to_period(frequency))]

    return found


def _grouped(days: pd.DatetimeIndex,
             labels: pd.Index) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The first and last of *days* with each label, in order."""
    spans = pd.Series(days, index=days).groupby(labels.to_numpy(), sort=True).agg(["min", "max"])

    return [(pd.Timestamp(first), pd.Timestamp(last))
            for first, last in zip(spans["min"], spans["max"], strict=True)]


class _Rows:
    """Each figure over each period, with why any is null."""

    def __init__(self,
                 periods: list[Period],
                 periods_per_year: int,
                 fewest: int):
        self.periods = periods
        self.periods_per_year = periods_per_year
        self.fewest = fewest
        self.reasons: dict[str, str] = {}
        self.too_short = REASONS[TOO_SHORT][1]

    def own(self,
            mine: pd.Series) -> dict[str, list[float]]:
        """The name's compounded return and annualised volatility over each
        period."""
        found: dict[str, list[float]] = {"return": [], "volatility": []}

        for _, start, end in self.periods:
            within = mine[(mine.index >= start) & (mine.index <= end)].to_frame()
            count = int(within.notna().sum().iloc[0])

            found["return"].append(self._figure("total_return", within, None, "return",
                                                count > 0, NO_RETURN))
            found["volatility"].append(self._figure("volatility", within, None, "volatility",
                                                    count >= self.fewest, self.too_short))

        return found

    def against(self,
                mine: pd.Series,
                theirs: pd.Series,
                daily: pd.DataFrame,
                portfolio: str,
                linking: str) -> dict[str, list[float]]:
        """The name against *portfolio* over each period, from its
        returns (*theirs*) and its days' contributions (*daily*,
        `Attribution.by_day`)."""
        found: dict[str, list[float]] = {figure: [] for figure in AGAINST}
        scaled = daily[CONTRIBUTION] * day_factors(daily, portfolio)

        for _, start, end in self.periods:
            within = mine[(mine.index >= start) & (mine.index <= end)].to_frame()
            both = int((within.iloc[:, 0].notna() & theirs.reindex(within.index).notna()).sum())
            days = daily[(daily.index >= start) & (daily.index <= end)]

            found["excess_return"].append(self._figure("excess_return", within, theirs,
                                                       f"excess_return/{portfolio}", both > 0,
                                                       NO_SHARED))

            for figure in ("beta", "correlation"):
                found[figure].append(self._figure(figure, within, theirs, f"{figure}/{portfolio}",
                                                  both >= self.fewest, self.too_short))

            found[CONTRIBUTION].append(float(scaled[days.index].sum()) if linking == WINDOW
                                       else linked(days, portfolio))
            found["volatility_share"].append(self._share(days, portfolio))

        return found

    def _share(self,
               days: pd.DataFrame,
               portfolio: str) -> float:
        """The name's share of the variance of every name's contributions to
        *portfolio* added up, over *days*: 0 when it contributed nothing."""
        column = f"volatility_share/{portfolio}"

        if not (days[CONTRIBUTION] != 0.0).any():
            return 0.0

        if len(days) < self.fewest:
            return self._null(column, self.too_short)

        variance = float(days[TOTAL].var())

        if not variance > 0.0:
            return self._null(column, FLAT)

        return float(days[CONTRIBUTION].cov(days[TOTAL])) / variance

    def _figure(self,
                figure: str,
                within: pd.DataFrame,
                theirs: pd.Series | None,
                column: str,
                given: bool,
                reason: str) -> float:
        """Figure *figure* of the name over *within* (`name_figures`) when
        *given*, else null with *reason* noted for *column*; one that comes
        out null is noted as the portfolio not varying."""
        if not given:
            return self._null(column, reason)

        value = float(name_figures.figure(figure, within, theirs,
                                          self.periods_per_year).iloc[0])

        return value if not np.isnan(value) else self._null(column, FLAT)

    def _null(self,
              column: str,
              reason: str) -> float:
        """Null, with *reason* noted for *column* unless one already is."""
        self.reasons.setdefault(column, reason)

        return float("nan")
