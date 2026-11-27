"""A name's profile's return and risk: over the window, each rebalance
period, and each year, quarter and month."""
# BN-477 (Karan, 2026-11-27): every cut in one table, told apart by period;
# no tracking error, which is the fund's against a comparator, not a name's.
# BN-481: every period's figures from running totals over the window's days,
# so each row is a few subtractions rather than a pass over its days.
from typing import Any

import numpy as np
import pandas as pd

from ...attribution.contribution import carino_factor
from ...panels import NAME_RETURNS
from ...performance.formulas import NEGLIGIBLE_SPREAD
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

# A variance this small is no spread, as `name_figures` treats a
# comparator's.
NEGLIGIBLE_VARIANCE = NEGLIGIBLE_SPREAD ** 2
# How many units in the last place of a running total of squares a period's
# sum taken from it can be out by: a variance within that of 0 is rounding,
# a period that did not vary.
ROUNDING = 1024

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


class _Spans:
    """Where each period's days are in a dated series: from ``lo`` to
    ``hi``, exclusive, so a period's sum of a running total is
    ``total[hi] - total[lo]``."""

    def __init__(self,
                 periods: list[Period],
                 index: pd.DatetimeIndex):
        lasts = pd.DatetimeIndex([last for _, _, last in periods])
        self.lo = index.searchsorted(pd.DatetimeIndex([first for _, first, _ in periods]))
        # Past each period's last day, counted in when the index has it.
        self.hi = index.searchsorted(lasts)
        inside = self.hi < len(index)
        self.hi[inside] += index[self.hi[inside]] == lasts[inside]

    def summed(self,
               values: np.ndarray) -> np.ndarray:
        """Each period's sum of *values*, a value a day of the index."""
        return self.bounded(values)[0]

    def bounded(self,
                values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each period's sum of *values*, and how far rounding in the
        running total can put it out when the values are not negative."""
        running = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
        found: np.ndarray = running[self.hi] - running[self.lo]

        return found, ROUNDING * np.finfo(float).eps * running[self.hi]

    def compounded(self,
                   values: np.ndarray,
                   kept: np.ndarray) -> np.ndarray:
        """Each period's compounded return of *values* over the days
        *kept*: its log returns summed, or, over a period with a return of
        -100% or worse, whose logarithm is undefined, its returns'
        product."""
        wiped = kept & (values <= -1.0)
        counted = kept & ~wiped
        logs = np.log1p(np.where(counted, values, 0.0))
        found: np.ndarray = np.expm1(self.summed(logs))

        for row in np.flatnonzero(self.summed(wiped) > 0):
            span = slice(self.lo[row], self.hi[row])
            found[row] = float(np.prod(1.0 + values[span][kept[span]])) - 1.0

        return found

    def moments(self,
                ours: np.ndarray,
                theirs: np.ndarray,
                kept: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Each period's count of days *kept*, the covariance of *ours* with
        *theirs* and each's variance over them, each from the deviations
        from the window's mean, so the running totals stay small."""
        count = self.summed(kept)
        mean_ours = float(ours[kept].mean()) if kept.any() else 0.0
        mean_theirs = float(theirs[kept].mean()) if kept.any() else 0.0
        x = np.where(kept, ours - mean_ours, 0.0)
        y = np.where(kept, theirs - mean_theirs, 0.0)
        sum_x, sum_y = self.summed(x), self.summed(y)
        squares_x, rounding_x = self.bounded(x * x)
        squares_y, rounding_y = self.bounded(y * y)

        with np.errstate(divide="ignore", invalid="ignore"):
            freedom = count - 1.0
            covariance = (self.summed(x * y) - sum_x * sum_y / count) / freedom
            own = (squares_x - sum_x * sum_x / count) / freedom
            other = (squares_y - sum_y * sum_y / count) / freedom
            own = np.where(own > np.maximum(rounding_x / freedom, NEGLIGIBLE_VARIANCE), own, 0.0)
            other = np.where(other > np.maximum(rounding_y / freedom, NEGLIGIBLE_VARIANCE),
                             other, 0.0)

        return count, covariance, own, other


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
        """The name's compounded return and annualised volatility of daily
        returns over each period."""
        spans = _Spans(self.periods, pd.DatetimeIndex(mine.index))
        values = mine.to_numpy(dtype=float)
        has = ~np.isnan(values)
        count, _, variance, _ = spans.moments(values, values, has)
        volatility = np.sqrt(variance * self.periods_per_year)

        return {"return": self._given("return", spans.compounded(values, has), count > 0,
                                      NO_RETURN),
                "volatility": self._given("volatility", volatility, count >= self.fewest,
                                          self.too_short)}

    def against(self,
                mine: pd.Series,
                theirs: pd.Series,
                daily: pd.DataFrame,
                portfolio: str,
                linking: str) -> dict[str, list[float]]:
        """The name against *portfolio* over each period, from its
        returns (*theirs*) and its days' contributions (*daily*,
        `Attribution.by_day`)."""
        spans = _Spans(self.periods, pd.DatetimeIndex(mine.index))
        ours = mine.to_numpy(dtype=float)
        other = theirs.reindex(mine.index).to_numpy(dtype=float)
        both = ~np.isnan(ours) & ~np.isnan(other)
        count, covariance, own, variance = spans.moments(ours, other, both)
        enough = count >= self.fewest
        excess = spans.compounded(ours, both) - spans.compounded(other, both)

        # A portfolio that did not vary leaves nothing to be sensitive to: a
        # beta of 0, as `name_figures` gives it.
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(variance > 0.0, covariance / variance, 0.0)
            correlation = np.where((variance > 0.0) & (own > 0.0),
                                   covariance / np.sqrt(variance * own), np.nan)

        return {"excess_return": self._given(f"excess_return/{portfolio}", excess, count > 0,
                                             NO_SHARED),
                CONTRIBUTION: self._contributions(daily, portfolio, linking),
                "beta": self._given(f"beta/{portfolio}", beta, enough, self.too_short),
                "correlation": self._given(f"correlation/{portfolio}", correlation, enough,
                                           self.too_short),
                "volatility_share": self._shares(daily, portfolio)}

    def _contributions(self,
                       daily: pd.DataFrame,
                       portfolio: str,
                       linking: str) -> list[float]:
        """The name's contribution to *portfolio*'s return over each period,
        linked over the window or each period on its own."""
        spans = _Spans(self.periods, pd.DatetimeIndex(daily.index))
        contributions = daily[CONTRIBUTION].to_numpy(dtype=float)

        if linking == WINDOW:
            return list(spans.summed(contributions * day_factors(daily, portfolio).to_numpy()))

        # Against a comparator the factors are the active ones, which do
        # not sum by period.
        if "comparator" in daily.columns:
            return [linked(daily.iloc[lo:hi], portfolio)
                    for lo, hi in zip(spans.lo, spans.hi, strict=True)]

        # Each day's Carino factor over its period's.
        returns = daily[portfolio]
        scaled = spans.summed(contributions * returns.map(carino_factor).to_numpy(dtype=float))
        totals = spans.compounded(returns.to_numpy(dtype=float),
                                  np.ones(len(returns), dtype=bool))

        return [float(found) / carino_factor(float(total))
                for found, total in zip(scaled, totals, strict=True)]

    def _shares(self,
                daily: pd.DataFrame,
                portfolio: str) -> list[float]:
        """The name's share of the variance of every name's contributions to
        *portfolio* added up, over each period: 0 when it contributed
        nothing."""
        spans = _Spans(self.periods, pd.DatetimeIndex(daily.index))
        contributions = daily[CONTRIBUTION].to_numpy(dtype=float)
        every = np.ones(len(contributions), dtype=bool)
        count, covariance, _, variance = spans.moments(contributions,
                                                       daily[TOTAL].to_numpy(dtype=float), every)
        contributed = spans.summed(contributions != 0.0) > 0
        column = f"volatility_share/{portfolio}"
        found = []

        for row in range(len(self.periods)):
            if not contributed[row]:
                found.append(0.0)
            elif count[row] < self.fewest:
                found.append(self._null(column, self.too_short))
            elif not variance[row] > 0.0:
                found.append(self._null(column, FLAT))
            else:
                found.append(float(covariance[row] / variance[row]))

        return found

    def _given(self,
               column: str,
               values: np.ndarray,
               given: np.ndarray,
               reason: str) -> list[float]:
        """*values* over each period where *given*, else null with *reason*
        noted for *column*; one that comes out null is noted as the
        portfolio not varying."""
        found = []

        for value, kept in zip(values, given, strict=True):
            if not kept:
                found.append(self._null(column, reason))
            elif np.isnan(value):
                found.append(self._null(column, FLAT))
            else:
                found.append(float(value))

        return found

    def _null(self,
              column: str,
              reason: str) -> float:
        """Null, with *reason* noted for *column* unless one already is."""
        self.reasons.setdefault(column, reason)

        return float("nan")
