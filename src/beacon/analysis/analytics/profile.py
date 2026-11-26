# src/beacon/analysis/analytics/profile.py
"""
One name over a run: `result.analytics.instrument.profile(name)`.

    profile = result.analytics.instrument.profile("AAPL")
    profile.rebalance_weights          # a row a rebalance
    profile.daily_weights              # a row a day
    profile.attribution                # a row a quarter, or as by= says

Each part is a dated table over the analysis window the profile was asked
for (``start``, ``end``), worked out the first time it is asked for.
``Profile.parts`` names them, in order.

``rebalance_weights`` has a row for each of the run's rebalances: the
strategy's weight in the name before any implementation stage (``target``),
after the screens and redistribution (``screened``) and what the trades aimed
at (``weight``), each 0 when that stage did not hold it; the weight a cap cut
it to (``capped``) and the screen that took it out (``removed``), blank when
none did; and each comparator's weight in it that day (``index``,
``optimised``, ``benchmark``, for those the run kept weights for).

``daily_weights`` has a row for each day: the fund's weight in the name as it
drifted between rebalances (the units held times the day's price, over the
NAV), each comparator's, each 0 on a day that portfolio did not hold it, and
the fund's active weight against the comparator (``active``): the one given
as ``against``, else the one the analysis tracks. Its ``attrs`` say which.

``attribution`` has a row for each month, quarter or year (``by``), dated by
its last day: the name's contribution to the fund's return
(``contribution``), its contribution to the fund's return over the
comparator from its active weight (``active_contribution``), and each's
total from the window's start (``contribution_to_date``,
``active_contribution_to_date``), the active ones when the run kept the
comparator's weights. Each day's contribution is as the run's attribution
gives it; the analysis options' ``attribution.linking`` says how the days
are linked. By default (``window``) each period is its share of the whole
window's contribution, so the periods add up to the per-name table's figure
over the window and the last total is it. With ``period`` each period is
linked on its own, as the per-name table over the window from the last day
before the period to its last day gives it, and each total as the table over
the window to the period's end: then the periods do not add up to the
total. Its ``attrs`` say the comparator, the periods and the linking.
"""
# BN-468 (Karan, 2026-11-25): the first of the enhanced Drilldown's
# sections; BN-469..BN-474 add theirs to the same profile.
from functools import cached_property
from typing import Any

import pandas as pd

from ...catalogue import FRACTION
from ..attribution.contribution import linking_factors
from ..panels import REBALANCE_WEIGHTS
from ..reports.context import FUND
from ..reports.options import WINDOW
from ..reports.statistics import RUN_PREDATES, TOO_SHORT, Column, Unavailable
from ..risk.liquidity import BOOKS
from .groupings import FREQUENCIES

# The rebalance table's columns as the profile names them.
STAGES = {"TARGET": "target", "SCREENED": "screened", "WEIGHT": "weight", "CAPPED": "capped",
          "REMOVED": "removed"}
HELD = ("target", "screened", "weight")
ACTIVE = "active"
REMOVED = "removed"
CONTRIBUTION, ACTIVE_CONTRIBUTION = "contribution", "active_contribution"
TO_DATE = "_to_date"

# Each column's label; every column is a weight but the screen's name.
LABELS = {"target": "Target", "screened": "After the screens", "weight": "Traded to",
          "capped": "Capped to", REMOVED: "Removed by", "fund": "Fund", "index": "Index",
          "optimised": "Optimised index", "benchmark": "Benchmark", ACTIVE: "Active",
          CONTRIBUTION: "Contribution", ACTIVE_CONTRIBUTION: "Active contribution",
          CONTRIBUTION + TO_DATE: "Contribution to date",
          ACTIVE_CONTRIBUTION + TO_DATE: "Active contribution to date"}


class Profile:
    """One name over a run: a dated table a part, worked out when first
    asked for."""

    parts = ("rebalance_weights", "daily_weights", "attribution")

    def __init__(self,
                 name: str,
                 context: Any,
                 against: str | None = None,
                 by: str = "quarter"):
        """
        Raises:
            ValueError: If no portfolio of the run held *name*, or *by* is
                not ``month``, ``quarter`` or ``year``.
            Unavailable: If the run lacks the comparator given.
        """
        if by not in FREQUENCIES:
            raise ValueError(f"by is {', '.join(FREQUENCIES)}; not {by!r}.")

        self.name = name
        self.by = by
        self._context = context
        self._against = None if against is None else context.require(against)

        if not any(name in weights.columns for weights in self._weights().values()):
            raise ValueError(f"No portfolio of the run held {name}.")

    @cached_property
    def rebalance_weights(self) -> pd.DataFrame:
        """At each rebalance: the strategy's weight in the name at each
        implementation stage, any cap or screen applied to it, and each
        comparator's weight in it that day.

        Raises:
            Unavailable: If the run was kept before its rebalances were.
        """
        tables = self._context.run.tables

        if REBALANCE_WEIGHTS not in tables:
            raise Unavailable(RUN_PREDATES)

        kept: pd.DataFrame = tables[REBALANCE_WEIGHTS]
        days = self._days()
        dates = pd.DatetimeIndex(pd.unique(kept.index))
        dates = dates[(dates >= days[0]) & (dates <= days[-1])]
        mine = kept.loc[kept["IDENTIFIER"] == self.name, list(STAGES)].rename(columns=STAGES)
        mine = mine[~mine.index.duplicated(keep="last")]

        found = mine.reindex(dates)
        found[list(HELD)] = found[list(HELD)].fillna(0.0)
        # Blank as None, as a stored run's all-blank column reads back NaN.
        found[REMOVED] = found[REMOVED].astype(object).where(found[REMOVED].notna(), None)

        for book, weights in self._weights().items():
            if book != FUND:
                found[book] = _column(weights, self.name).reindex(dates)

        return found.rename_axis("DATE")

    @cached_property
    def daily_weights(self) -> pd.DataFrame:
        """Each day: the fund's and each comparator's weight in the name,
        and the fund's active weight against the comparator."""
        days = self._days()
        found = pd.DataFrame({book: _column(weights, self.name).reindex(days).ffill()
                              for book, weights in self._weights().items()}, index=days)
        against = self._against or self._context.tracked

        if FUND in found and against in found:
            found[ACTIVE] = found[FUND] - found[against]

        found.attrs = {"against": against if ACTIVE in found else None}

        return found.rename_axis("DATE")

    @cached_property
    def attribution(self) -> pd.DataFrame:
        """Each period: the name's contribution to the fund's return and to
        its return over the comparator, and each's total to date, linked as
        the analysis options' ``attribution.linking`` says.

        Raises:
            Unavailable: If the window attributes no day.
        """
        # BN-469 (Karan, 2026-11-26): a share of the window by default, each
        # period on its own as an option.
        context = self._context
        start, end = context.window()
        linking = context.options.attribution.linking
        frequency = FREQUENCIES[self.by]
        daily = context.attribution.by_day(self.name, None, start, end)

        if daily.empty:
            raise Unavailable(TOO_SHORT)

        periods = daily.index.to_period(frequency)
        found = _linked(daily, periods, linking).add_prefix(CONTRIBUTION)
        against = self._against or context.tracked

        if against not in self._weights():
            against = None
        else:
            theirs = context.attribution.by_day(self.name, against, start, end)
            active = _linked(theirs, theirs.index.to_period(frequency), linking)
            found = found.join(active.add_prefix(ACTIVE_CONTRIBUTION))

        # Each period dated by its last day.
        found.index = pd.DatetimeIndex(daily.index.to_series().groupby(periods).max()
                                       .reindex(found.index))
        found.attrs = {"against": against, "by": self.by, "linking": linking}

        return found.rename_axis("DATE")

    def _days(self) -> pd.DatetimeIndex:
        """The analysis window's days, or `Unavailable` when it has none."""
        days = pd.DatetimeIndex(self._context.performance.returns.index)

        if not len(days):
            raise Unavailable(TOO_SHORT)

        return days

    def _weights(self) -> dict[str, pd.DataFrame]:
        """Each portfolio's daily weights the run kept, by portfolio."""
        tables = self._context.run.tables
        found = {book: tables.get(table_name) for book, table_name in BOOKS.items()}

        return {book: weights for book, weights in found.items()
                if weights is not None and not weights.empty}

    def __repr__(self) -> str:
        return f"Profile({self.name!r}: {', '.join(self.parts)})"


def columns_of(table: pd.DataFrame) -> list[Column]:
    """The label and unit of each of a part's columns, in its order."""
    return [Column(str(name), LABELS[str(name)], None if name == REMOVED else FRACTION)
            for name in table.columns]


def _linked(daily: pd.DataFrame,
            periods: pd.PeriodIndex,
            linking: str) -> pd.DataFrame:
    """A name's contribution each period (``""``) and in all to its end
    (``"_to_date"``), from its days (`Attribution.by_day`), linked over the
    whole window or each period on its own."""
    if linking == WINDOW:
        each = (daily[CONTRIBUTION] * _factors(daily)).groupby(periods).sum()

        return pd.DataFrame({"": each, TO_DATE: each.cumsum()})

    found = {}

    for period in periods.unique():
        found[period] = {"": _link(daily[periods == period]),
                         TO_DATE: _link(daily[periods <= period])}

    return pd.DataFrame.from_dict(found, orient="index")


def _link(daily: pd.DataFrame) -> float:
    """The days' contribution, linked over them."""
    return float((daily[CONTRIBUTION] * _factors(daily)).sum())


def _factors(daily: pd.DataFrame) -> pd.Series:
    return linking_factors(daily[FUND], daily.get("comparator"))


def _column(weights: pd.DataFrame,
            name: str) -> pd.Series:
    """*name*'s column of *weights*, 0 where it is blank or absent."""
    if name not in weights.columns:
        return pd.Series(0.0, index=weights.index)

    return weights[name].fillna(0.0)


__all__ = ["ACTIVE", "LABELS", "Profile", "columns_of"]
