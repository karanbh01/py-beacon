# src/beacon/analysis/attribution/analytics.py
"""
A run's attribution: what each name, or each group, contributed to a book's
return, or to the fund's return over a comparator.

Each day a name contributes its weight at the close before times its return
that day (the run's own daily total return for it, in the book's currency),
and the days are linked over the window by Carino's factors so the
contributions add up to the compounded return (see
`beacon.analysis.attribution.contribution`). A group contributes what its
names did, each in the group it was in at that close.

**Active** attribution takes the fund's weights less the comparator's, both
as they drift, against the fund's daily return less the comparator's. What the
names do not explain (cash, costs, fees, futures, the timing of dividends) is
the residual, reported with every result and never folded into a name.
"""
# BN-393.
import pandas as pd

from ...data.base import as_of_position
from ..data import INDEX, RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from ..performance.comparisons import aligned
from .contribution import AttributionResult, attribute

FUND = "fund"
BOOKS = {FUND: FUND_WEIGHTS, "index": INDEX_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS}
UNKNOWN = "Unknown"


class Attribution:
    """What produced a run's returns, by name or group.

    Args:
        run: The run's data, with its weights, name returns and reference.
    """

    def __init__(self,
                 run: RunData):
        self.run = run

    def contribution(self,
                     book: str = FUND,
                     field: str | None = None,
                     start: str | pd.Timestamp | None = None,
                     end: str | pd.Timestamp | None = None) -> AttributionResult:
        """Each name's (or, with *field*, each group's) contribution to a
        book's return (``fund``, ``index`` or ``benchmark``) over the window,
        the whole run by default."""
        returns = self._returns(book)
        weights = self.weights(book)

        return self._attributed(returns, weights, field, start, end)

    def active(self,
               against: str = INDEX,
               field: str | None = None,
               start: str | pd.Timestamp | None = None,
               end: str | pd.Timestamp | None = None) -> AttributionResult:
        """Each name's (or group's) contribution to the fund's daily return
        less *against*'s, from its active weight, over the window."""
        ours, theirs = aligned(self.run.returns, self._levels(against))
        fund, comparator = self.weights(FUND), self.weights(against)
        names = sorted(set(fund.columns) | set(comparator.columns))
        days = fund.index
        active = (fund.reindex(columns=names, fill_value=0.0)
                  - comparator.reindex(index=days, method="ffill")
                  .reindex(columns=names).fillna(0.0))

        return self._attributed(ours - theirs, active, field, start, end)

    def weights(self,
                book: str = FUND) -> pd.DataFrame:
        """A book's daily weights, by name."""
        table = self.run.tables.get(BOOKS[book])

        if table is None or table.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        table = table.sort_index().fillna(0.0)
        table.columns = table.columns.astype(str)

        return table

    def _attributed(self,
                    returns: pd.Series,
                    weights: pd.DataFrame,
                    field: str | None,
                    start: str | pd.Timestamp | None,
                    end: str | pd.Timestamp | None) -> AttributionResult:
        """*returns* attributed to *weights*' names, or to their groups."""
        name_returns = self.run.tables[NAME_RETURNS].sort_index()
        name_returns.columns = name_returns.columns.astype(str)
        # The days the names have returns and the book weights, so by name
        # and by group cover the same days.
        returns = _window(returns.dropna(), start, end)
        returns = returns.loc[returns.index.intersection(name_returns.index)
                              .intersection(weights.index)]

        if field is None:
            return attribute(returns, weights, name_returns)

        lagged = weights.reindex(returns.index, method="ffill").shift(1)
        names = [name for name in lagged.columns if name in name_returns.columns]
        contributions = lagged[names] * name_returns.reindex(returns.index)[names]
        groups = self._groups_by_day(field, lagged.index)
        held = _by_group(lagged[names], groups)
        earned = _by_group(contributions.fillna(0.0), groups)
        group_returns = (earned / held.where(held != 0.0)).fillna(0.0)

        # attribute() takes the weights held at the end of each day and lags
        # them itself, so the lagged group weights go in one day earlier.
        return attribute(returns, held.shift(-1), group_returns)

    def _groups_by_day(self,
                       field: str,
                       days: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Series]:
        """Each day's group of every name, from the rebalance before the day
        the weights were held at (the close before)."""
        reference = self.run.tables[REFERENCE]

        if field not in reference.columns:
            raise KeyError(f"The run kept no {field} for its names.")

        rebalances = pd.DatetimeIndex(sorted(reference.index.get_level_values(0).unique()))
        memberships = {day: reference.loc[day].set_index("IDENTIFIER")[field].astype(str)
                       for day in rebalances}
        found = {}

        for position, day in enumerate(days):
            held_on = days[position - 1] if position else day
            at = as_of_position(rebalances, held_on)
            found[day] = memberships[rebalances[at]] if at is not None else pd.Series(dtype=str)

        return found

    def _returns(self,
                 book: str) -> pd.Series:
        """A book's daily returns: the fund's, or a comparator's from its
        levels."""
        if book == FUND:
            return self.run.returns

        _, theirs = aligned(self.run.returns, self._levels(book))

        return theirs

    def _levels(self,
                book: str) -> pd.Series:
        if book not in self.run.comparators:
            raise KeyError(f"The run has no {book}: it has "
                           f"{', '.join(self.run.comparators) or 'no comparators'}.")

        return self.run.comparators[book]


def _by_group(table: pd.DataFrame,
              groups: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """*table*'s columns added up by each day's group of each name."""
    rows = {}

    for day, row in table.iterrows():
        membership = groups.get(day, pd.Series(dtype=str)).reindex(row.index).fillna(UNKNOWN)
        rows[day] = row.groupby(membership).sum()

    return pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)


def _window(series: pd.Series,
            start: str | pd.Timestamp | None,
            end: str | pd.Timestamp | None) -> pd.Series:
    """*series* from *start* to *end*, both included."""
    if start is not None:
        series = series[series.index >= pd.Timestamp(start)]

    if end is not None:
        series = series[series.index <= pd.Timestamp(end)]

    return series
