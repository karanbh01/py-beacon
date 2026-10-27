# src/beacon/analysis/performance/portfolio.py
"""
What a run held: how many names, and its weight in each group of a grouping
field (sector, region, currency, ...) each day, against its index's and its
benchmark's.

    portfolio = PortfolioAnalytics(RunData.from_record(folder))
    portfolio.group_weights("SECTOR")

A name's group on a day is the one it had at the latest rebalance on or
before that day, as the run's reference table recorded it (the first
rebalance's before it). A name with no group recorded is in ``Unknown``. A
book's group weights add up to its invested weight each day: the fund's cash
is in none of them.
"""
# BN-382.
import numpy as np
import pandas as pd

from ..data import BENCHMARK, INDEX, RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, REFERENCE

# The books, by name, and the tables their daily weights are in.
FUND = "fund"
BOOKS = {FUND: FUND_WEIGHTS, INDEX: INDEX_WEIGHTS, BENCHMARK: BENCHMARK_WEIGHTS}

# The group of a name with none recorded.
UNKNOWN = "Unknown"


class PortfolioAnalytics:
    """What a run held, and in which groups.

    Args:
        run: The run's data.
    """

    def __init__(self,
                 run: RunData):
        self.run = run
        self._groups: dict[str, dict[pd.Timestamp, pd.Series]] = {}

    def weights(self,
                book: str = FUND) -> pd.DataFrame | None:
        """A book's daily weights, dates by name: ``fund``, ``index`` or
        ``benchmark``; None for a book the run did not keep weights for (a
        benchmark given as a level series holds no names)."""
        table = BOOKS[book]

        return self.run.tables.get(table)

    def holdings_count(self,
                       book: str = FUND) -> pd.Series | None:
        """How many names a book held each day."""
        weights = self.weights(book)

        return None if weights is None else (weights.fillna(0.0) > 0).sum(axis=1)

    def group_weights(self,
                      field: str,
                      book: str = FUND) -> pd.DataFrame | None:
        """A book's weight in each group of *field* each day, dates by
        group, the groups sorted; None for a book without weights.

        Raises:
            KeyError: If the run kept no reference data, or none for *field*.
        """
        weights = self.weights(book)

        if weights is None:
            return None

        memberships = self._memberships(field)
        values = weights.fillna(0.0)
        grid = values.to_numpy(dtype=float)
        names = [str(name) for name in values.columns]
        each = {start: membership.reindex(names).fillna(UNKNOWN).to_numpy(dtype=object)
                for start, membership in memberships.items()}
        groups = sorted(pd.unique(np.concatenate(list(each.values()))))
        found = np.zeros((len(values), len(groups)))
        starts = sorted(memberships)
        # Each day's groups are its latest rebalance's, the first's before it.
        which = np.clip(np.searchsorted(np.array(starts, dtype="datetime64[ns]"),
                                        values.index.to_numpy(dtype="datetime64[ns]"),
                                        side="right") - 1, 0, len(starts) - 1)

        for position, start in enumerate(starts):
            rows = which == position

            if not rows.any():
                continue

            # Each name's group as a column of a one-hot matrix, by code.
            codes = pd.Categorical(each[start], categories=groups).codes
            found[rows] = grid[rows] @ np.eye(len(groups))[codes]

        return pd.DataFrame(found, index=values.index, columns=pd.Index(groups, name=field))

    def active_group_weights(self,
                             field: str,
                             against: str = INDEX) -> pd.DataFrame | None:
        """The fund's weight in each group less *against*'s (``index`` or
        ``benchmark``), on the days the fund has; None when the comparator
        holds no names."""
        ours = self.group_weights(field)
        theirs = self.group_weights(field, against)

        if ours is None or theirs is None:
            return None

        groups = sorted(set(ours.columns) | set(theirs.columns))
        theirs = theirs.reindex(index=ours.index, columns=groups).ffill().fillna(0.0)

        return ours.reindex(columns=groups, fill_value=0.0) - theirs

    def _memberships(self,
                     field: str) -> dict[pd.Timestamp, pd.Series]:
        """Each rebalance's group of every name, by date."""
        if field in self._groups:
            return self._groups[field]

        reference = self.run.tables[REFERENCE]

        if field not in reference.columns:
            raise KeyError(f"The run kept no {field} for its names: it has "
                           f"{', '.join(c for c in reference.columns if c != 'IDENTIFIER')}.")

        found = {}

        for date, rows in reference.groupby(level=0):
            found[pd.Timestamp(date)] = (rows.set_index("IDENTIFIER")[field].astype(object)
                                         .where(lambda s: s.notna(), UNKNOWN).astype(str))

        self._groups[field] = found

        return found
