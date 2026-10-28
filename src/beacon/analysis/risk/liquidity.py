# src/beacon/analysis/risk/liquidity.py
"""
How quickly a book could be sold: each holding's days to sell, and the share
of the book sellable within 1, 5 and 20 days, at each rebalance and month end.

A holding worth ``V`` is sold at no more than *participation* of the name's
average daily traded value ``ADV`` a day, so it takes

    days = V / (participation x ADV)

and within ``d`` days the book can sell ``sum min(V, d x participation x ADV)``
of itself. A name with no traded value in the window cannot be sold at all
and is counted apart.

The average daily traded value is the close times the volume, in the book's
currency, over the name's last *lookback_days* days with a volume, as the
liquidity screens measure it (`beacon.analysis.snapshots`), read from the
data the run used. Without it, a stored run's own traded values are used,
which the run kept over the screens' 63 days.
"""
# BN-392. Agreed with Karan on 2026-10-27: 20% of 20 days' traded value.
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ...data.fetcher import DataFetcher
from ..data import RunData
from ..panels import BENCHMARK_WEIGHTS, FUND_WEIGHTS, INDEX_WEIGHTS, TRADED_VALUES
from ..snapshots import IDENTIFIER, TRADED_VALUE, traded_value_snapshots

BOOKS = {"fund": FUND_WEIGHTS, "index": INDEX_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS}
PARTICIPATION = 0.2
LOOKBACK_DAYS = 20
HORIZONS = (1, 5, 20)


class LiquidityAnalytics:
    """A run's liquidity at each rebalance and month end.

    Args:
        run: The run's data.
        source: The data the run read, which traded values are measured
            from; without it, the run's own stored traded values.
        participation: The share of a name's daily traded value sold a day.
        lookback_days: The days with a volume its traded value is averaged
            over, when measured from *source*.

    Raises:
        ValueError: If *participation* is not above 0 and at most 1.
    """

    def __init__(self,
                 run: RunData,
                 source: DataFetcher | None = None,
                 participation: float = PARTICIPATION,
                 lookback_days: int = LOOKBACK_DAYS):
        if not 0.0 < participation <= 1.0:
            raise ValueError(f"participation is above 0 and at most 1, got {participation!r}.")

        self.run = run
        self.source = source
        self.participation = participation
        self.lookback_days = lookback_days
        self._traded: pd.DataFrame | None = None

    def dates(self) -> list[pd.Timestamp]:
        """The rebalances and month ends the run kept traded values on."""
        stored = self.run.tables[TRADED_VALUES]

        return sorted(pd.Timestamp(day) for day in stored.index.unique())

    def traded_values(self) -> pd.DataFrame:
        """Each name's average daily traded value on each date, dates by
        name."""
        if self._traded is None:
            if self.source is None:
                long = self.run.tables[TRADED_VALUES]
            else:
                names = sorted({str(name) for book in BOOKS.values()
                                if (table := self.run.tables.get(book)) is not None
                                for name in table.columns})
                long = traded_value_snapshots(self.source, names, self.dates(),
                                              self.run.currency,
                                              lookback_days=self.lookback_days)

            self._traded = (long.reset_index().pivot_table(index=long.index.name or "index",
                                                           columns=IDENTIFIER,
                                                           values=TRADED_VALUE)
                            if not long.empty else pd.DataFrame())

        return self._traded

    def days_to_sell(self,
                     date: str | pd.Timestamp,
                     book: str = "fund") -> pd.DataFrame:
        """Each holding's value, average daily traded value and days to sell
        on *date*, slowest first; infinite for a name with no traded value."""
        day = pd.Timestamp(date)
        values = self._values(day, book)
        traded = self._traded_on(day).reindex(values.index)
        rate = self.participation * traded
        days = (values / rate.where(rate > 0.0)).fillna(np.inf)
        table = pd.DataFrame({"value": values, "traded_value": traded, "days": days})

        return table.sort_values("days", ascending=False)

    def profile(self,
                book: str = "fund",
                horizons: Sequence[int] = HORIZONS) -> pd.DataFrame:
        """At each date: the share of the book sellable within each horizon,
        the days to sell all of it but the names with no traded value, the
        value-weighted mean days, and the share with no traded value."""
        rows = {}

        for day in self.dates():
            table = self.days_to_sell(day, book)
            total = float(table["value"].sum())

            if total <= 0.0:
                continue

            rate = self.participation * table["traded_value"].fillna(0.0)
            row = {f"sellable_{horizon}d": float(np.minimum(table["value"], horizon * rate).sum())
                   / total for horizon in horizons}
            finite = table[np.isfinite(table["days"])]
            row["days_to_sell_all"] = float(finite["days"].max()) if len(finite) else np.inf
            row["mean_days"] = (float((finite["days"] * finite["value"]).sum()
                                      / finite["value"].sum()) if len(finite) else np.inf)
            row["no_traded_value"] = float(table.loc[~np.isfinite(table["days"]), "value"].sum()
                                           / total)
            rows[day] = row

        return pd.DataFrame.from_dict(rows, orient="index")

    def _values(self,
                day: pd.Timestamp,
                book: str) -> pd.Series:
        """The book's holdings' values on *day*, at the fund's net assets."""
        weights = self.run.tables.get(BOOKS[book])

        if weights is None or weights.empty:
            raise KeyError(f"The run kept no weights for its {book}.")

        weights = weights.sort_index()
        at = as_of_position(weights.index, day)
        nav_at = as_of_position(self.run.nav.index, day)

        if at is None or nav_at is None:
            return pd.Series(dtype=float)

        row = weights.iloc[at].fillna(0.0)
        row.index = row.index.astype(str)

        return row[row > 0.0] * float(self.run.nav.iloc[nav_at])

    def _traded_on(self,
                   day: pd.Timestamp) -> pd.Series:
        """Each name's traded value on *day*."""
        traded = self.traded_values()

        if traded.empty or day not in traded.index:
            return pd.Series(dtype=float)

        row = traded.loc[day]
        row.index = row.index.astype(str)

        return row
