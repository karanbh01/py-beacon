# src/beacon/analysis/risk/read_once.py
"""
A data source read once for a run's factor analytics: every name's closes over
the run, and each feature as of every rebalance, cut to what each rebalance
asks for, which is what the source itself would give.
"""
# BN-397 (closes), BN-411 (features).
from collections.abc import Sequence
from typing import Any

import pandas as pd

from ...data.features import MAX_AGE_DAYS
from ...data.fetcher import DataFetcher


class FeaturesReadOnce:
    """A source's features, whose as-of tables for a set of names on a set of
    dates are read once a field: a read of fewer names on some of the dates
    is cut from it, which is the same answer, since a name's value on a date
    reads only that name's rows; every other read goes to the source."""

    def __init__(self,
                 features: Any,
                 names: Sequence[str],
                 dates: Sequence[pd.Timestamp]):
        self._features = features
        self._names = sorted(set(names))
        self._dates = [pd.Timestamp(day) for day in dates]
        self._tables: dict[tuple[Any, ...], pd.DataFrame] = {}

    def table_as_of(self,
                    identifiers: list[str],
                    field: str,
                    dates: list[pd.Timestamp],
                    feature_type: str | None = None,
                    max_age_days: int | None = MAX_AGE_DAYS) -> pd.DataFrame:
        """The table, cut from the one read when it covers the names and
        dates."""
        asked = [pd.Timestamp(day) for day in dates]
        known = set(self._names)

        if not set(asked) <= set(self._dates) or not set(identifiers) <= known:
            return self._features.table_as_of(identifiers, field, dates, feature_type,
                                              max_age_days)

        key = (field, feature_type, max_age_days)

        if key not in self._tables:
            self._tables[key] = self._features.table_as_of(self._names, field, self._dates,
                                                           feature_type, max_age_days)

        found = self._tables[key].loc[asked, list(identifiers)]
        found.index = pd.DatetimeIndex(asked)

        return found

    def __getattr__(self,
                    name: str) -> Any:
        return getattr(self._features, name)


class PricesReadOnce:
    """A data source whose closes for a set of names, in one currency, over
    one window, are read once; a read of fewer names or a shorter window is
    cut from it as a read of its own would find them (only the days and
    names with a price in it), and every other read goes to the source.
    Its features' as-of tables on the given dates are read once a field
    too (`FeaturesReadOnce`).

    Args:
        source: The data.
        names: The names read once.
        start: The window's first day.
        end: Its last.
        currency: The currency the closes are in.
        dates: The dates features are read as of once.
    """

    def __init__(self,
                 source: DataFetcher,
                 names: Sequence[str],
                 start: pd.Timestamp,
                 end: pd.Timestamp,
                 currency: str,
                 dates: Sequence[pd.Timestamp] = ()):
        self._source = source
        self._names = set(names)
        self._start, self._end = pd.Timestamp(start), pd.Timestamp(end)
        self._currency = currency
        self._closes: pd.DataFrame | None = None
        # BN-411: P/B as of each of 20 rebalances filtered the whole feature
        # table each time, 3.9 s at 6,000 names.
        self.features = FeaturesReadOnce(source.features, names, dates)

    def fetch_prices(self,
                     identifiers: list[str],
                     start_date: str | None = None,
                     end_date: str | None = None,
                     currency: str | None = None,
                     column: str = "CLOSE") -> pd.DataFrame:
        """The closes, cut from the one read when it covers them."""
        covered = (column == "CLOSE" and currency == self._currency
                   and start_date is not None and end_date is not None
                   and pd.Timestamp(start_date) >= self._start
                   and pd.Timestamp(end_date) <= self._end
                   and set(identifiers) <= self._names)

        if not covered:
            return self._source.fetch_prices(identifiers, start_date, end_date,
                                             currency=currency, column=column)

        if self._closes is None:
            self._closes = self._source.fetch_prices(sorted(self._names),
                                                     str(self._start.date()),
                                                     str(self._end.date()),
                                                     currency=self._currency)

        names = [name for name in identifiers if name in self._closes.columns]
        window = self._closes.loc[start_date:end_date, names]

        return window.dropna(how="all", axis=0).dropna(how="all", axis=1)

    def __getattr__(self,
                    name: str) -> Any:
        return getattr(self._source, name)
