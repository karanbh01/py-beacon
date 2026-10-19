# src/beacon/data/reference_as_of.py
"""
Reference columns for many names on many dates, each as it stood then.

A weight history asks for the same few columns of the same names on every
rebalance date. Asking the reference data once per date would read it as many
times as there are dates; this reads every validity window of the names once
and answers each (name, date) from them, by the rule
`ReferenceData.get` uses: the row whose ``DATE_FROM`` is on or before the date
and whose ``DATE_TO`` is on or after it, or open.

The columns that say what a series is (``ASSET_CLASS``, ``SECURITY_TYPE`` and
``TRADABLE``) are answered as `beacon.data.kinds` reads them, so a dataset with
no ``ASSET_CLASS`` column answers ``Equity``, as the engine takes it to be.
Any other column the dataset lacks answers None.
"""
# BN-325.
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .fetcher import DataFetcher
from .kinds import ASSET_CLASS, EQUITY, SECURITY_TYPE, TRADABLE, SeriesError, describe

# The columns answered as the engine reads them rather than as stored.
KIND_COLUMNS = (ASSET_CLASS, SECURITY_TYPE, TRADABLE)


class ReferenceAsOf:
    """Reference columns of a set of names, answerable on any date.

    Args:
        fetcher: The loaded data.
        identifiers: The names that will be asked about.
        columns: The columns to answer, as stored (upper case).
    """

    def __init__(self,
                 fetcher: DataFetcher,
                 identifiers: Iterable[str],
                 columns: list[str]):
        self.columns = list(columns)
        names = sorted(set(identifiers))
        frame = fetcher.fetch_reference_data(names) if names else pd.DataFrame()
        # Each window's answer is worked out once, as a plain mapping, so a
        # long history costs a dictionary copy per row rather than a pandas
        # lookup and a classification (measured: 11 ms a date for 500 names
        # before, BN-325).
        self._windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None,
                                            dict[str, Any]]]] = {}

        # Plain records rather than iterrows: a pandas Series per row and a
        # lookup per column were most of the cost (measured, BN-325).
        records = frame.reset_index().to_dict("records") if not frame.empty else []

        for record in records:
            identifier = str(record["IDENTIFIER"])
            end = record.get("DATE_TO")
            self._windows.setdefault(identifier, []).append(
                (pd.Timestamp(record["DATE_FROM"]),
                 None if end is None or pd.isna(end) else pd.Timestamp(end),
                 self._answer(identifier, record)))

        for windows in self._windows.values():
            windows.sort(key=lambda window: window[0])

    def of(self,
           identifier: str,
           date: str | pd.Timestamp) -> dict[str, Any] | None:
        """The columns of *identifier* as they stood on *date*, keyed in
        lower case; None when no reference row was valid then."""
        day = pd.Timestamp(date)
        answer = None

        # The latest-starting valid window wins, as windows are in start order.
        for start, end, values in self._windows.get(identifier, ()):
            if start > day:
                break

            if end is None or end >= day:
                answer = values

        return dict(answer) if answer is not None else None

    def map_of(self,
               identifiers: Iterable[str],
               date: str | pd.Timestamp) -> dict[str, dict[str, Any] | None]:
        """Each of *identifiers*' columns on one date."""
        day = pd.Timestamp(date)

        return {identifier: self.of(identifier, day) for identifier in identifiers}

    def windows_of(self,
                   identifier: str,
                   start: str | pd.Timestamp,
                   end: str | pd.Timestamp) -> list[dict[str, Any]]:
        """The windows of *identifier* that overlap *start* to *end*, each its
        ``date_from``, ``date_to`` (None when open) and ``reference`` columns:
        what a daily panel needs to read any of its dates."""
        first, last = pd.Timestamp(start), pd.Timestamp(end)

        return [{"date_from": str(window_start.date()),
                 "date_to": str(window_end.date()) if window_end is not None else None,
                 "reference": dict(values)}
                for window_start, window_end, values in self._windows.get(identifier, ())
                if window_start <= last and (window_end is None or window_end >= first)]

    def _answer(self,
                identifier: str,
                record: dict[str, Any]) -> dict[str, Any]:
        """One window's columns, keyed in lower case."""
        kinds = (_kinds(identifier, record)
                 if any(column in KIND_COLUMNS for column in self.columns) else {})

        return {column.lower(): kinds[column] if column in kinds else _plain(record.get(column))
                for column in self.columns}


def _kinds(identifier: str,
           record: dict[str, Any]) -> dict[str, Any]:
    """What a reference row says its series is, as the engine reads it; empty
    when it cannot be read, so the stored values are given instead."""
    # A row saying nothing of its kind is an equity a fund can hold: the common
    # case, answered without building a Series to classify.
    if all(_plain(record.get(column)) is None for column in KIND_COLUMNS):
        return {ASSET_CLASS: EQUITY, SECURITY_TYPE: None, TRADABLE: True}

    try:
        described = describe(identifier, pd.Series(record))
    except SeriesError:
        return {}

    return {ASSET_CLASS: described.asset_class, SECURITY_TYPE: described.security_type,
            TRADABLE: described.tradable}


def _plain(value: Any) -> Any:
    """A stored value as JSON can carry it: None for missing, an ISO string
    for a date, a plain number for a numpy one."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value.item() if hasattr(value, "item") else value
