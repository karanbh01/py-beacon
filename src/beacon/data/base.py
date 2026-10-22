"""
Base data containers for market and reference data.
"""

import logging
import os
from collections.abc import Collection

import numpy as np
import pandas as pd


def _read_file(file_path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame based on extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv, .xls, or .xlsx.")



def as_of_position(index: pd.Index,
                   date: str | pd.Timestamp) -> int | None:
    """Where the last observation at or before *date* sits, or None.

    The one way a point-in-time lookup is done, so that look-ahead is not
    something a caller has to remember to avoid.

    The natural expression, `index.searchsorted(date, side="right") - 1`,
    returns **-1** when every observation is dated after *date*, and in
    pandas -1 is a legal index meaning *the last element*: the newest
    observation, the largest look-ahead available. Clamping to 0 instead
    still returns a future value. This returns None, which a caller cannot
    accidentally index with.

    Args:
        index: Ascending dates. Sorting is the caller's business; a search
            over an unsorted index is meaningless under any convention.
        date: The point in time being asked about.

    Returns:
        int | None: Position of the last entry at or before *date*, or None
        when the index begins after it.
    """
    position = int(index.searchsorted(pd.Timestamp(date), side="right"))

    return None if position == 0 else position - 1


logger = logging.getLogger(__name__)

# The columns a price is read from, which a price at or below zero cannot be.
PRICE_COLUMNS = ("OPEN", "HIGH", "LOW", "CLOSE")

# Names a warning lists before saying how many more there are.
LISTED_NAMES = 5


def drop_non_positive_prices(frame: pd.DataFrame,
                             exempt: Collection[str] = ()) -> None:
    """Read a price at or below zero as no price at all, in place.

    *exempt* names the series whose `CLOSE` is not a price, such as a short
    rate, which may well be below zero.

    A price cannot be zero or negative, and one that is would fail a backtest
    the moment it is traded and turn an index's market values negative.
    Treated as a missing bar instead, every reader handles it as one: the last
    good close is carried and the day recorded as a price gap. One warning per
    load names the identifiers and each one's first such date.
    """
    # BN-334: a synthetic store generated before 0.9.5 can hold such prices,
    # where a fat-tailed return below -100% turned a name's path negative.
    # Imports and Postgres stores are refused at load for them instead.
    columns = [column for column in PRICE_COLUMNS
               if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column])]

    if not columns:
        return

    bad = frame[columns] <= 0

    if exempt:
        bad[frame.index.get_level_values("IDENTIFIER").isin(list(exempt))] = False

    if not bad.to_numpy().any():
        return

    rows = bad.any(axis=1)
    first = (rows[rows].reset_index().groupby("IDENTIFIER")["DATE"].min())
    listed = ", ".join(f"{name} from {date:%Y-%m-%d}"
                       for name, date in first.iloc[:LISTED_NAMES].items())
    more = len(first) - LISTED_NAMES
    logger.warning("%d bar(s) of %d identifier(s) have a price at or below zero, read as "
                   "missing: %s%s.", int(rows.sum()), len(first), listed,
                   f" and {more} more" if more > 0 else "")

    for column in columns:
        frame.loc[bad[column], column] = np.nan


class MarketData:
    """Time-series data container backed by a MultiIndex DataFrame.

    The source file must contain at least ``IDENTIFIER`` and ``DATE`` columns.
    After loading the DataFrame is indexed on ``(IDENTIFIER, DATE)`` and sorted,
    enabling fast ``.loc`` slicing by identifier or list of identifiers.
    """

    def __init__(self,
                 file_path: str,
                 date_format: str = "%Y-%m-%d"):
        df = _read_file(file_path)
        self._df = self._prepare(df, date_format)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame,
                       date_format: str = "%Y-%m-%d") -> "MarketData":
        """Create a MarketData instance from an existing DataFrame."""
        instance = object.__new__(cls)
        instance._df = cls._prepare(df.copy(), date_format)
        return instance

    @staticmethod
    def _prepare(df: pd.DataFrame,
                 date_format: str) -> pd.DataFrame:
        missing = [c for c in ("IDENTIFIER", "DATE") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        df["DATE"] = pd.to_datetime(df["DATE"], format=date_format)
        df = df.set_index(["IDENTIFIER", "DATE"]).sort_index()
        return df

    def drop_non_positive_prices(self,
                                 exempt: Collection[str] = ()) -> None:
        """Read every price at or below zero as missing; see the function."""
        drop_non_positive_prices(self._df, exempt)

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

    @property
    def identifiers(self) -> list[str]:
        """Unique identifiers present in the dataset.

        Cached on the frame's identity, so the scan of the whole MultiIndex
        happens once per frame. `fetch_fx_rates` consults this on every call
        to decide whether a pair exists.
        """
        # Why cached: the calculator makes that FX call once per foreign
        # holding per day. Against a single-currency universe that never
        # fired; against a global one it turned an O(rows) scan into an inner
        # loop, and an index over eighty names took longer than the entire
        # rest of the test suite.
        #
        # Keyed on the frame's identity rather than a flag, so replacing `_df`
        # invalidates it automatically instead of relying on every future
        # mutation remembering to.
        cached: tuple[object, list[str]] | None = getattr(
            self, "_identifier_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1]

        values = [str(value) for value
                  in self._df.index.get_level_values("IDENTIFIER").unique()]
        self._identifier_cache = (self._df, values)

        return values

    @property
    def identifier_set(self) -> frozenset[str]:
        """:attr:`identifiers` as a set, cached the same way."""
        cached: tuple[object, frozenset[str]] | None = getattr(self, "_identifier_set", None)

        if cached is not None and cached[0] is self._df:
            return cached[1]

        values = frozenset(self.identifiers)
        self._identifier_set = (self._df, values)

        return values

    @property
    def columns(self) -> list[str]:
        """Non-index column names.

        Cached on the frame's identity, like `identifiers`, because it is
        consulted on every price read.
        """
        # BN-214. `_market_scalar` asks `column not in market.columns` on every
        # price read (186,400 times over a 200-name three-year run), and this
        # built a fresh list of strings each time to answer a membership test.
        cached: tuple[object, list[str]] | None = getattr(
            self, "_columns_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1]

        values = [str(name) for name in self._df.columns]
        self._columns_cache = (self._df, values)

        return values

    @property
    def sessions(self) -> pd.DatetimeIndex:
        """The distinct dates the dataset carries, ascending.

        Cached on the frame's identity, like `identifiers`. Materialising the
        DATE level is an O(rows) take over the whole frame, and `date_range`
        and `last_session_on_or_before` both read it.
        """
        # BN-190. Before the cache, `date_range` and `last_session_on_or_before`
        # each materialised the level on every call, which `resolve_session`
        # makes once *per name* while a methodology walks a universe. On a
        # 1,600-name preview that was 59% of the runtime spent re-deriving a
        # constant: every name on a given date resolves to the same session,
        # and the frame does not move underneath them.
        cached: tuple[object, pd.DatetimeIndex] | None = getattr(
            self, "_session_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1]

        values = pd.DatetimeIndex(
            self._df.index.get_level_values("DATE")).unique().sort_values()
        self._session_cache = (self._df, values)

        return values

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest, latest) timestamps in the dataset."""
        sessions = self.sessions

        if not len(sessions):
            return pd.NaT, pd.NaT

        return pd.Timestamp(sessions[0]), pd.Timestamp(sessions[-1])

    def last_session_on_or_before(self,
                                  date: str | pd.Timestamp) -> pd.Timestamp | None:
        """The latest date the dataset carries at or before *date*.

        A date inside the coverage that has no rows is an ordinary closed
        market, and this is the session that was in force through it. None
        when the dataset begins after *date*, since then there is no earlier
        session to be in force.
        """
        sessions = self.sessions
        position = as_of_position(sessions, date)

        return None if position is None else pd.Timestamp(sessions[position])

    # -- query ---------------------------------------------------------------

    def session_columns(self,
                        date: str | pd.Timestamp) -> tuple[dict[str, dict[str, object]], set[str]]:
        """Every column's values for one session, keyed by identifier.

        The read the daily loops make once per session, done without touching
        pandas. The frame is indexed `(IDENTIFIER, DATE)`, so a single day's
        rows are scattered through it rather than adjacent, and `get()` has to
        filter the whole frame to find them. This looks up the row positions
        for the date from a per-frame index instead, so the read is a gather
        and two dict builds. Where an (identifier, date) pair is repeated, the
        first row wins, as it does for `get()` followed by `.iloc[0]`.

        Returns:
            tuple: ``({column: {identifier: value}}, identifiers)``, exactly
            what a :class:`~beacon.data.session.SessionPanel` holds, so no
            DataFrame is built on the way. Empty for a date the data has no
            rows on.
        """
        # BN-213. `get()` measured over 156,400 rows: 13.5 ms a day, which is
        # most of an index calculation, and the daily loops make this read
        # eight hundred times a run. `_date_index` holds the per-frame index.
        order, bounds, identifiers, arrays = self._date_index()
        span = bounds.get(pd.Timestamp(date))

        if span is None:
            return {}, set()

        rows = order[span[0]:span[1]]

        # Reversed so that a repeated (identifier, date) keeps its FIRST row:
        # `dict` takes the last value written for a key, so feeding the pairs
        # backwards makes the earliest win. That is what `get()` followed by
        # `.iloc[0]` already did, and a panel that disagreed with the read it
        # replaces would be a quieter bug than the slowness it fixes.
        names = identifiers[rows][::-1]

        values = {column: dict(zip(names, array[rows][::-1], strict=True))
                  for column, array in arrays.items()}

        return values, {str(name) for name in names}

    def session_rows(self,
                     date: str | pd.Timestamp) -> tuple["np.ndarray", "np.ndarray",
                                                        dict[str, "np.ndarray"]] | None:
        """The frame's identifiers, one session's row positions, and the
        column arrays, with nothing gathered or built; None for a date with
        no rows. What a lazy session panel holds (BN-348, BN-351)."""
        order, bounds, identifiers, arrays = self._date_index()
        span = bounds.get(pd.Timestamp(date))

        if span is None:
            return None

        rows = order[span[0]:span[1]]

        return identifiers, rows, arrays

    def codes_for(self,
                  identifiers: list[str]) -> "np.ndarray":
        """Each identifier's code among this frame's rows, -1 for a name with
        no rows (BN-351).

        The last answer is kept for the same list object, so a caller asking
        about the same names day after day looks them up once.
        """
        cached = getattr(self, "_codes_for_cache", None)

        if cached is not None and cached[0] is identifiers and cached[1] is self._df:
            return cached[2]  # type: ignore[no-any-return]

        _, lookup = self._identifier_codes()
        codes = np.array([lookup.get(name, -1) for name in identifiers], dtype=np.int64)
        self._codes_for_cache = (identifiers, self._df, codes)

        return codes

    def session_values(self,
                       date: pd.Timestamp,
                       codes: "np.ndarray",
                       column: str) -> "np.ndarray | None":
        """*column* on *date* for the names with *codes*, NaN where a name has
        no row; None when the column is missing or not a float column.

        The values the session panel gives, as one array (BN-351): where a
        name has several rows on the day the first is taken, as the panel's
        reversed fill takes it.
        """
        order, bounds, _, arrays = self._date_index()
        array = arrays.get(column)

        if array is None or array.dtype.kind != "f":
            return None

        values = np.full(len(codes), np.nan)
        span = bounds.get(pd.Timestamp(date))

        if span is None:
            return values

        rows = order[span[0]:span[1]]
        row_codes = self._identifier_codes()[0][rows]

        # Stable, so among a name's rows the first stays first, and a
        # left-hand search lands on it.
        sorter = np.argsort(row_codes, kind="stable")
        ordered = row_codes[sorter]
        at = np.minimum(np.searchsorted(ordered, codes), len(ordered) - 1)
        found = (codes >= 0) & (ordered[at] == codes)
        values[found] = array[rows[sorter[at[found]]]]

        return values

    def _identifier_codes(self) -> tuple["np.ndarray", dict[str, int]]:
        """A code per row for its identifier, and identifier -> code, built
        once per frame like `_date_index`."""
        cached = getattr(self, "_identifier_codes_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1], cached[2]

        codes, uniques = pd.factorize(self._date_index()[2])
        lookup = {str(name): position for position, name in enumerate(uniques)}
        self._identifier_codes_cache = (self._df, codes, lookup)

        return codes, lookup

    def _date_index(self) -> tuple["np.ndarray", dict[pd.Timestamp, tuple[int, int]],
                                   "np.ndarray", dict[str, "np.ndarray"]]:
        """Everything a session read needs, built once per frame.

        Four things, none of them a copy of the data:

        * `order` sorts the frame's row positions by date, and `bounds` says
          where each date's run begins and ends within it -- so a day's rows
          are `order[lo:hi]`, found by dict lookup rather than by scanning.
        * `identifiers` and `arrays` are the frame's own buffers, converted
          once. For float and integer columns `to_numpy()` is a view, so this
          costs nothing and duplicates nothing.

        The last two were originally converted inside `session_columns`, which
        made them per-session work: 3.8 ms each time for the identifier array
        alone, because it is a string (object) array and pandas scans all of
        it for nulls on every conversion. Over 782 sessions that was 2.1 s,
        more than a quarter of the whole calculation, to rebuild an array that
        cannot change (BN-214).

        What this still does **not** hold is a sorted copy of the data.
        Storing sorted columns reads 1.2x faster and duplicates every value:
        1.2 GB over six thousand names and twenty-five years, against one
        index array. Measured both; the gather is the better trade, and the
        difference is 0.017 ms against 0.014 on a read that was 13.5.

        Cached on the frame's identity rather than a flag, the same way
        `identifiers` is, so replacing `_df` invalidates it automatically
        instead of relying on every future mutation remembering to.
        """
        cached = getattr(self, "_date_index_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1], cached[2], cached[3], cached[4]

        dates = self._df.index.get_level_values("DATE").to_numpy()
        order = np.argsort(dates, kind="stable")
        unique, first = np.unique(dates[order], return_index=True)
        edges = np.append(first, len(order))

        bounds = {pd.Timestamp(day): (int(edges[index]), int(edges[index + 1]))
                  for index, day in enumerate(unique)}

        identifiers = self._df.index.get_level_values("IDENTIFIER").to_numpy()
        arrays = {str(column): self._df[column].to_numpy()
                  for column in self._df.columns}

        self._date_index_cache = (self._df, order, bounds, identifiers, arrays)

        return order, bounds, identifiers, arrays


    def get(self,
            identifier: str | list[str],
            start_date: str | None = None,
            end_date: str | None = None,
            columns: list[str] | None = None) -> pd.DataFrame:
        """Return data for one or more identifiers, optionally filtered.

        Args:
            identifier: Single identifier or list of identifiers.
            start_date: Date string to slice the start of the date range.
            end_date: Date string to slice the end of the date range.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Single identifier: indexed by ``DATE``. List of
            identifiers: MultiIndexed by ``(IDENTIFIER, DATE)``. Empty
            DataFrame if no matching data is found.
        """
        if isinstance(identifier, list):
            # Against the cached set (BN-337): the IDENTIFIER level rebuilt on
            # each call is every row label of the frame, 11.6M of them on the
            # default synthetic store, for a read that wants one name's rows.
            existing = self.identifier_set
            identifier = [i for i in identifier if i in existing]
            if not identifier:
                return pd.DataFrame()

        try:
            subset = self._df.loc[identifier]
        except KeyError:
            return pd.DataFrame()

        if start_date is not None or end_date is not None:
            if isinstance(identifier, list):
                dates = subset.index.get_level_values("DATE")
            else:
                dates = subset.index

            mask = pd.Series(True, index=subset.index)
            if start_date is not None:
                mask &= dates >= pd.Timestamp(start_date)
            if end_date is not None:
                mask &= dates <= pd.Timestamp(end_date)
            subset = subset.loc[mask]

        if columns is not None:
            subset = subset[columns]

        return subset


class ReferenceData:
    """Reference data container with validity ranges.

    The source file must contain ``IDENTIFIER`` and ``DATE_FROM`` columns.
    ``DATE_TO`` is optional: a missing column, or NaT in it, marks a
    currently-active record.

    Indexed on ``IDENTIFIER`` (non-unique, since an identifier may have
    multiple validity periods).
    """

    def __init__(self,
                 file_path: str):
        df = _read_file(file_path)
        self._df = self._prepare(df)

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame) -> "ReferenceData":
        """Create a ReferenceData instance from an existing DataFrame."""
        instance = object.__new__(cls)
        instance._df = cls._prepare(df.copy())
        return instance

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in ("IDENTIFIER", "DATE_FROM") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        if "DATE_TO" in df.columns:
            df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
        else:
            df["DATE_TO"] = pd.NaT

        df = df.set_index("IDENTIFIER").sort_index()
        return df

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

    @property
    def identifiers(self) -> list[str]:
        """Unique identifiers present in the dataset, cached on the frame's identity."""
        cached: tuple[object, list[str]] | None = getattr(
            self, "_identifier_cache", None)

        if cached is not None and cached[0] is self._df:
            return cached[1]

        values = [str(value) for value in self._df.index.unique()]
        self._identifier_cache = (self._df, values)

        return values

    @property
    def columns(self) -> list[str]:
        """Column names (including DATE_FROM, DATE_TO)."""
        return list(self._df.columns)

    # -- query ---------------------------------------------------------------

    def get(self,
            identifier: str | list[str],
            date: str | None = None,
            columns: list[str] | None = None) -> pd.DataFrame:
        """Return reference data for one or more identifiers.

        Args:
            identifier: Single identifier or list of identifiers.
            date: Point-in-time date. If provided, only rows where
                ``DATE_FROM <= date`` and (``DATE_TO >= date`` or
                ``DATE_TO`` is NaT) are returned.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Indexed by ``IDENTIFIER``. Empty DataFrame if no
            match.
        """
        if isinstance(identifier, list):
            existing = self._df.index
            identifier = [i for i in identifier if i in existing]
            if not identifier:
                return pd.DataFrame()

        try:
            subset = self._df.loc[identifier]
        except KeyError:
            return pd.DataFrame()

        # .loc on a non-unique index with a single str returns a Series
        # if exactly one row matches — normalize to DataFrame.
        if isinstance(subset, pd.Series):
            subset = subset.to_frame().T
            subset.index.name = "IDENTIFIER"

        if date is not None:
            ts = pd.Timestamp(date)
            mask = subset["DATE_FROM"] <= ts
            mask &= subset["DATE_TO"].isna() | (subset["DATE_TO"] >= ts)
            subset = subset.loc[mask]

        if columns is not None:
            subset = subset[columns]

        return subset
