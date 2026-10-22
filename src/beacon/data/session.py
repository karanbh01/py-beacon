# src/beacon/data/session.py
"""One market session, read for a whole set of names at once."""
from collections.abc import Iterable

import numpy as np
import pandas as pd


class SessionPanel:
    """Every market column of one session, for a named set of instruments.

    A methodology that walks a universe asks the same question of every name on
    the same day: what did it close at, how many shares are out, how much of it
    floats. Asked one name at a time, each answer is a slice of the whole market
    frame, so the cost of a single lookup grows with the *frame* rather than
    with the question, making a preview superlinear in its universe size.
    Asked once for the whole list it is one slice, and the per-name reads that
    follow are dictionary lookups.

    The values are held as plain dicts rather than as the frame they came from
    because the per-name read is the whole point: ``.at`` on a DataFrame is
    microseconds, and microseconds across a six-thousand-name universe times
    three columns is the cost this exists to remove.

    A panel is a pure function of (market frame, identifiers, session), so it
    can be held and reused for exactly as long as all three hold, and no
    longer. It answers only for the identifiers it was built with and only for
    its own session; anything else must go back to the data. That is what keeps
    a reused panel from being a different question's answer: there is no key
    under which name A's row can be returned for name B, or Monday's for
    Tuesday's.
    """
    # BN-190: the per-name slicing is what made a preview superlinear.

    def __init__(self,
                 session: pd.Timestamp,
                 frame: pd.DataFrame):
        """Build a panel from rows already reduced to one session.

        Args:
            session: The session these rows are for.
            frame: Rows indexed by ``IDENTIFIER``, one per instrument.
        """
        self.session: pd.Timestamp = pd.Timestamp(session)
        # The same date as a `%Y-%m-%d` string, which is the spelling every
        # per-name read arrives in. Comparing strings rather than parsing each
        # read's date into a Timestamp keeps a *missed* panel free, which
        # matters more than a hit: the daily calculation loop reads a different
        # date from the rebalance panel it is still holding, thousands of times.
        self.stamp: str = self.session.strftime("%Y-%m-%d")
        self._values: dict[str, dict[str, object]] = {
            str(column): frame[column].to_dict() for column in frame.columns}
        self._identifiers: set[str] = {str(name) for name in frame.index}
        # The rows a panel made `from_rows` builds its columns from (BN-348).
        self._lazy: tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]] | None = None
        self._names: np.ndarray | None = None
        # Whether this holds every row the data has on its session (BN-347).
        # Then a name it lacks has no row that day, which is the answer the
        # data would give, so it answers for that name too.
        self.complete: bool = False

    @classmethod
    def from_columns(cls,
                     session: pd.Timestamp,
                     values: dict[str, dict[str, object]],
                     identifiers: set[str]) -> "SessionPanel":
        """Build a panel from column dictionaries, with no frame in between.

        `MarketData.session_columns` produces exactly what a panel holds, so
        no DataFrame is constructed only to be taken apart again.
        """
        # BN-213. Building the DataFrame cost more than the lookup it enabled.
        panel = cls(session, pd.DataFrame())
        panel._values = values
        panel._identifiers = identifiers
        # `session_columns` gives every row the data has on the session.
        panel.complete = True
        return panel

    @classmethod
    def from_rows(cls,
                  session: pd.Timestamp,
                  identifiers: np.ndarray,
                  rows: np.ndarray,
                  arrays: dict[str, np.ndarray]) -> "SessionPanel":
        """Build a whole-session panel that turns a column into a dict only
        when that column is read (BN-348).

        A daily valuation reads the close and nothing else, so building every
        column's dict for every name every day was most of the read.
        *identifiers* and *arrays* are the frame's, one entry per row, and
        *rows* the session's row positions, as `session_rows` gives them.
        """
        panel = cls(session, pd.DataFrame())
        # No set of its names (BN-351): a complete panel answers for every
        # name, so `covers` and `answers` never consult one.
        panel._lazy = (identifiers, rows, arrays)
        panel._values = {}
        panel.complete = True

        return panel

    @classmethod
    def from_market_frame(cls,
                          session: pd.Timestamp,
                          frame: pd.DataFrame) -> "SessionPanel":
        """Build a panel from what a multi-identifier market fetch returns.

        That frame is MultiIndexed by ``(IDENTIFIER, DATE)`` and has already
        been filtered to the one date, so the date level is dropped rather than
        filtered again. A repeated ``(identifier, date)`` keeps its first row,
        as a single-name fetch followed by ``.iloc[0]`` does, so the panel and
        the read it replaces agree.
        """
        if frame.empty:
            return cls(session, pd.DataFrame())

        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.droplevel("DATE")

        return cls(session, frame[~frame.index.duplicated(keep="first")])

    def _column(self,
                name: str) -> dict[str, object] | None:
        """One column's values by identifier, built on first read for a panel
        made `from_rows`; None for a column it does not hold."""
        if name in self._values:
            return self._values[name]

        lazy = self._lazy

        if lazy is None or name not in lazy[2]:
            return None

        identifiers, rows, arrays = lazy

        # The names gathered on the first column read, not when the panel is
        # built (BN-351): most days nothing reads it, and gathering thousands
        # of names scattered through the frame was most of a warm's cost.
        # Reversed so that a repeated name keeps its first row.
        if self._names is None:
            self._names = identifiers[rows][::-1]

        self._values[name] = dict(zip(self._names, arrays[name][rows][::-1], strict=True))

        return self._values[name]

    def covers(self,
               identifiers: Iterable[str]) -> bool:
        """Whether this panel answers for every one of *identifiers*: all of
        them, when it holds the whole session."""
        return self.complete or all(identifier in self._identifiers
                                    for identifier in identifiers)

    def answers(self,
                identifier: str,
                date: str) -> bool:
        """Whether this panel can answer for *identifier* on *date*.

        Both halves, and nothing looser. A panel holds one session for the set
        of names it was built from; asked about another date or another name it
        says no, and the caller goes to the data. There is deliberately no
        nearest-session behaviour here: a panel that answered for another day
        would be a different question's answer wearing this one's key.

        A panel built from the whole session (`complete`) answers for every
        name on its date: one it lacks has no row that day, and the data's own
        read would find none either. Before BN-347 such a name went back to the
        data, and on a session the data does not trade at all, every holding
        did, which was most of a backtest on another market's calendar.

        Args:
            identifier: The instrument.
            date: The date as a ``%Y-%m-%d`` string, compared with `stamp`.
        """
        return date == self.stamp and (self.complete or identifier in self._identifiers)

    def column(self,
               name: str) -> dict[str, object]:
        """Every instrument's value in one column, identifier to value.

        The whole day in one call, for a caller valuing a book rather than a
        name. Values are as stored, NaN included, because deciding what a
        missing value means is the caller's business; `DataFetcher.prices_on`
        turns NaN into None. Empty for a column the panel does not hold, on
        the same terms as `value`.
        """
        # BN-218.
        return self._column(name) or {}

    def value(self,
              identifier: str,
              column: str) -> float | None:
        """One instrument's value in *column*, or None where there is none.

        None for an absent column and for a null value alike, on the same terms
        as :meth:`~beacon.data.fetcher.DataFetcher.fetch_price`: both mean
        "the data does not say", and a caller that has to tell them apart is
        asking a question about the store rather than about the instrument.
        """
        column_values = self._column(column)

        if column_values is None:
            return None

        value = column_values.get(identifier)

        if value is None or pd.isna(value):
            return None

        return float(value)  # type: ignore[arg-type]
