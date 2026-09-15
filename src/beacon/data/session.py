# src/beacon/data/session.py
"""One market session, read for a whole set of names at once."""
from collections.abc import Iterable

import pandas as pd


class SessionPanel:
    """Every market column of one session, for a named set of instruments.

    A methodology that walks a universe asks the same question of every name on
    the same day: what did it close at, how many shares are out, how much of it
    floats. Asked one name at a time, each answer is a slice of the whole market
    frame, so the cost of a single lookup grows with the *frame* rather than
    with the question — which is what made a preview superlinear in its universe
    size (BN-190). Asked once for the whole list it is one slice, and the
    per-name reads that follow are dictionary lookups.

    The values are held as plain dicts rather than as the frame they came from
    because the per-name read is the whole point: ``.at`` on a DataFrame is
    microseconds, and microseconds across a six-thousand-name universe times
    three columns is the cost this exists to remove.

    A panel is a pure function of (market frame, identifiers, session), so it
    can be held and reused for exactly as long as all three hold — and no
    longer. It answers only for the identifiers it was built with and only for
    its own session; anything else must go back to the data. That is what keeps
    a reused panel from being a different question's answer: there is no key
    under which name A's row can be returned for name B, or Monday's for
    Tuesday's.
    """

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

    @classmethod
    def from_market_frame(cls,
                          session: pd.Timestamp,
                          frame: pd.DataFrame) -> "SessionPanel":
        """Build a panel from what a multi-identifier market fetch returns.

        That frame is MultiIndexed by ``(IDENTIFIER, DATE)`` and has already
        been filtered to the one date, so the date level is dropped rather than
        filtered again. A repeated ``(identifier, date)`` keeps its first row,
        which is what a single-name fetch followed by ``.iloc[0]`` already did
        — a panel that disagreed with the read it replaces would be a quieter
        bug than the slowness it fixes.
        """
        if frame.empty:
            return cls(session, pd.DataFrame())

        if isinstance(frame.index, pd.MultiIndex):
            frame = frame.droplevel("DATE")

        return cls(session, frame[~frame.index.duplicated(keep="first")])

    def covers(self,
               identifiers: Iterable[str]) -> bool:
        """Whether every one of *identifiers* has a row in this panel."""
        return all(identifier in self._identifiers for identifier in identifiers)

    def answers(self,
                identifier: str,
                date: str) -> bool:
        """Whether this panel can answer for *identifier* on *date*.

        Both halves, and nothing looser. A panel holds one session for the set
        of names it was built from; asked about another date or another name it
        says no, and the caller goes to the data. There is deliberately no
        nearest-session or missing-name behaviour here — a panel that answered
        beyond what it was built from would be a different question's answer
        wearing this one's key.
        """
        return date == self.stamp and identifier in self._identifiers

    def value(self,
              identifier: str,
              column: str) -> float | None:
        """One instrument's value in *column*, or None where there is none.

        None for an absent column and for a null value alike, on the same terms
        as :meth:`~beacon.data.fetcher.DataFetcher._market_scalar`: both mean
        "the data does not say", and a caller that has to tell them apart is
        asking a question about the store rather than about the instrument.
        """
        column_values = self._values.get(column)

        if column_values is None:
            return None

        value = column_values.get(identifier)

        if value is None or pd.isna(value):
            return None

        return float(value)  # type: ignore[arg-type]
