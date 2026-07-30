# src/beacon/data/corporate_actions.py
"""
Corporate-action history.

`IndexCalculator` has always been able to adjust a divisor for an action it is
*handed*. Nothing stored a series of them, so there was no way to ask what a
constituent paid over the last year, and no way to serve
`/data/corporate-actions/{ticker}`. This is that store, sitting beside
`MarketData` and `ReferenceData` and loaded the same way.

One row is one action: an identifier, an ex-date, a type and a value. What the
value *means* depends on the type, and conflating the two is the mistake this
module is arranged to prevent:

* **Cash actions** — ``DIVIDEND``, ``SPECIAL_DIVIDEND``, ``RETURN_OF_CAPITAL``
  — carry an amount per share. They add up. Two dividends of 0.25 make 0.50.
* **Ratio actions** — ``SPLIT``, ``REVERSE_SPLIT``, ``STOCK_DIVIDEND`` — carry
  a multiplier on the share count. They compound. Two 2-for-1 splits make a
  factor of 4, not 4-for-1 in any additive sense.

So there is no single "total actions" helper. Summing a split ratio into a
dividend total would produce a number with no meaning, and the only way to stop
that happening is to not offer the operation.

## What trailing twelve months means here

Twelve *calendar* months back from the as-of date, not 365 days. The two differ
across a leap day, and "twelve months" is what a yield is quoted on. The window
is half-open — ``as_of - 1 year < ex_date <= as_of`` — so an action exactly one
year old has rolled out and one dated today is in. Without that, a dividend
paid on the anniversary would be counted in two consecutive years' figures.
"""
import logging

import pandas as pd

from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

# Cash per share. These add up.
DIVIDEND = "DIVIDEND"
SPECIAL_DIVIDEND = "SPECIAL_DIVIDEND"
RETURN_OF_CAPITAL = "RETURN_OF_CAPITAL"
CASH_ACTIONS = frozenset({DIVIDEND, SPECIAL_DIVIDEND, RETURN_OF_CAPITAL})

# Multipliers on the share count. These compound.
SPLIT = "SPLIT"
REVERSE_SPLIT = "REVERSE_SPLIT"
STOCK_DIVIDEND = "STOCK_DIVIDEND"
RATIO_ACTIONS = frozenset({SPLIT, REVERSE_SPLIT, STOCK_DIVIDEND})

# Structural events, carrying no directly aggregable value.
RIGHTS_ISSUE = "RIGHTS_ISSUE"
SPIN_OFF = "SPIN_OFF"
MERGER = "MERGER"
STRUCTURAL_ACTIONS = frozenset({RIGHTS_ISSUE, SPIN_OFF, MERGER})

ACTION_TYPES = CASH_ACTIONS | RATIO_ACTIONS | STRUCTURAL_ACTIONS

REQUIRED_COLUMNS = ("IDENTIFIER", "EX_DATE", "TYPE", "VALUE")

# One trailing year. A DateOffset rather than a Timedelta so February the 29th
# behaves.
TRAILING_YEAR = pd.DateOffset(years=1)


class CorporateActions:
    """A history of corporate actions, indexed by identifier and ex-date.

    The source must carry ``IDENTIFIER``, ``EX_DATE``, ``TYPE`` and ``VALUE``.
    Anything else — ``PAY_DATE``, ``CURRENCY``, ``DECLARED_DATE`` — is carried
    through untouched.
    """

    def __init__(self,
                 frame: pd.DataFrame):
        self._df = self._prepare(frame.copy())

    @classmethod
    def from_dataframe(cls,
                       frame: pd.DataFrame) -> "CorporateActions":
        """Build from an existing DataFrame, matching the other containers."""
        return cls(frame)

    @classmethod
    def empty(cls) -> "CorporateActions":
        """A store with no actions in it.

        Worth having: most tests and most universes have no action history, and
        the alternative is every caller checking for None before asking.
        """
        return cls(pd.DataFrame(columns=list(REQUIRED_COLUMNS)))

    @staticmethod
    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        """Validate, type and sort the history."""
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise CalculationError(
                "CorporateActions",
                f"missing required column(s): {', '.join(missing)}.")

        if frame.empty:
            frame = frame.astype({"IDENTIFIER": "object", "TYPE": "object",
                                  "VALUE": "float64"})
            frame["EX_DATE"] = pd.to_datetime(frame["EX_DATE"])

            return frame.set_index(["IDENTIFIER", "EX_DATE"]).sort_index()

        frame["EX_DATE"] = pd.to_datetime(frame["EX_DATE"])
        frame["TYPE"] = frame["TYPE"].astype(str).str.upper()
        frame["VALUE"] = frame["VALUE"].astype(float)

        unknown = sorted(set(frame["TYPE"]) - ACTION_TYPES)
        if unknown:
            raise CalculationError(
                "CorporateActions",
                f"unknown action type(s): {', '.join(unknown)}. Known types: "
                f"{', '.join(sorted(ACTION_TYPES))}.")

        # Kept as columns as well as index levels, so a returned slice still
        # answers "which identifier, which date" without an index reset.
        return frame.set_index(["IDENTIFIER", "EX_DATE"], drop=False).sort_index()

    @property
    def data(self) -> pd.DataFrame:
        """A copy of the whole history."""
        return self._df.copy()

    @property
    def identifiers(self) -> list[str]:
        """Identifiers with at least one action."""
        return list(self._df.index.get_level_values("IDENTIFIER").unique())

    @property
    def is_empty(self) -> bool:
        """Whether the store holds nothing."""
        return bool(self._df.empty)

    def get(self,
            identifier: str,
            start_date: str | pd.Timestamp | None = None,
            end_date: str | pd.Timestamp | None = None,
            types: list[str] | None = None) -> pd.DataFrame:
        """Actions for one identifier, oldest first.

        Args:
            identifier: The instrument.
            start_date: Earliest ex-date to include, inclusive.
            end_date: Latest ex-date to include, inclusive.
            types: Restrict to these action types. None returns all.

        Returns:
            pd.DataFrame: Matching rows, or an empty frame with the right
            columns when there are none — so a caller can read ``VALUE`` off
            the result without checking first.
        """
        if identifier not in set(self.identifiers):
            return self._df.iloc[0:0].copy()

        subset = self._df.loc[[identifier]].copy()
        dates = subset.index.get_level_values("EX_DATE")

        if start_date is not None:
            subset = subset.loc[dates >= pd.Timestamp(start_date)]
            dates = subset.index.get_level_values("EX_DATE")

        if end_date is not None:
            subset = subset.loc[dates <= pd.Timestamp(end_date)]

        if types is not None:
            wanted = {value.upper() for value in types}
            subset = subset.loc[subset["TYPE"].isin(wanted)]

        return subset

    def trailing_cash(self,
                      identifier: str,
                      as_of: str | pd.Timestamp,
                      types: list[str] | None = None) -> float:
        """Cash per share over the twelve months ending *as_of*.

        Args:
            identifier: The instrument.
            as_of: End of the window, inclusive.
            types: Which cash actions to count. None counts ordinary dividends
                only — the conventional basis for a trailing yield, since a
                special dividend is by definition not expected to repeat.

        Returns:
            float: The total. 0.0 when there is nothing in the window.

        Raises:
            CalculationError: If a non-cash type is requested. A split ratio
                added to a dividend total would produce a number with no
                meaning.
        """
        wanted = {DIVIDEND} if types is None else {value.upper() for value in types}

        not_cash = sorted(wanted - CASH_ACTIONS)
        if not_cash:
            raise CalculationError(
                "CorporateActions",
                f"{', '.join(not_cash)} carr{'ies' if len(not_cash) == 1 else 'y'} "
                f"a ratio, not an amount, and cannot be added to a cash total. "
                f"Use cumulative_ratio() instead.")

        window = self._window(identifier, as_of, sorted(wanted))

        return float(window["VALUE"].sum())

    def trailing_dividend(self,
                          identifier: str,
                          as_of: str | pd.Timestamp) -> float:
        """Ordinary dividends per share over the trailing twelve months."""
        return self.trailing_cash(identifier, as_of)

    def trailing_dividend_yield(self,
                                identifier: str,
                                as_of: str | pd.Timestamp,
                                price: float) -> float:
        """Trailing dividends as a fraction of *price*.

        Args:
            identifier: The instrument.
            as_of: End of the trailing window.
            price: Price to divide by, normally the close on *as_of*.

        Returns:
            float: The yield.

        Raises:
            CalculationError: If *price* is not positive. A yield on a zero
                price is not a large number, it is an undefined one.
        """
        if price <= 0.0:
            raise CalculationError(
                "CorporateActions",
                f"a dividend yield needs a positive price, got {price}.")

        return self.trailing_dividend(identifier, as_of) / price

    def _window(self,
                identifier: str,
                as_of: str | pd.Timestamp,
                types: list[str]) -> pd.DataFrame:
        """Rows in the half-open trailing year ending *as_of*."""
        end = pd.Timestamp(as_of)
        start = end - TRAILING_YEAR

        subset = self.get(identifier, types=types)
        if subset.empty:
            return subset

        dates = subset.index.get_level_values("EX_DATE")

        # Half-open: an action exactly a year old has rolled out. Otherwise a
        # dividend paid on the anniversary lands in two consecutive years.
        return subset.loc[(dates > start) & (dates <= end)]

    def cumulative_ratio(self,
                         identifier: str,
                         start_date: str | pd.Timestamp | None = None,
                         end_date: str | pd.Timestamp | None = None) -> float:
        """Compounded share-count multiplier over a window.

        Two 2-for-1 splits give 4.0, not 4 in any additive sense — which is why
        this is separate from the cash helpers rather than a mode of them.

        Args:
            identifier: The instrument.
            start_date: Earliest ex-date, inclusive.
            end_date: Latest ex-date, inclusive.

        Returns:
            float: The product of every ratio in the window. 1.0 when there are
            none, which is the identity and leaves a share count unchanged.

        Raises:
            CalculationError: If a ratio is not positive. A zero or negative
                multiplier would erase or invert a position.
        """
        window = self.get(identifier, start_date, end_date, types=sorted(RATIO_ACTIONS))
        if window.empty:
            return 1.0

        ratios = window["VALUE"].to_numpy(dtype=float)
        if (ratios <= 0.0).any():
            raise CalculationError(
                "CorporateActions",
                f"{identifier} has a non-positive split ratio, which would "
                f"erase or invert a share count.")

        return float(ratios.prod())

    def as_records(self,
                   identifier: str,
                   date: str | pd.Timestamp) -> list[dict[str, object]]:
        """Actions on one date, in the shape IndexCalculator expects.

        The calculator takes ``{"type", "asset", "value", "ex_date"}``. Building
        that here keeps the mapping in one place rather than at every call site.

        Args:
            identifier: The instrument.
            date: The ex-date.

        Returns:
            list: One dict per action, empty when there are none.
        """
        stamp = pd.Timestamp(date)
        window = self.get(identifier, stamp, stamp)

        return [{"type": row["TYPE"],
                 "asset": identifier,
                 "value": float(row["VALUE"]),
                 "ex_date": stamp}
                for _, row in window.iterrows()]
