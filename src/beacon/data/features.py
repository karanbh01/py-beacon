# src/beacon/data/features.py
"""
Features: everything about an instrument that is not price, reference or action.

Fundamentals, alternative datasets, macroeconomic series, and values somebody
derived and imported all share one shape, so they share one table. The
alternative was a surface per kind — a fundamentals endpoint, then an
alternative-data endpoint, then a macro one — each with its own schema and its
own point-in-time rules to get subtly wrong.

## The shape

Field-value pairs, keyed by instrument and date:

    IDENTIFIER   ticker, ISIN, or whatever the loaded data keys on
    DATE        the date the value became knowable — see below
    TYPE         which dataset it came from
    FIELD        the datapoint: revenue, total_debt, pe_ratio, card_spend
    VALUE        the number
    DETAIL       free-form context, no defaults and no fixed vocabulary

`TYPE` is what lets several datasets share the table without colliding.
`revenue` from a vendor and `revenue` from a user's own model are different
series, and a query has to be able to say which it means.

## DATE is the announcement date, not the period end

This is the decision the whole table rests on.

A fundamental has at least two dates: the period it describes (Q1 2024) and
the date it was published (2024-05-15). **Only the second decides
visibility.** A backtest standing on 2024-04-01 must not see Q1 revenue that
nobody knew until May — that is look-ahead bias, it is the most common way a
fundamental-driven strategy shows returns it could never have earned, and the
dangerous part is that the obvious point-in-time query looks correct while
producing it.

So `DATE` holds when the value became knowable, and the period it describes
goes in `DETAIL`.

The column is called `DATE`, matching what every other dataset here names its
date, rather than something like `AS_OF` that encodes the rule. The
point-in-time behaviour is a property of *how the table is read* — it belongs
to the accessor, not to the column name. Storing the period end in `DATE` would make every
downstream query wrong in the same direction at once.

`DETAIL` is deliberately unconstrained. A restatement, a fiscal-period label,
a vendor revision number, a units note — all real, all things somebody will
need to record, and a fixed schema invented here would be invented wrong.

## Restatements are kept, not overwritten

A vendor revising Q1 revenue in August does not erase what it said in May, and
a backtest standing in June must still see the May figure. So a repeated
``(IDENTIFIER, DATE, TYPE, FIELD)`` is kept rather than deduplicated, and the
accessor resolves which one is in force. Overwriting would make the table
smaller and the history unrecoverable.

Two rows with the *same* key are a genuine duplicate — the same claim loaded
twice — and the last one wins, because that is what re-importing a corrected
file should do.
"""
from typing import Any

import pandas as pd

from .base import _read_file

# The columns a feature row cannot do without.
REQUIRED_COLUMNS = ("IDENTIFIER", "DATE", "TYPE", "FIELD", "VALUE")

# Optional, and never defaulted: an absent DETAIL means the dataset had no
# further context, which is different from an empty one.
DETAIL_COLUMN = "DETAIL"

COLUMNS = (*REQUIRED_COLUMNS, DETAIL_COLUMN)

# How old a value may be before it stops counting as current.
#
# A staleness bound is not optional, and the reason is asymmetric. Serving a
# six-year-old fundamental as though it were current is worse than serving
# nothing: nothing is visibly a gap, and a stale number is a plausible answer
# that quietly makes every screen built on it wrong.
#
# Two years, because it is generous enough that a genuinely annual series
# survives one missed publication and tight enough that a name whose coverage
# stopped is reported as having stopped. Callers wanting a different bound
# pass one; callers wanting none pass None and take responsibility for it.
MAX_AGE_DAYS = 730


class FeatureData:
    """Per-instrument datapoints, keyed by identifier and as-of date.

    Indexed on ``(IDENTIFIER, DATE)``, non-unique: an instrument carries many
    fields on one date, and one field across many dates.
    """

    def __init__(self,
                 file_path: str):
        self._df = self._prepare(_read_file(file_path))

    @classmethod
    def from_dataframe(cls,
                       df: pd.DataFrame) -> "FeatureData":
        """Create a FeatureData instance from an existing DataFrame."""
        instance = object.__new__(cls)
        instance._df = cls._prepare(df.copy())

        return instance

    @classmethod
    def empty(cls) -> "FeatureData":
        """A container holding nothing.

        So a dataset without features is still a dataset, and callers can ask
        it questions without checking for None first.
        """
        return cls.from_dataframe(pd.DataFrame(columns=list(COLUMNS)))

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Validate and index a feature frame."""
        missing = [column for column in REQUIRED_COLUMNS
                   if column not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required column(s): {', '.join(missing)}")

        if DETAIL_COLUMN not in df.columns:
            df[DETAIL_COLUMN] = None

        # Pinned to object, whatever it arrived as. A column that is entirely
        # null reads back from CSV as float64 NaN, so a table saved with no
        # detail and reloaded would hand a caller floats where it had stored
        # text -- and only for the tables where every row happened to be
        # empty, which is the worst way for a dtype to vary.
        df[DETAIL_COLUMN] = df[DETAIL_COLUMN].astype(object).where(
            df[DETAIL_COLUMN].notna(), None)

        df["DATE"] = pd.to_datetime(df["DATE"])
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")

        for column in ("TYPE", "FIELD"):
            blank = df[column].isna() | (df[column].astype(str).str.strip() == "")

            if blank.any():
                raise ValueError(
                    f"{int(blank.sum())} row(s) have an empty {column}; it "
                    f"decides which dataset a value belongs to and cannot be "
                    f"inferred.")

        # The last row wins for an exact duplicate — the same claim loaded
        # twice, which is what re-importing a corrected file produces. A
        # *restatement* has a different DATE and is not a duplicate: both are
        # kept, because a backtest standing between them must see the earlier
        # one.
        df = df.drop_duplicates(subset=["IDENTIFIER", "DATE", "TYPE", "FIELD"],
                                keep="last")

        return df.set_index(["IDENTIFIER", "DATE"]).sort_index()

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """A copy of the underlying frame."""
        return self._df.copy()

    @property
    def is_empty(self) -> bool:
        """Whether anything is loaded."""
        return bool(self._df.empty)

    @property
    def identifiers(self) -> list[str]:
        """Instruments the table carries anything for."""
        if self._df.empty:
            return []

        return [str(value) for value
                in self._df.index.get_level_values("IDENTIFIER").unique()]

    @property
    def columns(self) -> list[str]:
        """Column names, excluding the index keys."""
        return list(self._df.columns)

    @property
    def types(self) -> list[str]:
        """The datasets loaded, so a client can populate a control."""
        if self._df.empty:
            return []

        return sorted({str(value) for value in self._df["TYPE"].unique()})

    def fields(self,
               feature_type: str | None = None) -> list[str]:
        """The fields loaded, optionally within one dataset.

        Args:
            feature_type: Restrict to one `TYPE`. None returns every field
                across every dataset, which will collide names where two
                datasets share one.
        """
        if self._df.empty:
            return []

        frame = self._df

        if feature_type is not None:
            frame = frame[frame["TYPE"] == feature_type]

        return sorted({str(value) for value in frame["FIELD"].unique()})

    def value_as_of(self,
                    identifier: str,
                    field: str,
                    date: pd.Timestamp | str | None = None,
                    feature_type: str | None = None,
                    max_age_days: int | None = MAX_AGE_DAYS) -> float | None:
        """The value in force on a date, or None.

        The most recent row whose own `DATE` is on or before `date` — not the
        row *for* that date. Fundamentals are quarterly and a backtest runs
        daily, so "the latest thing knowable on this date" is the only
        question worth asking.

        **It never looks forward.** A value published on 2024-05-15 is
        invisible to a query standing on 2024-04-01, however recently the
        period it describes ended. That is the whole reason `DATE` holds the
        announcement date, and this is where it pays off or fails to.

        Args:
            identifier: The instrument.
            field: The datapoint.
            date: Stand here. None uses the latest date in the table, which is
                the right default for "what do we know now" and the wrong one
                for a backtest — which always passes its own.
            feature_type: Restrict to one dataset. None searches every type,
                which will pick arbitrarily between two carrying the same
                field name, so a caller that has both should say which.
            max_age_days: How stale is too stale. None disables the check.

        Returns:
            float | None: The value, or None when nothing is knowable — no
            coverage, nothing published yet, or nothing recent enough.
        """
        rows = self.history(identifier, field, date, feature_type)

        if rows.empty:
            return None

        latest = rows.index.get_level_values("DATE").max()

        if max_age_days is not None:
            standing = (self._latest_date() if date is None
                        else pd.Timestamp(date))

            if (standing - latest).days > max_age_days:
                return None

        value = rows.loc[rows.index.get_level_values("DATE") == latest,
                         "VALUE"].iloc[-1]

        return None if pd.isna(value) else float(value)

    def history(self,
                identifier: str,
                field: str,
                date: pd.Timestamp | str | None = None,
                feature_type: str | None = None) -> pd.DataFrame:
        """Every row for one instrument and field, up to a date.

        Exposed because a restatement is kept rather than overwritten (see the
        module docstring), so "what did we believe, and when" is a question
        this table can answer and a caller may need to.
        """
        if self._df.empty:
            return self._df

        try:
            rows = self._df.xs(identifier, level="IDENTIFIER", drop_level=False)
        except KeyError:
            return self._df.iloc[0:0]

        rows = rows[rows["FIELD"] == field]

        if feature_type is not None:
            rows = rows[rows["TYPE"] == feature_type]

        if date is not None:
            standing = pd.Timestamp(date)
            rows = rows[rows.index.get_level_values("DATE") <= standing]

        return rows

    def _latest_date(self) -> pd.Timestamp:
        """The most recent date anywhere in the table."""
        return pd.Timestamp(self._df.index.get_level_values("DATE").max())

    def coverage(self) -> dict[str, Any]:
        """How much of each dataset is present, for `/data/coverage`."""
        if self._df.empty:
            return {"identifiers": 0, "types": [], "fields": 0, "rows": 0}

        return {"identifiers": len(self.identifiers),
                "types": self.types,
                "fields": len(self.fields()),
                "rows": len(self._df)}
