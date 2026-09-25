# src/beacon/data/layout.py
"""The table layout data is supplied in, and the checks every row must pass.

One layout serves every way a user brings their own data: CSV files, an Excel
workbook, and the tables or views of a Postgres database. Each dataset is a
sheet (or file, or table) with a fixed name and fixed columns:

- **market** (required): `IDENTIFIER`, `DATE`, `CLOSE`; optionally `OPEN`,
  `HIGH`, `LOW`, `VOLUME`, `SHARES_OUTSTANDING`, `FREE_FLOAT`.
- **reference** (required): `IDENTIFIER`, `NAME`, `CURRENCY`, `EXCHANGE`,
  `DATE_FROM`; optionally `DATE_TO` and descriptive columns such as `SECTOR`
  and `COUNTRY`.
- **fx**: `PAIR` (such as `GBPUSD`), `DATE`, `RATE`.
- **corporate_actions**: `IDENTIFIER`, `EX_DATE`, `TYPE`, `VALUE`; optionally
  `PAY_DATE`, `STATUS`.
- **features**: `IDENTIFIER`, `DATE`, `FIELD`, `VALUE`; optionally `TYPE`
  (default "imported") and `DETAIL`.

Extra columns are kept, so a reference sheet can carry any descriptive fields.

`check` reports every problem at once, each naming its sheet, row and column,
so a user can fix a file in one pass. Nothing is loaded from sheets that fail
it: a partly valid import is refused whole rather than half-loaded.
"""
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..exceptions import InvalidRuleError
from .base import MarketData, ReferenceData
from .corporate_actions import ACTION_TYPES, STATUSES, CorporateActions
from .features import FeatureData
from .fetcher import DataFetcher

# The number of problems reported. A file with a systematic mistake (a date
# column in the wrong format, say) would otherwise produce one per row, which
# helps nobody. The total is still reported.
MAX_PROBLEMS = 200

# Dates are read in one form only: YYYY-MM-DD (a time after it is allowed, as
# Excel date cells come through with one). Guessing each value's format would
# read 01/02/2024 as January or February depending on the machine, silently.
DATE_FORMAT = "ISO8601"

# The feature TYPE used when a features sheet does not say.
DEFAULT_FEATURE_TYPE = "imported"


@dataclass(frozen=True)
class Sheet:
    """One dataset's name, columns and rules.

    Attributes:
        name: The sheet, file or table name, lower case.
        required: Columns every row must fill.
        optional: Known columns that may be left out or blank.
        dates: Columns that must hold dates.
        numbers: Columns that must hold numbers.
        positive: Numeric columns that must be above zero.
        key: Columns that together identify a row; no two rows may share
            them. Empty when repeats are allowed.
        needed: Whether the sheet must be supplied at all.
    """
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    dates: tuple[str, ...] = ()
    numbers: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()
    key: tuple[str, ...] = ()
    needed: bool = False


MARKET = Sheet("market",
               required=("IDENTIFIER", "DATE", "CLOSE"),
               optional=("OPEN", "HIGH", "LOW", "VOLUME",
                         "SHARES_OUTSTANDING", "FREE_FLOAT"),
               dates=("DATE",),
               numbers=("CLOSE", "OPEN", "HIGH", "LOW", "VOLUME",
                        "SHARES_OUTSTANDING", "FREE_FLOAT"),
               positive=("CLOSE",),
               key=("IDENTIFIER", "DATE"),
               needed=True)
REFERENCE = Sheet("reference",
                  required=("IDENTIFIER", "NAME", "CURRENCY", "EXCHANGE",
                            "DATE_FROM"),
                  optional=("DATE_TO",),
                  dates=("DATE_FROM", "DATE_TO"),
                  key=("IDENTIFIER", "DATE_FROM"),
                  needed=True)
FX = Sheet("fx",
           required=("PAIR", "DATE", "RATE"),
           dates=("DATE",),
           numbers=("RATE",),
           positive=("RATE",),
           key=("PAIR", "DATE"))
CORPORATE_ACTIONS = Sheet("corporate_actions",
                          required=("IDENTIFIER", "EX_DATE", "TYPE", "VALUE"),
                          optional=("PAY_DATE", "STATUS"),
                          dates=("EX_DATE", "PAY_DATE"),
                          numbers=("VALUE",))
FEATURES = Sheet("features",
                 required=("IDENTIFIER", "DATE", "FIELD", "VALUE"),
                 optional=("TYPE", "DETAIL"),
                 dates=("DATE",),
                 numbers=("VALUE",),
                 key=("IDENTIFIER", "DATE", "FIELD", "TYPE"))

# Columns whose values must come from a fixed set, by sheet. Matched without
# regard to case, and stored in the spelling given here.
CHOICES: dict[str, dict[str, frozenset[str]]] = {
    "corporate_actions": {"TYPE": frozenset(ACTION_TYPES),
                          "STATUS": frozenset(STATUSES)},
}

SHEETS: tuple[Sheet, ...] = (MARKET, REFERENCE, FX, CORPORATE_ACTIONS, FEATURES)
BY_NAME = {sheet.name: sheet for sheet in SHEETS}


@dataclass(frozen=True)
class Problem:
    """One thing wrong with the supplied data.

    Attributes:
        sheet: The sheet, file or table.
        row: The row as a spreadsheet shows it (the header is row 1), or None
            for a problem with the sheet as a whole.
        column: The column, or None.
        code: A stable code, such as MISSING_COLUMN or BAD_DATE.
        message: What is wrong, in a sentence.
    """
    sheet: str
    row: int | None
    column: str | None
    code: str
    message: str


class DataImportError(InvalidRuleError):
    """The supplied data has problems, and nothing was loaded.

    Attributes:
        problems: Up to `MAX_PROBLEMS` of them, each naming its sheet,
            row and column.
        total: How many problems there are in all.
        findings: The same problems in the shape every other refusal with
            findings uses (`path`, `severity`, `code`, `message`), plus
            `sheet`, `row` and `column`. This is what the API server sends.
    """
    def __init__(self,
                 problems: list[Problem],
                 total: int):
        self._problems = problems
        self.total = total
        self.findings = [_finding(problem) for problem in problems]

        super().__init__("data import",
                         f"{total} problem(s) in the supplied data")

    @property
    def problems(self) -> list[Problem]:
        return self._problems


def _finding(problem: Problem) -> dict[str, object]:
    """A problem as a finding: "market, row 4, DATE" and the rest."""
    place = [problem.sheet]

    if problem.row is not None:
        place.append(f"row {problem.row}")

    if problem.column is not None:
        place.append(problem.column)

    return {"path": ", ".join(place),
            "rule_id": None,
            "severity": "error",
            "code": problem.code,
            "message": problem.message,
            "sheet": problem.sheet,
            "row": problem.row,
            "column": problem.column}


def sheet_name(raw: str) -> str:
    """A sheet or file name as the layout spells it: "Corporate Actions" is
    `corporate_actions`."""
    return re.sub(r"[\s\-]+", "_", raw.strip().lower())


def tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Column names in the layout's spelling, rows numbered from zero, and
    fully blank rows (a common spreadsheet leftover) dropped."""
    frame = frame.rename(columns=lambda column: sheet_name(str(column)).upper())
    frame = frame.dropna(how="all")

    return frame.reset_index(drop=True)


def check(sheets: dict[str, pd.DataFrame]) -> tuple[list[Problem], int]:
    """Every problem in the supplied sheets.

    Args:
        sheets: Sheet name (lower case) to its rows, as read from the source.

    Returns:
        tuple: The first `MAX_PROBLEMS` problems, and how many there are in
        all. No problems means the data can be loaded.
    """
    problems = [Problem(name, None, None, "UNKNOWN_SHEET",
                        f"'{name}' is not a sheet this layout uses. "
                        f"Expected: {', '.join(BY_NAME)}.")
                for name in sheets if name not in BY_NAME]

    for sheet in SHEETS:
        if sheet.name in sheets:
            problems.extend(_check_sheet(sheet, sheets[sheet.name]))
        elif sheet.needed:
            problems.append(Problem(sheet.name, None, None, "MISSING_SHEET",
                                    f"The {sheet.name} sheet is required."))

    problems.extend(_check_identifiers(sheets))

    # Top to bottom, sheet by sheet, the way a person fixes a file.
    order = {sheet.name: position for position, sheet in enumerate(SHEETS)}
    problems.sort(key=lambda problem: (order.get(problem.sheet, -1),
                                       problem.row or 0,
                                       problem.column or ""))

    return problems[:MAX_PROBLEMS], len(problems)


def to_fetcher(sheets: dict[str, pd.DataFrame],
               **settings: Any) -> DataFetcher:
    """Build the data from sheets that passed `check`.

    FX pairs become market-data identifiers named by the pair, which is how
    every stored dataset holds them.

    Args:
        sheets: The checked sheets.
        **settings: Passed to `DataFetcher`: `fx_policy`,
            `max_price_staleness_days`, `free_float_backfill_days`.
    """
    market = _parsed(MARKET, sheets[MARKET.name])

    if FX.name in sheets:
        rates = _parsed(FX, sheets[FX.name]).rename(columns={"PAIR": "IDENTIFIER"})
        rates["IDENTIFIER"] = rates["IDENTIFIER"].str.upper()
        market = pd.concat([market, rates], ignore_index=True)

    actions = None
    if CORPORATE_ACTIONS.name in sheets:
        actions = CorporateActions.from_dataframe(
            _parsed(CORPORATE_ACTIONS, sheets[CORPORATE_ACTIONS.name]))

    features = None
    if FEATURES.name in sheets:
        frame = _parsed(FEATURES, sheets[FEATURES.name])

        if "TYPE" in frame:
            frame["TYPE"] = frame["TYPE"].where(~_blank(frame["TYPE"]),
                                                DEFAULT_FEATURE_TYPE)
        else:
            frame["TYPE"] = DEFAULT_FEATURE_TYPE

        features = FeatureData.from_dataframe(frame)

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(
                           _parsed(REFERENCE, sheets[REFERENCE.name])),
                       actions,
                       features,
                       **settings)


def _parsed(sheet: Sheet,
            frame: pd.DataFrame) -> pd.DataFrame:
    """A checked sheet with its text tidied and its dates and numbers converted."""
    parsed = frame.copy()

    for column in ("IDENTIFIER", "PAIR"):
        if column in parsed:
            parsed[column] = parsed[column].astype(str).str.strip()

    # `check` accepts a code in any case; it is stored as the library spells it.
    for column, allowed in CHOICES.get(sheet.name, {}).items():
        if column in parsed:
            spelling = {choice.lower(): choice for choice in allowed}
            parsed[column] = parsed[column].where(
                _blank(parsed[column]),
                parsed[column].astype(str).str.strip().str.lower().map(spelling))

    for column in sheet.dates:
        if column in parsed:
            parsed[column] = pd.to_datetime(parsed[column], errors="coerce", format=DATE_FORMAT)

    for column in sheet.numbers:
        if column in parsed:
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")

    return parsed


def _blank(values: pd.Series) -> pd.Series:
    """Which cells are empty: missing, or only spaces."""
    return values.isna() | values.astype(str).str.strip().eq("")


def _row(position: int) -> int:
    """The spreadsheet row of a data row: the header is row 1."""
    return position + 2


def _check_sheet(sheet: Sheet,
                 frame: pd.DataFrame) -> Iterable[Problem]:
    """The problems within one sheet."""
    missing = [column for column in sheet.required if column not in frame]

    if missing:
        yield Problem(sheet.name, None, ", ".join(missing), "MISSING_COLUMN",
                      f"The {sheet.name} sheet needs the column(s) "
                      f"{', '.join(missing)}.")
        return

    if frame.empty:
        yield Problem(sheet.name, None, None, "EMPTY_SHEET",
                      f"The {sheet.name} sheet has no rows.")
        return

    yield from _check_cells(sheet, frame)
    yield from _check_rules(sheet, frame)
    yield from _check_duplicates(sheet, frame)


def _check_cells(sheet: Sheet,
                 frame: pd.DataFrame) -> Iterable[Problem]:
    """Blank required cells, and cells of the wrong kind."""
    for column in sheet.required:
        for position in frame.index[_blank(frame[column])]:
            yield Problem(sheet.name, _row(position), column, "BLANK_VALUE",
                          f"{column} is empty.")

    for column in sheet.dates:
        if column not in frame:
            continue

        filled = ~_blank(frame[column])
        parsed = pd.to_datetime(frame[column], errors="coerce", format=DATE_FORMAT)

        for position in frame.index[filled & parsed.isna()]:
            yield Problem(sheet.name, _row(position), column, "BAD_DATE",
                          f"{column} '{frame[column][position]}' is not a date "
                          f"written as YYYY-MM-DD.")

    for column in sheet.numbers:
        if column not in frame:
            continue

        filled = ~_blank(frame[column])
        parsed = pd.to_numeric(frame[column], errors="coerce")

        for position in frame.index[filled & parsed.isna()]:
            yield Problem(sheet.name, _row(position), column, "BAD_NUMBER",
                          f"{column} '{frame[column][position]}' is not a "
                          f"number.")

        if column in sheet.positive:
            for position in frame.index[filled & (parsed <= 0)]:
                yield Problem(sheet.name, _row(position), column,
                              "OUT_OF_RANGE", f"{column} must be above zero.")


def _check_rules(sheet: Sheet,
                 frame: pd.DataFrame) -> Iterable[Problem]:
    """Rules particular to one sheet."""
    if sheet is MARKET and "FREE_FLOAT" in frame:
        values = pd.to_numeric(frame["FREE_FLOAT"], errors="coerce")

        for position in frame.index[(values < 0) | (values > 1)]:
            yield Problem(sheet.name, _row(position), "FREE_FLOAT",
                          "OUT_OF_RANGE",
                          "FREE_FLOAT is a fraction: between 0 and 1.")

    if sheet is REFERENCE and "DATE_TO" in frame:
        start = pd.to_datetime(frame["DATE_FROM"], errors="coerce", format=DATE_FORMAT)
        end = pd.to_datetime(frame["DATE_TO"], errors="coerce", format=DATE_FORMAT)

        for position in frame.index[end < start]:
            yield Problem(sheet.name, _row(position), "DATE_TO", "OUT_OF_RANGE",
                          "DATE_TO is before DATE_FROM.")

    if sheet is FX:
        pairs = frame["PAIR"].astype(str).str.strip()
        wrong = ~pairs.str.fullmatch(r"[A-Za-z]{6}")

        for position in frame.index[wrong & ~_blank(frame["PAIR"])]:
            yield Problem(sheet.name, _row(position), "PAIR", "BAD_PAIR",
                          f"PAIR '{pairs[position]}' is not two currency "
                          f"codes, such as GBPUSD.")

    for column, allowed in CHOICES.get(sheet.name, {}).items():
        if column in frame:
            yield from _check_choices(sheet, frame, column, allowed)


def _check_choices(sheet: Sheet,
                   frame: pd.DataFrame,
                   column: str,
                   allowed: Iterable[str]) -> Iterable[Problem]:
    """Cells that must be one of a fixed set of values, in any case."""
    choices = set(allowed)
    lowered = {choice.lower() for choice in choices}
    values = frame[column].astype(str).str.strip().str.lower()

    for position in frame.index[~_blank(frame[column]) & ~values.isin(lowered)]:
        yield Problem(sheet.name, _row(position), column, "UNKNOWN_VALUE",
                      f"{column} '{frame[column][position]}' is not one of "
                      f"{', '.join(sorted(choices))}.")


def _check_duplicates(sheet: Sheet,
                      frame: pd.DataFrame) -> Iterable[Problem]:
    """Rows that repeat another row's key."""
    key = [column for column in sheet.key if column in frame]

    if not key:
        return

    repeated = frame.duplicated(subset=key, keep="first")

    for position in frame.index[repeated]:
        described = ", ".join(f"{column} {frame[column][position]}"
                              for column in key)
        yield Problem(sheet.name, _row(position), None, "DUPLICATE_ROW",
                      f"Another row already has {described}.")


def _check_identifiers(sheets: dict[str, pd.DataFrame]) -> Iterable[Problem]:
    """Names priced or acted on that the reference sheet does not describe.

    Reported once per name, at its first row: a name missing from the
    reference sheet is one mistake, however many rows it has.
    """
    reference = sheets.get(REFERENCE.name)

    if reference is None or "IDENTIFIER" not in reference:
        return

    known = set(reference["IDENTIFIER"].astype(str).str.strip())

    for sheet in (MARKET, CORPORATE_ACTIONS, FEATURES):
        frame = sheets.get(sheet.name)

        if frame is None or "IDENTIFIER" not in frame:
            continue

        names = frame["IDENTIFIER"].astype(str).str.strip()
        first = ~names.duplicated() & ~names.isin(known) & ~_blank(frame["IDENTIFIER"])

        for position in frame.index[first]:
            yield Problem(sheet.name, _row(position), "IDENTIFIER",
                          "UNKNOWN_IDENTIFIER",
                          f"'{names[position]}' is not in the reference "
                          f"sheet.")
