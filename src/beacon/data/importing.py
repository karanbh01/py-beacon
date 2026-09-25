# src/beacon/data/importing.py
"""Load data from CSV files or an Excel workbook, in the layout `beacon.data.layout` sets out.

Give CSV files named after their sheet (`market.csv`, `reference.csv`, ...),
or one Excel workbook whose sheets carry those names. Names are matched
without regard to case or spaces, so a sheet called "Corporate Actions" is
the `corporate_actions` sheet, and a column called "close" is `CLOSE`.

    from beacon.data import importing

    data = importing.load_files(["market.csv", "reference.csv"])

Everything is checked before anything is loaded. If any row is wrong,
`load_files` raises `DataImportError` listing every problem, each with its
sheet, row and column, so the files can be fixed in one pass.

`template` writes a blank template to start from, as an Excel workbook or a
zip of CSV files.
"""
import io
import re
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .._optional import require
from ..exceptions import InvalidRuleError
from . import layout
from .fetcher import DataFetcher

EXCEL_SUFFIXES = (".xlsx", ".xlsm")
CSV_SUFFIX = ".csv"
TEMPLATE_FORMATS = ("xlsx", "csv")

# One made-up row per sheet, so the template shows what a filled row looks
# like, not only its headers.
EXAMPLES: dict[str, dict[str, object]] = {
    "market": {"IDENTIFIER": "AAA", "DATE": "2024-01-02", "CLOSE": 100.0,
               "VOLUME": 1_000_000, "SHARES_OUTSTANDING": 5_000_000,
               "FREE_FLOAT": 0.9},
    "reference": {"IDENTIFIER": "AAA", "NAME": "Example Company",
                  "CURRENCY": "USD", "EXCHANGE": "XNYS",
                  "DATE_FROM": "2020-01-01", "SECTOR": "Technology"},
    "fx": {"PAIR": "GBPUSD", "DATE": "2024-01-02", "RATE": 1.27},
    "corporate_actions": {"IDENTIFIER": "AAA", "EX_DATE": "2024-03-15",
                          "TYPE": "DIVIDEND", "VALUE": 0.5,
                          "PAY_DATE": "2024-03-29", "STATUS": "PAID"},
    "features": {"IDENTIFIER": "AAA", "DATE": "2024-01-02", "FIELD": "revenue",
                 "VALUE": 1_000_000, "TYPE": "fundamentals"},
}


class DataImportError(InvalidRuleError):
    """The supplied data has problems, and nothing was loaded.

    Attributes:
        problems: Up to `layout.MAX_PROBLEMS` of them, each naming its sheet,
            row and column.
        total: How many problems there are in all.
        findings: The same problems in the shape every other refusal with
            findings uses (`path`, `severity`, `code`, `message`), plus
            `sheet`, `row` and `column`. This is what the API server sends.
    """
    def __init__(self,
                 problems: list[layout.Problem],
                 total: int):
        self._problems = problems
        self.total = total
        self.findings = [_finding(problem) for problem in problems]

        super().__init__("data import",
                         f"{total} problem(s) in the supplied data")

    @property
    def problems(self) -> list[layout.Problem]:
        return self._problems


def _finding(problem: layout.Problem) -> dict[str, object]:
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


def read(paths: Iterable[str | Path]) -> tuple[dict[str, pd.DataFrame],
                                               list[layout.Problem]]:
    """Read the files into sheets, and the problems reading them.

    Every value is read as text, so the checks see exactly what was written,
    and a date or number in the wrong form is reported rather than guessed.
    """
    sheets: dict[str, pd.DataFrame] = {}
    problems: list[layout.Problem] = []

    for raw in paths:
        path = Path(raw)

        try:
            found = _read_one(path)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            problems.append(layout.Problem(path.name, None, None,
                                           "UNREADABLE_FILE",
                                           f"{path.name} cannot be read: "
                                           f"{error}"))
            continue

        for name, frame in found.items():
            if name in sheets:
                problems.append(layout.Problem(name, None, None,
                                               "DUPLICATE_SHEET",
                                               f"The {name} sheet was given "
                                               f"more than once."))
                continue

            sheets[name] = frame

    return sheets, problems


def load_files(paths: Iterable[str | Path]) -> DataFetcher:
    """Read, check and load CSV files or an Excel workbook.

    Raises:
        DataImportError: If a file cannot be read or any row has a problem.
            Nothing is loaded in that case.
    """
    sheets, problems = read(paths)
    found, total = layout.check(sheets)
    problems = [*problems, *found]
    total += len(problems) - len(found)

    if problems:
        raise DataImportError(problems[:layout.MAX_PROBLEMS], total)

    return layout.to_fetcher(sheets)


def template(fmt: str = "xlsx") -> bytes:
    """A blank template with every sheet's columns and one example row.

    Args:
        fmt: "xlsx" for an Excel workbook, or "csv" for a zip of CSV files.

    Raises:
        ValueError: For any other format.
    """
    frames = {sheet.name: pd.DataFrame([EXAMPLES[sheet.name]],
                                       columns=[*sheet.required, *sheet.optional,
                                                *[column for column
                                                  in EXAMPLES[sheet.name]
                                                  if column not in sheet.required
                                                  and column not in sheet.optional]])
              for sheet in layout.SHEETS}
    buffer = io.BytesIO()

    if fmt == "xlsx":
        require("openpyxl", "Excel templates")

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for name, frame in frames.items():
                frame.to_excel(writer, sheet_name=name, index=False)
    elif fmt == "csv":
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, frame in frames.items():
                archive.writestr(f"{name}.csv", frame.to_csv(index=False))
    else:
        raise ValueError(f"'{fmt}' is not a template format. Use one of "
                         f"{', '.join(TEMPLATE_FORMATS)}.")

    return buffer.getvalue()


def _read_one(path: Path) -> dict[str, pd.DataFrame]:
    """One file's sheets, by layout name."""
    suffix = path.suffix.lower()

    if suffix == CSV_SUFFIX:
        return {sheet_name(path.stem): _tidy(pd.read_csv(path, dtype=str))}

    if suffix in EXCEL_SUFFIXES:
        require("openpyxl", "Excel import")
        workbook = pd.read_excel(path, sheet_name=None, dtype=str)

        return {sheet_name(name): _tidy(frame) for name, frame in workbook.items()}

    raise ValueError(f"it is not a CSV ({CSV_SUFFIX}) or Excel "
                     f"({', '.join(EXCEL_SUFFIXES)}) file")


def _tidy(frame: pd.DataFrame) -> pd.DataFrame:
    """Column names in the layout's spelling, rows numbered from zero, and
    fully blank rows (a common spreadsheet leftover) dropped."""
    frame = frame.rename(columns=lambda column: sheet_name(str(column)).upper())
    frame = frame.dropna(how="all")

    return frame.reset_index(drop=True)
