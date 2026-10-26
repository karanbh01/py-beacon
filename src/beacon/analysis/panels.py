# src/beacon/analysis/panels.py
"""
The large tables a run's analysis reads, kept as Parquet files in a folder.

A run's record is a JSON document: its summary and its small series. The
tables that grow with the number of names and days, such as each day's weights
in every name held, are too large for JSON and slow to read back from it, so
they are kept beside the record as Parquet files, one per table, with a small
manifest saying what is there.

Each table is wide, dates by name, with a date index. It reads back exactly:
the same values, names and dates. A record kept before tables were written
has no folder, and reading one gives no tables rather than an error.
"""
# BN-361. Wide rather than long: at 6,000 names over 1,750 days one column per
# name wrote in 1.2 s and read in 0.6 s, against 1.6 s and 1.3 s for a long
# table pivoted back, at the same size.
import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ..backtest.breakdown import daily_cash, daily_costs
from ..backtest.result import BacktestResult
from ..data.returns import name_returns
from ..data.store import _read_frame, _write_frame
from ..exceptions import CalculationError, ConfigurationError, DataNotFoundError

logger = logging.getLogger(__name__)

# Bump when the folder's layout changes. A folder written by a newer version
# is refused rather than misread.
PANELS_VERSION = 1

MANIFEST_FILE = "panels.json"

# The column the date index is kept in.
INDEX_COLUMN = "DATE"

# A table's name is its file's name, so it is kept to plain characters.
_NAME = re.compile(r"[a-z][a-z0-9_]*")

# The tables a run writes, by name.
FUND_WEIGHTS = "fund_weights"
NAME_RETURNS = "name_returns"
COSTS = "costs"
CASH = "cash"

# How far before the run's first day name returns are read from, so its first
# day has a close before it.
_LEAD = pd.Timedelta(days=14)


def run_panels(result: BacktestResult) -> dict[str, pd.DataFrame]:
    """The tables a finished run keeps beside its record.

    Args:
        result: The finished run.

    Returns:
        dict: Each table by name:

        - ``fund_weights``: the fund's weight in each name held on each day
          it was recorded, unabridged.
        - ``name_returns``: each day's total return, in the book's currency,
          of every name the fund held or its index or benchmark held, as
          `beacon.data.returns.name_returns` measures it. Left out, with a
          warning, when the run kept no data to read them from or they cannot
          be measured (a currency with no rate into the book's).
        - ``costs``: what the run cost each day, by kind, as
          `beacon.backtest.breakdown.daily_costs` gives it.
        - ``cash``: its cash, income, fees and flows each day, as
          `beacon.backtest.breakdown.daily_cash` gives them.
    """
    weights = result.portfolio.weights
    panels = {FUND_WEIGHTS: weights, COSTS: daily_costs(result), CASH: daily_cash(result)}
    source = result.portfolio.source
    days = result.portfolio.nav.index

    if source is None or len(days) == 0:
        return panels

    # A table that cannot be built is left out rather than failing the run
    # it would have been kept with; its analysis then says the returns are
    # missing.
    try:
        returns = name_returns(source, held_names(result),
                               str((days[0] - _LEAD).date()), str(days[-1].date()),
                               result.currency)
    except (CalculationError, DataNotFoundError) as error:
        logger.warning("The run's name returns were not kept: %s", error)
        return panels

    panels[NAME_RETURNS] = returns.reindex(days).rename(columns=str)

    return panels


def held_names(result: BacktestResult) -> list[str]:
    """Every name the fund, its index (both books) or its benchmark held at
    any point in the run, sorted."""
    books = [result.index.target, result.index.optimised, result.benchmark]
    frames = [result.portfolio.weights] + [book.weights for book in books if book is not None]

    return sorted({str(name) for frame in frames for name in frame.columns})


def write_panels(folder: Path,
                 panels: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Write *panels* to *folder*, replacing whatever it held.

    The tables are written to a new folder beside it, which then takes its
    place, so a failure part way leaves the previous tables as they were.

    Args:
        folder: Where the tables go.
        panels: Each table by name: dates by name, with a date index.

    Returns:
        dict: The manifest written beside them.

    Raises:
        ConfigurationError: If a table's name is not lower-case letters,
            digits and underscores, or its columns are not text.
    """
    for name, frame in panels.items():
        _check(name, frame)

    folder.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=folder.parent, prefix=f".{folder.name}."))

    try:
        manifest: dict[str, Any] = {"version": PANELS_VERSION, "panels": {}}

        for name, frame in panels.items():
            _write_frame(frame.rename_axis(INDEX_COLUMN).reset_index(), staging / f"{name}.parquet")
            manifest["panels"][name] = {"rows": len(frame),
                                        "columns": len(frame.columns),
                                        "index": frame.index.name}

        (staging / MANIFEST_FILE).write_text(json.dumps(manifest, sort_keys=True),
                                             encoding="utf-8")
        _replace(folder, staging)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return manifest


def read_manifest(folder: Path) -> dict[str, Any] | None:
    """What *folder* holds, or None when there is no such folder.

    Raises:
        ConfigurationError: If the manifest cannot be read, or was written by
            a newer version than this one reads.
    """
    path = folder / MANIFEST_FILE

    if not path.exists():
        return None

    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(str(path), f"the run's tables cannot be read: {error}") from error

    version = int(manifest.get("version", 0))

    if version > PANELS_VERSION:
        raise ConfigurationError(
            str(path),
            f"the run's tables were written in version {version} of their layout, newer "
            f"than this version of Beacon reads ({PANELS_VERSION}). Upgrade py-beacon-kit.")

    return manifest


def read_panels(folder: Path,
                names: Iterable[str] | None = None) -> dict[str, pd.DataFrame]:
    """Read tables back from *folder*.

    Args:
        folder: Where the tables were written.
        names: The tables wanted; every one there when None.

    Returns:
        dict: Each table by name, as it was written. Empty when there is no
        folder, as for a record kept before tables were.

    Raises:
        DataNotFoundError: If a table asked for by name is not there.
        ConfigurationError: If the folder cannot be read.
    """
    manifest = read_manifest(folder)

    if manifest is None:
        return {}

    held: dict[str, Any] = manifest.get("panels", {})
    wanted = list(held) if names is None else list(names)
    missing = [name for name in wanted if name not in held]

    if missing:
        raise DataNotFoundError(f"the run's {', '.join(missing)} table",
                                source=f"its tables hold {', '.join(sorted(held)) or 'none'}")

    return {name: _read_panel(folder / f"{name}.parquet", held[name]) for name in wanted}


def _read_panel(path: Path,
                entry: Mapping[str, Any]) -> pd.DataFrame:
    """One table, its date index restored under its own name."""
    frame = _read_frame(path).set_index(INDEX_COLUMN)
    frame.index.name = entry.get("index")
    frame.columns = frame.columns.astype(object)

    return frame


def _check(name: str,
           frame: pd.DataFrame) -> None:
    """Refuse a table whose name or columns would not survive the file."""
    if not _NAME.fullmatch(name):
        raise ConfigurationError(f"table '{name}'",
                                 "a table's name is lower-case letters, digits and underscores")

    if not all(isinstance(column, str) for column in frame.columns):
        raise ConfigurationError(f"table '{name}'", "its columns must be named by text")

    if INDEX_COLUMN in frame.columns:
        raise ConfigurationError(f"table '{name}'", f"'{INDEX_COLUMN}' names its index's column")


def _replace(folder: Path,
             staging: Path) -> None:
    """Put *staging* where *folder* is.

    A folder cannot be swapped for another in one step on every platform, so
    the old one is moved aside first and removed once the new one is in.
    """
    retired = None

    if folder.exists():
        retired = folder.with_name(f".{folder.name}.old")
        shutil.rmtree(retired, ignore_errors=True)
        os.replace(folder, retired)

    os.replace(staging, folder)

    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)
