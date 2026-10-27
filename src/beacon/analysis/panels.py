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

from ..backtest.breakdown import daily_cash, daily_costs, trades
from ..backtest.result import BacktestResult
from ..data.prices import PRICE_COLUMN, converted
from ..data.returns import name_returns
from ..data.store import _read_frame, _write_frame
from ..exceptions import CalculationError, ConfigurationError, DataNotFoundError
from .holdings import (
    fund_weights_from,
    holding_changes,
    index_weights_from,
    patch_of,
    patched,
    rate_grid,
    wide_of,
)
from .packing import StoredSource, market_inputs
from .snapshots import (
    characteristics_snapshots,
    liquidity_dates,
    market_history,
    rebalance_dates,
    reference_snapshots,
    traded_value_snapshots,
)

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
INDEX_WEIGHTS = "index_weights"
OPTIMISED_WEIGHTS = "optimised_weights"
BENCHMARK_WEIGHTS = "benchmark_weights"
REFERENCE = "reference"
CHARACTERISTICS = "characteristics"
TRADED_VALUES = "traded_values"
# The run's trades, each with why it was made (BN-406).
TRADES = "trades"

# The market inputs the name returns are worked out from (BN-403).
CLOSES = "closes"
CURRENCIES = "currencies"
RATES = "rates"
ACTIONS = "actions"

# What a book's daily weights are worked out again from (BN-403): its
# holdings on the days they changed, its days and names, its total value
# each day (the fund's), and the cells kept as recorded.
HOLDINGS, DAYS, NAMES, TOTALS, PATCH = "_holdings", "_days", "_names", "_totals", "_patch"
WEIGHT_TABLES = (FUND_WEIGHTS, INDEX_WEIGHTS, OPTIMISED_WEIGHTS, BENCHMARK_WEIGHTS)



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
        - ``trades``: every trade, why it was made and its tags, as
          `beacon.backtest.breakdown.trades` gives them.
        - ``costs``: what the run cost each day, by kind, as
          `beacon.backtest.breakdown.daily_costs` gives it.
        - ``cash``: its cash, income, fees and flows each day, as
          `beacon.backtest.breakdown.daily_cash` gives them.
        - ``index_weights``, ``optimised_weights`` and ``benchmark_weights``:
          each comparator's daily weights in its names, for those the run has
          and that hold names (a benchmark given as a bare level series has
          none).
        - ``reference``: the grouping fields (sector, industry, region,
          country, currency, exchange) of every name held, as they stood at
          each rebalance; see `beacon.analysis.snapshots`.
        - ``characteristics``: their market cap, P/E, P/B, dividend yield and
          momentum at each rebalance, as
          `beacon.analysis.snapshots.characteristics_snapshots` gives them;
          left out, with a warning, when they cannot be measured.
        - ``closes``, ``currencies``, ``rates`` and ``actions``: the closes
          the name returns are worked out from, in each name's own currency,
          and what converts them and accounts for splits and cash; see
          `beacon.analysis.packing`.
        - ``traded_values``: their average daily traded value at each
          rebalance and month end, as the liquidity screens measure it
          (`beacon.analysis.snapshots.traded_value_snapshots`).
    """
    weights = result.portfolio.weights
    panels = {FUND_WEIGHTS: weights, COSTS: daily_costs(result), CASH: daily_cash(result),
              TRADES: trades(result)}
    comparators = {INDEX_WEIGHTS: result.index.target, OPTIMISED_WEIGHTS: result.index.optimised,
                   BENCHMARK_WEIGHTS: result.benchmark}

    for name, book in comparators.items():
        if book is not None and not book.weights.empty:
            panels[name] = book.weights.rename(columns=str)
    source = result.portfolio.source
    days = result.portfolio.nav.index

    if source is None or len(days) == 0:
        return panels

    names = held_names(result)
    dates = rebalance_dates(result)
    panels[REFERENCE] = reference_snapshots(source, names, dates)
    # One read of the market data for every table built from it.
    market = market_history(source, names, dates or [days[0]], until=days[-1])

    closes = market[PRICE_COLUMN].rename(columns=str)
    panels[CLOSES] = closes
    panels.update(market_inputs(source, closes, result.currency))

    # A table that cannot be built is left out rather than failing the run
    # it would have been kept with; its analysis then says what is missing.
    try:
        panels[NAME_RETURNS] = name_returns_from(panels, result.currency, days)
    except (CalculationError, DataNotFoundError) as error:
        logger.warning("The run's name returns were not kept: %s", error)

    panels.update(weights_inputs(result, panels))

    try:
        panels[CHARACTERISTICS] = characteristics_snapshots(source, names, dates,
                                                            result.currency, market)
    except (CalculationError, DataNotFoundError) as error:
        logger.warning("The run's characteristics were not kept: %s", error)

    try:
        panels[TRADED_VALUES] = traded_value_snapshots(source, names, liquidity_dates(result),
                                                       result.currency, market)
    except (CalculationError, DataNotFoundError) as error:
        logger.warning("The run's traded values were not kept: %s", error)

    return panels


def name_returns_from(tables: Mapping[str, pd.DataFrame],
                      currency: str,
                      days: pd.DatetimeIndex) -> pd.DataFrame:
    """The name returns, worked out from a run's kept closes, currencies,
    rates and corporate actions, on its NAV *days* from the first kept close:
    a year before the fund's first holding, enough for a risk model's window.

    The same calculation as `beacon.data.returns.name_returns` on the data
    store, on the same numbers, so a finished run and a stored one give the
    same table to the last bit.
    """
    stored = StoredSource(tables[CURRENCIES], tables[RATES], tables[ACTIONS], currency)
    closes = tables[CLOSES]
    # No rows before the closes start, which would all be blank (BN-403).
    kept = days[days >= closes.index[0]] if len(closes.index) else days

    return (name_returns(stored, list(closes.columns), currency=currency, closes=closes)
            .reindex(kept).rename(columns=str))


def weights_inputs(result: BacktestResult,
                   tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """What each book's daily weights are worked out again from: the fund's
    quantities and daily totals, each index's units, and for every book its
    days, its names and the cells that differ from what was recorded."""
    kept: dict[str, pd.DataFrame] = {}
    closes = tables[CLOSES].ffill()
    currency = result.currency
    stored = StoredSource(tables[CURRENCIES], tables[RATES], tables[ACTIONS], currency)
    weights = result.portfolio.weights

    if not weights.empty:
        quantities = wide_of(result.portfolio.positions, "DATE", "ASSET_ID", "QUANTITY")
        changes = holding_changes(quantities)
        totals = result.portfolio.nav
        rebuilt = fund_weights_from(changes, totals, converted(stored, closes, currency),
                                    pd.DatetimeIndex(weights.index), list(weights.columns))
        kept.update(_book_tables(FUND_WEIGHTS, weights, changes, rebuilt))
        kept[FUND_WEIGHTS + TOTALS] = totals.to_frame("TOTAL")

    books = {INDEX_WEIGHTS: result.index.target, OPTIMISED_WEIGHTS: result.index.optimised,
             BENCHMARK_WEIGHTS: result.benchmark}

    for name, book in books.items():
        if book is None or book.source is None or book.weights.empty:
            continue

        daily = book.source.daily_weights
        units = wide_of(daily, "DATE", "IDENTIFIER", "AMOUNT")
        changes = holding_changes(units)
        recorded = book.weights
        names = [str(column) for column in recorded.columns]
        days = pd.DatetimeIndex(recorded.index)
        rebuilt = index_weights_from(changes, closes, rate_grid(tables[CURRENCIES], tables[RATES],
                                                               days, names),
                                     days, names, str(recorded.dtypes.iloc[0]))
        kept.update(_book_tables(name, recorded, changes, rebuilt))

    return kept


def _book_tables(name: str,
                 recorded: pd.DataFrame,
                 changes: pd.DataFrame,
                 rebuilt: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """A book's kept holdings, days, names and patch."""
    return {name + HOLDINGS: changes,
            name + DAYS: pd.DataFrame(index=pd.DatetimeIndex(recorded.index)),
            name + NAMES: pd.DataFrame(index=pd.Index([str(item) for item in recorded.columns])),
            name + PATCH: patch_of(rebuilt, recorded)}


def weights_from(name: str,
                 tables: Mapping[str, pd.DataFrame],
                 currency: str,
                 dtype: str,
                 last: int | None = None) -> pd.DataFrame:
    """A book's daily weights, worked out again from what was kept, with the
    recorded cells put back, so they are exactly what was recorded; for its
    *last* days alone when given, which are those days of the whole table."""
    days = pd.DatetimeIndex(tables[name + DAYS].index)
    days = days[-last:] if last is not None and last > 0 else days
    # Carried over gaps across the whole run, then only the days worked out.
    closes = tables[CLOSES].ffill().reindex(days)
    names = [str(item) for item in tables[name + NAMES].index]
    changes = tables[name + HOLDINGS]
    patch = tables[name + PATCH]
    patch = patch[pd.DatetimeIndex(patch.index) >= days[0]] if len(days) else patch

    if name == FUND_WEIGHTS:
        stored = StoredSource(tables[CURRENCIES], tables[RATES], tables[ACTIONS], currency)
        rebuilt = fund_weights_from(changes, tables[name + TOTALS]["TOTAL"],
                                    converted(stored, closes, currency), days, names)
    else:
        rebuilt = index_weights_from(changes, closes,
                                     rate_grid(tables[CURRENCIES], tables[RATES], days, names),
                                     days, names, dtype)

    frame = patched(rebuilt, patch)
    frame.index.name, frame.columns.name = None, None

    return frame.astype(dtype)


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
            _write_frame(frame.rename_axis(INDEX_COLUMN).reset_index(), staging / f"{name}.parquet",
                         delta_integers=True)
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
