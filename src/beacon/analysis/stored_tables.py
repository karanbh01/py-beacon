# src/beacon/analysis/stored_tables.py
"""
A stored run's tables: each read from its folder, from its data store's
shared folder, or worked out again from what it kept, the first time it is
asked for. `beacon.analysis.data.RunData` writes and opens a run with these.
"""
# BN-403, BN-412, BN-413: split from beacon.analysis.data.
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from .holding_replay import DROPS, PATCH, pack_holdings, unpack_holdings
from .panels import (
    ACTIONS,
    CLOSES,
    COSTS,
    CURRENCIES,
    FUND_WEIGHTS,
    HOLDINGS,
    NAME_RETURNS,
    RATES,
    TRADES,
    WEIGHT_TABLES,
    name_returns_from,
    read_panels,
    weights_from,
)
from .shared_indices import read as read_index
from .shared_inputs import SharedInputs, deleted_warning
from .trade_packing import PACKED_TRADES, pack_trades, unpack_trades

logger = logging.getLogger(__name__)

# The pieces of its data store's shared market inputs a run used (BN-413).
SHARED_PIECES = "shared_pieces"
# The fund's holdings, kept as what its trades do not give back (BN-413).
FUND_HOLDINGS = FUND_WEIGHTS + HOLDINGS
HOLDINGS_KEPT = {PATCH: FUND_HOLDINGS + "_patch", DROPS: FUND_HOLDINGS + "_drops"}
# A series is kept as a table of this one column.
VALUE = "value"


class StoredTables(Mapping[str, pd.DataFrame]):
    """A stored run's tables, each read from its folder, or worked out again
    from the ones it kept, the first time it is asked for."""

    def __init__(self,
                 kept: dict[str, pd.DataFrame],
                 rebuilds: dict[str, Callable[[], pd.DataFrame]]):
        self._kept = kept
        self._rebuilds = rebuilds
        self._windows: dict[str, Callable[[int], pd.DataFrame]] = {}

    def __getitem__(self,
                    name: str) -> pd.DataFrame:
        if name not in self._kept and name in self._rebuilds:
            self._kept[name] = self._rebuilds[name]()

        return self._kept[name]

    def __iter__(self) -> Iterator[str]:
        yield from self._kept
        yield from (name for name in self._rebuilds if name not in self._kept)

    def __len__(self) -> int:
        return len(set(self._kept) | set(self._rebuilds))

    def add(self,
            name: str,
            rebuild: Callable[[], pd.DataFrame],
            window: Callable[[int], pd.DataFrame] | None = None) -> None:
        """Have *name* read or worked out the first time it is asked for;
        *window* works out only its last rows, when it can."""
        self._rebuilds[name] = rebuild

        if window is not None:
            self._windows[name] = window

    def latest(self,
               name: str,
               rows: int) -> pd.DataFrame:
        """The last *rows* rows of *name*: worked out for those alone when it
        is worked out again and has not been already."""
        if name in self._kept or name not in self._windows:
            return self[name].tail(rows)

        return self._windows[name](rows)


def relative_folder(target: Path,
              start: Path) -> str:
    """*target* relative to *start*, so a run and its store's folder can move
    together; absolute when the two are on different drives."""
    try:
        return os.path.relpath(Path(target).resolve(), Path(start).resolve())
    except ValueError:
        return str(Path(target).resolve())


def add_shared(tables: "StoredTables",
                folder: Path,
                inputs: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Add to *tables* the market inputs a stored run keeps in its data
    store's shared folder, each read when first asked for; the warnings the
    run opens with."""
    if not inputs:
        return ()

    shared = SharedInputs((folder / inputs["folder"]).resolve())
    pieces: dict[str, pd.DataFrame] = {}

    def table(name: str) -> pd.DataFrame:
        if "all" not in pieces:
            pieces["all"] = read_panels(folder, [SHARED_PIECES])[SHARED_PIECES]

        held = pieces["all"]

        return shared.read(name, held.loc[held["table"] == name, "fingerprint"],
                           inputs["layouts"][name])

    for name in inputs["layouts"]:
        tables.add(name, partial(table, name))

    _add_index(tables, shared, inputs.get("index"))

    deleted = shared.deleted()

    if deleted is None:
        return ()

    warning = deleted_warning(deleted)
    logger.warning(warning)

    return (warning,)


def _add_index(tables: "StoredTables",
               shared: SharedInputs,
               index: Mapping[str, Any] | None) -> None:
    """Add to *tables* the index's results the run keeps in the shared
    folder, all read when the first is asked for (BN-414)."""
    if not index:
        return

    found: dict[str, dict[str, pd.DataFrame]] = {}

    def part(name: str) -> pd.DataFrame:
        if "all" not in found:
            found["all"] = read_index(shared.folder, index["key"], index["tables"])

        return found["all"][name]

    for name in index["tables"]:
        tables.add(name, partial(part, name))


def packed_trades(tables: Mapping[str, pd.DataFrame],
                   currency: str) -> tuple[dict[str, pd.DataFrame], float] | None:
    """A run's trades packed, and the fixed rate; None when the run has no
    trades or lacks what they are worked out from, or they would not come
    back exactly."""
    if TRADES not in tables or not all(name in tables for name in (CLOSES, CURRENCIES, RATES,
                                                                 ACTIONS)):
        return None

    return pack_trades(tables[TRADES], tables, currency)


def packed_holdings(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame] | None:
    """The fund's holdings as what its trades do not give back; None when
    the run lacks them or what they are worked out from, or they would not
    come back exactly."""
    if not all(name in tables for name in (FUND_HOLDINGS, TRADES, ACTIONS)):
        return None

    return pack_holdings(tables[FUND_HOLDINGS], tables[TRADES], tables[ACTIONS])


def holdings_from(folder: Path,
                   tables: Mapping[str, pd.DataFrame],
                   run: Mapping[str, Any]) -> pd.DataFrame:
    """A stored run's fund holdings, worked out again from its trades."""
    read = read_panels(folder, list(HOLDINGS_KEPT.values()))

    return unpack_holdings(tables[TRADES], tables[ACTIONS],
                           {part: read[name] for part, name in HOLDINGS_KEPT.items()},
                           run["holdings"]["dtypes"])


def trades_from(folder: Path,
                 tables: Mapping[str, pd.DataFrame],
                 run: Mapping[str, Any]) -> pd.DataFrame:
    """A stored run's trades, worked out again from what it kept."""
    return unpack_trades(read_panels(folder, list(PACKED_TRADES)), tables, run["currency"],
                         float(run["trades"]["rate"]), list(run["trades"]["columns"]))


def read_one(folder: Path,
              name: str) -> pd.DataFrame:
    """One stored table."""
    return read_panels(folder, [name])[name]


def rebuilds_of(tables: Mapping[str, pd.DataFrame],
              run: Mapping[str, Any]) -> dict[str, Callable[[], pd.DataFrame]]:
    """How each table a stored run worked out again is rebuilt from the ones
    it kept."""
    rebuilds: dict[str, Callable[[], pd.DataFrame]] = {}

    if NAME_RETURNS in run.get("derived", []):
        # The run's days are the costs table's, a row for each.
        rebuilds[NAME_RETURNS] = lambda: name_returns_from(tables, run["currency"],
                                                           pd.DatetimeIndex(tables[COSTS].index))

    for name in WEIGHT_TABLES:
        if name in run.get("derived", []):
            rebuilds[name] = partial(weights_from, name, tables, run["currency"],
                                     run.get("dtypes", {}).get(name, "float64"))

    return rebuilds


def window_of(name: str,
            tables: Mapping[str, pd.DataFrame],
            run: Mapping[str, Any]) -> Callable[[int], pd.DataFrame] | None:
    """How a stored run's daily weights are worked out for their last days
    alone; None for any other table."""
    if name not in WEIGHT_TABLES:
        return None

    return partial(latest_weights, name, tables, run["currency"],
                   run.get("dtypes", {}).get(name, "float64"))


def latest_weights(name: str,
                    tables: Mapping[str, pd.DataFrame],
                    currency: str,
                    dtype: str,
                    rows: int) -> pd.DataFrame:
    """*name*'s daily weights on its last *rows* days."""
    return weights_from(name, tables, currency, dtype, last=rows)


def column_of(table: pd.DataFrame) -> pd.Series:
    """A series kept as a table of one column."""
    column = table[VALUE]
    column.name = None

    return column
