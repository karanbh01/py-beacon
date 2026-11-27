# src/beacon/analysis/performance/style_store.py
"""
Where each name held sat in the style box at each rebalance, kept beside the
run, so a name's profile reads it rather than placing every name held at
every rebalance again.

    rebalance_positions(snapshots, weights, settings, usd_rate)   # worked out
    keep(tables, settings, folder)                                # after working it out
    restore(settings, folder)                                     # before, on a stored run

``style_positions`` has a row for each name a portfolio held and that could
be sized, at each date the run kept characteristics on, as `style_positions`
places the names held that day: ``size_position`` and ``style_position``,
from 0 to 3, and the cell, ``size`` (large, mid or small) and ``style``
(value, blend or growth). ``style_dates`` has every such date, a name placed
on it or not. They are kept keyed by the style settings, so a run asked
about other size bands or another growth measure works them out again.
"""
# BN-481 (Karan, 2026-11-27): worked out once with the analysis, as the
# risk model's work is (risk_store), so Drilldown does not place every name
# held at every rebalance each time a name is opened.
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...data.base import as_of_position
from ..panels import read_manifest, read_panels, write_panels
from .style import SIZES, STYLES, StyleSettings, cells, style_positions

FOLDER = "style"
KEY_FILE = "style.json"
VERSION = 1

TABLE, DATES = "style_positions", "style_dates"
TABLES = (TABLE, DATES)
IDENTIFIER, DATE = "IDENTIFIER", "DATE"
COLUMNS = ["size_position", "style_position", "size", "style"]


def rebalance_positions(snapshots: pd.DataFrame,
                        weights: Mapping[str, pd.DataFrame | None],
                        settings: StyleSettings,
                        usd_rate: Callable[[pd.Timestamp], float]) -> dict[str, pd.DataFrame]:
    """Each name's place in the style box at each date of *snapshots* (the
    run's characteristics, a row a name a date), among the names each of
    *weights*' portfolios held that day, with caps put in dollars at
    *usd_rate*'s rate that day."""
    dates = pd.DatetimeIndex(pd.unique(snapshots.index)).sort_values()
    pieces = []

    for day in dates:
        snapshot = snapshots.loc[[day]].set_index(IDENTIFIER)
        snapshot.index = snapshot.index.astype(str)
        held = {portfolio: None if found is None else weights_on(found, day)
                for portfolio, found in weights.items()}
        placed = style_positions(snapshot, held, settings, usd_rate(day))
        pieces.append(_cells(placed).assign(**{DATE: day}))

    table = (pd.concat(pieces, ignore_index=True) if pieces
             else pd.DataFrame(columns=[IDENTIFIER, *COLUMNS, DATE]))

    return {TABLE: table.set_index(DATE)[[IDENTIFIER, *COLUMNS]],
            DATES: pd.DataFrame(index=pd.DatetimeIndex(dates, name=DATE))}


def weights_on(weights: pd.DataFrame,
               day: pd.Timestamp) -> pd.Series:
    """A portfolio's weights on *day*, or its latest before it."""
    position = as_of_position(weights.index, day)
    row = weights.iloc[position if position is not None else 0].fillna(0.0)
    row.index = row.index.astype(str)

    return row


def _cells(placed: pd.DataFrame) -> pd.DataFrame:
    """The names placed, with their positions and cells."""
    size = placed["size"].astype(float)
    position = placed["style"].astype(float)

    return pd.DataFrame({IDENTIFIER: placed["name"].astype(str).to_numpy(),
                         "size_position": size.to_numpy(),
                         "style_position": position.to_numpy(),
                         "size": np.array(SIZES, dtype=object)[cells(size)],
                         "style": np.array(STYLES, dtype=object)[cells(position)]})


def fingerprint(settings: StyleSettings) -> str:
    """A key for the style *settings*."""
    text = json.dumps({"version": VERSION, "settings": asdict(settings)}, sort_keys=True,
                      default=str)

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keep(tables: dict[str, pd.DataFrame],
         settings: StyleSettings,
         folder: Path) -> bool:
    """Keep *tables* under *folder*; False, keeping nothing, when the run
    kept no characteristics to place its names by."""
    if not len(tables[DATES].index):
        return False

    write_panels(folder / FOLDER, tables)
    (folder / FOLDER / KEY_FILE).write_text(json.dumps({"key": fingerprint(settings)}),
                                            encoding="utf-8")

    return True


def restore(settings: StyleSettings,
            folder: Path) -> dict[str, pd.DataFrame] | None:
    """The tables kept under *folder* for these settings; None when none
    were."""
    path = folder / FOLDER / KEY_FILE

    if not path.exists() or read_manifest(folder / FOLDER) is None:
        return None

    kept: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    if kept.get("key") != fingerprint(settings):
        return None

    tables = read_panels(folder / FOLDER, TABLES)
    tables[TABLE][IDENTIFIER] = tables[TABLE][IDENTIFIER].astype(str)

    return tables


__all__ = ["fingerprint", "keep", "rebalance_positions", "restore", "weights_on"]
