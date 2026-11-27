# src/beacon/analysis/analytics/profile/characteristics.py
"""A name's profile's style, key data and corporate actions, from what the
run kept of it."""
# BN-472 (Karan, 2026-11-25).
from typing import Any

import numpy as np
import pandas as pd

from ...panels import ACTIONS, CHARACTERISTICS, REFERENCE, TRADED_VALUES
from ...performance import style_store
from ...reports.statistics import RUN_PREDATES, TOO_SHORT, Unavailable

IDENTIFIER = "IDENTIFIER"
STYLE_COLUMNS = ["size_position", "style_position", "size", "style"]


def style_positions(name: str,
                    context: Any,
                    days: pd.DatetimeIndex) -> pd.DataFrame:
    """At each rebalance over *days* the run kept characteristics on: where
    *name* sat in the style box, by position and by cell, as
    `instrument.style_positions()` places the names held, and whether its
    cell differs from the one before; blank where no portfolio held it or
    it had no market cap.

    Raises:
        Unavailable: If the run was kept before its characteristics were.
    """
    # BN-481: the name's rows of the style box at each rebalance, worked out
    # once for every name.
    kept = context.rebalance_style
    dates = _within(kept[style_store.DATES].index, days)
    placed = kept[style_store.TABLE]
    found = (placed[placed[IDENTIFIER] == name][STYLE_COLUMNS].reindex(dates)
             .astype({"size_position": float, "style_position": float,
                      "size": object, "style": object}))
    # Read back from the kept table, its column names are strings.
    found.columns = pd.Index(STYLE_COLUMNS, dtype=object)
    found[["size", "style"]] = found[["size", "style"]].where(found[["size", "style"]].notna(),
                                                             None)
    boxes = pd.Series([None if size is None else f"{size} {cell}"
                       for size, cell in zip(found["size"], found["style"], strict=True)],
                      index=found.index, dtype=object)
    known = boxes.dropna()
    moved = known.ne(known.shift()) & known.shift().notna()
    found["moved"] = moved.reindex(found.index).fillna(False).astype(bool)

    return found.rename_axis("DATE")


def key_data(name: str,
             context: Any,
             days: pd.DatetimeIndex) -> pd.DataFrame:
    """At each rebalance over *days*: the reference fields the run kept for
    *name* (its currency, exchange, sector and the others the run grouped
    by), its market cap and average daily traded value in the portfolio's
    currency, and, with the data store loaded, its name and free float.

    Raises:
        Unavailable: If the run kept no reference for its rebalances.
    """
    tables = context.run.tables
    reference = tables.get(REFERENCE)

    if reference is None or reference.empty:
        raise Unavailable(RUN_PREDATES)

    dates = _within(reference.index, days)
    mine = _rows_of(reference, name).reindex(dates)
    found = mine.rename(columns=str.lower)
    found["market_cap"] = _field(tables.get(CHARACTERISTICS), name, "MARKET_CAP", dates)
    found["traded_value"] = _field(tables.get(TRADED_VALUES), name, "TRADED_VALUE", dates)

    if context.source is not None:
        found.insert(0, "name", [_reference(context.source, name, day) for day in dates])
        found["free_float"] = [context.source.fetch_free_float_factor(name, str(day.date()))
                               for day in dates]
        found["free_float"] = found["free_float"].astype(float)

    return found.rename_axis("DATE")


def corporate_actions(name: str,
                      tables: dict[str, pd.DataFrame],
                      days: pd.DatetimeIndex) -> pd.DataFrame:
    """Each corporate action on *name* with its ex-date over *days*, as the
    run applied it: its ``type`` and its ``amount`` (a dividend per share in
    the name's currency, a split's ratio).

    Raises:
        Unavailable: If the run was kept before its actions were.
    """
    actions = tables.get(ACTIONS)

    if actions is None:
        raise Unavailable(RUN_PREDATES)

    dates = pd.to_datetime(actions["EX_DATE"])
    mine = actions.loc[(actions[IDENTIFIER].astype(str) == name) & (dates >= days[0])
                       & (dates <= days[-1])]
    found = pd.DataFrame({"type": mine["TYPE"].astype(object).to_numpy(),
                          "amount": mine["VALUE"].astype(float).to_numpy()},
                         index=pd.DatetimeIndex(pd.to_datetime(mine["EX_DATE"]), name="DATE"))

    return found.sort_index(kind="stable")


def _within(index: pd.Index,
            days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The dates of *index* over *days*, or `Unavailable` when none are."""
    dates = pd.DatetimeIndex(sorted(pd.unique(index)))
    dates = dates[(dates >= days[0]) & (dates <= days[-1])]

    if not len(dates):
        raise Unavailable(TOO_SHORT)

    return dates


def _rows_of(table: pd.DataFrame,
             name: str) -> pd.DataFrame:
    """*name*'s rows of a long table kept by date, one a date."""
    mine = table.loc[table[IDENTIFIER].astype(str) == name].drop(columns=IDENTIFIER)

    return mine[~mine.index.duplicated(keep="last")]


def _field(table: pd.DataFrame | None,
           name: str,
           field: str,
           dates: pd.DatetimeIndex) -> pd.Series:
    """*name*'s *field* of a long table kept by date, on *dates*."""
    if table is None or table.empty or field not in table.columns:
        return pd.Series(np.nan, index=dates)

    return _rows_of(table, name)[field].astype(float).reindex(dates)


def _reference(source: Any,
               name: str,
               day: pd.Timestamp) -> str | None:
    """*name*'s name in the data store on *day*."""
    found = source.fetch_reference_data(name, str(day.date()), columns=["NAME"])

    if found.empty or name not in found.index or pd.isna(found.at[name, "NAME"]):
        return None

    return str(found.at[name, "NAME"])
