# src/beacon/analysis/holding_replay.py
"""
The fund's holdings worked out again from its trades.

A holding changes only when the fund trades the name or a split changes its
share count, and the books add each trade to the holding before it, in
order. So the fund's holdings on the days they changed (as
`beacon.analysis.holdings.holding_changes` gives them) are worked out again
by replaying the trades and the splits in the books' own arithmetic: a split
multiplies the holding first, then each buy adds to it and each sell takes
from it, a holding below 1e-9 ending.

What the replay does not give back to the last bit is kept: a patch of the
recorded rows it gets wrong or misses, and the rows it gives that were not
recorded (a split on a day nothing was recorded, say). A stored run keeps
those and its trades rather than the holdings, and checks on writing that
they come back exactly.
"""
# BN-413 (agreed with Karan on 2026-10-29). At 6,000 names over seven years
# every one of the 127,647 recorded changes replays exactly, with 1,360 rows
# to drop.
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .holdings import AMOUNT, HELD, NAME

# Below this a sold holding ends, as the books end it.
ENDED = 1e-9
SPLIT = "SPLIT"
CANCELLED = "cancelled"

# The tables a run keeps in place of the fund's holdings.
PATCH, DROPS = "patch", "drops"


def replayed(trades: pd.DataFrame,
             actions: pd.DataFrame) -> pd.DataFrame:
    """The fund's holdings on the days they changed, from its *trades* (a
    run's trades table) and the *actions*' splits, by day and then name."""
    splits = actions[actions["TYPE"] == SPLIT] if "TYPE" in actions.columns else actions.iloc[0:0]

    if "STATUS" in splits.columns:
        splits = splits[splits["STATUS"] != CANCELLED]

    split_days: dict[pd.Timestamp, list[tuple[str, float]]] = {}

    for name, day, ratio in zip(splits["IDENTIFIER"], pd.DatetimeIndex(splits["EX_DATE"]),
                                splits["VALUE"], strict=True):
        split_days.setdefault(day, []).append((str(name), float(ratio)))

    trade_days: dict[pd.Timestamp, list[tuple[str, bool, float]]] = {}

    for day, name, side, quantity in zip(pd.DatetimeIndex(trades.index), trades["IDENTIFIER"],
                                         trades["SIDE"], trades["QUANTITY"], strict=True):
        trade_days.setdefault(day, []).append((str(name), side == "BUY", float(quantity)))

    held: dict[str, float] = {}
    rows: list[tuple[pd.Timestamp, str, float]] = []

    for day in sorted(set(split_days) | set(trade_days)):
        touched = _split(held, split_days.get(day, [])) | _trade(held, trade_days.get(day, []))
        rows.extend((day, name, held.get(name, np.nan)) for name in sorted(touched))

    return _changes(rows)


def _split(held: dict[str, float],
           splits: list[tuple[str, float]]) -> set[str]:
    """Apply a day's splits to the names held; the names they changed."""
    touched = set()

    for name, ratio in splits:
        if name in held:
            held[name] *= ratio
            touched.add(name)

    return touched


def _trade(held: dict[str, float],
           trades: list[tuple[str, bool, float]]) -> set[str]:
    """Apply a day's trades in order; the names they changed."""
    touched = set()

    for name, bought, quantity in trades:
        touched.add(name)

        if bought:
            held[name] = held.get(name, 0.0) + quantity
            continue

        held[name] = held.get(name, 0.0) - quantity

        if held[name] < ENDED:
            del held[name]

    return touched


def _changes(rows: list[tuple[pd.Timestamp, str, float]]) -> pd.DataFrame:
    """Rows of a day, a name and its holding at the day's end as a changes
    table, keeping a row only where the holding differs from the day before
    it was last touched."""
    last: dict[str, float] = {}
    kept = []

    for day, name, amount in rows:
        before = last.get(name, np.nan)

        if (np.isnan(amount) and np.isnan(before)) or amount == before:
            continue

        kept.append((day, name, amount))
        last[name] = amount

    days = pd.DatetimeIndex([day for day, _, _ in kept], name="DATE")
    amounts = np.array([amount for _, _, amount in kept], dtype=float)

    return pd.DataFrame({NAME: [name for _, name, _ in kept], AMOUNT: amounts,
                         HELD: ~np.isnan(amounts)}, index=days)


def pack_holdings(changes: pd.DataFrame,
                  trades: pd.DataFrame,
                  actions: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """What keeps *changes* given the trades and actions: the recorded rows
    the replay gets wrong or misses (``patch``) and the rows it gives that
    were not recorded (``drops``); None when they would not give *changes*
    back exactly."""
    replay = replayed(trades, actions)
    recorded, again = _keyed(changes), _keyed(replay)
    joined = recorded.join(again, how="outer", lsuffix="_kept", rsuffix="_again")
    both = joined[AMOUNT + "_kept"].notna() | joined[HELD + "_kept"].notna()
    same = ((joined[AMOUNT + "_kept"] == joined[AMOUNT + "_again"])
            | (joined[AMOUNT + "_kept"].isna() & joined[AMOUNT + "_again"].isna()
               & joined[HELD + "_again"].notna()))
    patch = changes[_flags(changes, joined.index[both & ~same])]
    dropped = joined.index[~both]
    drops = pd.DataFrame({NAME: dropped.get_level_values(1)},
                         index=pd.DatetimeIndex(dropped.get_level_values(0), name="DATE"))
    kept = {PATCH: patch, DROPS: drops}

    if not _same(unpack_holdings(trades, actions, kept, changes.dtypes.to_dict()), changes):
        return None

    return kept


def unpack_holdings(trades: pd.DataFrame,
                    actions: pd.DataFrame,
                    kept: Mapping[str, pd.DataFrame],
                    dtypes: Mapping[str, object] | None = None) -> pd.DataFrame:
    """The fund's holdings on the days they changed, from the replay with
    *kept*'s rows dropped and patched, by day and then name."""
    replay = replayed(trades, actions)
    drops = _keyed(kept[DROPS].assign(**{AMOUNT: 0.0, HELD: False}))
    patch = kept[PATCH]
    gone = set(drops.index) | set(_keyed(patch).index)
    keys = list(zip(pd.DatetimeIndex(replay.index), replay[NAME], strict=True))
    left = replay[[key not in gone for key in keys]]
    frame = pd.concat([left, patch]) if len(patch) else left
    order = np.lexsort((pd.factorize(frame[NAME].astype(str), sort=True)[0],
                        pd.DatetimeIndex(frame.index).asi8))
    frame = frame.iloc[order]
    frame.index = pd.DatetimeIndex(frame.index, name="DATE")

    return frame.astype(dict(dtypes)) if dtypes else frame


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    """*frame*'s amounts and held flags by day and name."""
    index = pd.MultiIndex.from_arrays([pd.DatetimeIndex(frame.index), frame[NAME].astype(str)])

    return pd.DataFrame({AMOUNT: frame[AMOUNT].to_numpy(dtype=float),
                         HELD: frame[HELD].to_numpy()}, index=index)


def _flags(frame: pd.DataFrame,
           keys: pd.Index) -> np.ndarray:
    """Which of *frame*'s rows are at *keys* (day and name)."""
    wanted = set(keys)

    return np.array([key in wanted for key in zip(pd.DatetimeIndex(frame.index),
                                                  frame[NAME].astype(str), strict=True)],
                    dtype=bool)


def _same(found: pd.DataFrame,
          expected: pd.DataFrame) -> bool:
    return (found.equals(expected) and found.index.equals(expected.index)
            and found.index.name == expected.index.name
            and list(found.dtypes.astype(str)) == list(expected.dtypes.astype(str)))
