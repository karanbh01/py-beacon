# src/beacon/analysis/attribution/groups.py
"""
Which group each name was in on each day, and a table's columns added up by
it, for attribution by group.

A name's weight at a close earns the next day's return, so a day's
contributions belong to the groups the names were in at the close before: the
groups of the last rebalance before the day. Memberships change only at
rebalances, so the days sharing one are grouped at once.
"""
# BN-393, and BN-394's Brinson-Fachler effects.
import pandas as pd

from ...data.base import as_of_position
from ..data import RunData
from ..panels import REFERENCE

UNKNOWN = "Unknown"


def memberships(run: RunData,
                field: str) -> dict[pd.Timestamp, pd.Series]:
    """Each rebalance's group of every name by *field*, by date.

    Raises:
        KeyError: If the run kept no such field.
    """
    reference = run.tables[REFERENCE]

    if field not in reference.columns:
        raise KeyError(f"The run kept no {field} for its names.")

    return {pd.Timestamp(day): rows.set_index("IDENTIFIER")[field].astype(str)
            for day, rows in reference.groupby(level=0)}


def by_group(table: pd.DataFrame,
             groups: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """*table*'s columns (names) added up by group, each day's by the groups
    of the last rebalance before it, or of the first for a day at or before
    it; ``Unknown`` for names with none."""
    rebalances = pd.DatetimeIndex(sorted(groups))
    keys = []

    for day in table.index:
        at = as_of_position(rebalances, day - pd.Timedelta(days=1))
        keys.append(rebalances[at] if at is not None
                    else rebalances[0] if len(rebalances) else None)

    pieces = []

    for key in dict.fromkeys(keys):
        rows = table[[found == key for found in keys]]
        membership = groups[key] if key is not None else pd.Series(dtype=str)
        labels = membership.reindex(rows.columns).fillna(UNKNOWN)
        pieces.append(rows.T.groupby(labels.to_numpy()).sum().T)

    if not pieces:
        return pd.DataFrame(index=table.index)

    return pd.concat(pieces).sort_index().fillna(0.0)
