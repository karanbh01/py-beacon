# src/beacon/server/runs.py
"""
Reading a stored run's composition.

Three small readers, in their own module because both `views` and `weights`
need them and neither should have to import the other to get at them. Every
view of a completed backtest starts by turning the stored payload back into
snapshots, so this is the shared floor rather than a utility drawer.
"""
import logging
from typing import Any

import pandas as pd

from ..exceptions import DataNotFoundError
from .schemas import RebalanceSnapshot

logger = logging.getLogger(__name__)


def snapshots_from(run: dict[str, Any]) -> list[RebalanceSnapshot]:
    """Read the rebalance snapshots off a stored run.

    Raises:
        DataNotFoundError: If the run carries none. A result stored before
            BN-71 extended the payload has a level and metrics but no
            composition, and saying so is better than serving an empty index.
    """
    raw = run.get("rebalances")
    if not raw or not isinstance(raw, list):
        raise DataNotFoundError(
            "rebalance snapshots on this run",
            source="the run predates composition being stored; re-run the backtest")

    return [RebalanceSnapshot.model_validate(entry) for entry in raw]


def weight_map(snapshots: list[RebalanceSnapshot],
               uncapped: bool = False) -> dict[pd.Timestamp, dict[str, float]]:
    """Snapshots keyed by timestamp, as the analysis helpers expect."""
    return {pd.Timestamp(snapshot.date):
            (snapshot.uncapped_weights if uncapped else snapshot.weights)
            for snapshot in snapshots}


def snapshot_at(snapshots: list[RebalanceSnapshot],
                as_of: str | None) -> RebalanceSnapshot:
    """The rebalance in force on a date.

    The latest snapshot at or before *as_of*, because an index holds the
    weights set at its last rebalance until the next one. A date before the
    first rebalance has no answer and says so rather than returning the first,
    which would report weights that were not yet in force.
    """
    if as_of is None:
        return snapshots[-1]

    date = pd.Timestamp(as_of)
    eligible = [snapshot for snapshot in snapshots
                if pd.Timestamp(snapshot.date) <= date]

    if not eligible:
        raise DataNotFoundError(
            f"a rebalance on or before {date.date()}",
            source=f"the first rebalance is {snapshots[0].date}")

    return eligible[-1]
