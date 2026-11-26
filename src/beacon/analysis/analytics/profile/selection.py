# src/beacon/analysis/analytics/profile/selection.py
"""A name's profile's selection: how far it was from failing each
eligibility rule at each rebalance."""
# BN-474 (Karan, 2026-11-26): the margin as a share of the threshold, and a
# near miss within the selection options' share of it.
import pandas as pd

from ....index.margins import margin
from ...panels import SELECTION_MARGINS
from ...reports.statistics import RUN_PREDATES, Unavailable

COLUMNS = {"RULE": "rule", "RULE_NAME": "rule_name", "QUANTITY": "measure",
           "VALUE": "measured", "COMPARISON": "comparison", "THRESHOLD": "threshold",
           "PASSED": "passed"}


def selection(name: str,
              tables: dict[str, pd.DataFrame],
              days: pd.DatetimeIndex) -> pd.DataFrame:
    """At each rebalance over *days*: each threshold each eligibility rule
    tested *name* against, a row each: what it measured (``measure``),
    the name's value (``measured``), the threshold, its margin
    (`beacon.index.margins`) and whether it passed; none for a rule
    it did not reach, having been removed before it.

    Raises:
        Unavailable: If the run was kept before its margins were.
    """
    kept = tables.get(SELECTION_MARGINS)

    if kept is None:
        raise Unavailable(RUN_PREDATES)

    dates = pd.DatetimeIndex(kept.index)
    mine = kept.loc[(kept["IDENTIFIER"].astype(str) == name) & (dates >= days[0])
                    & (dates <= days[-1])]
    found = mine[list(COLUMNS)].rename(columns=COLUMNS)
    found = found.astype({"rule": "int64", "rule_name": object, "measure": object,
                          "comparison": object, "measured": float, "threshold": float,
                          "passed": bool})
    margins = [margin(value, comparison, threshold) for value, comparison, threshold
               in zip(_numbers(found["measured"]), found["comparison"],
                      _numbers(found["threshold"]), strict=True)]
    found.insert(len(found.columns) - 1, "margin", pd.Series(margins, index=found.index,
                                                            dtype=float))

    return found.sort_index(kind="stable").rename_axis("DATE")


def near_misses(table: pd.DataFrame,
                within: float) -> pd.DataFrame:
    """The rows of *table* (as `selection` gives it) a name passed with a
    margin under *within*."""
    close = table["passed"] & table["margin"].notna() & (table["margin"] >= 0.0)

    return table.loc[close & (table["margin"] < within)]


def _numbers(values: pd.Series) -> list[float | None]:
    return [None if pd.isna(value) else float(value) for value in values]
