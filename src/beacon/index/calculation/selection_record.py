# src/beacon/index/calculation/selection_record.py
"""
An index's selection record as plain data and one table, so it can be kept
beside a calculated result and read back the same.

    plain, measures = encoded(result.selections)
    decoded(plain, measures) == result.selections

``plain`` holds each rebalance's universe, rungs and exclusions, dates as ISO
strings; ``measures`` has a row for each threshold each rule tested each name
against, in the order tested (`beacon.index.margins`).
"""
# BN-474: the index cache kept no selections, so a run served from it had
# neither its funnel nor its margins.
from dataclasses import asdict
from typing import Any

import pandas as pd

from ..margins import Measure
from .selection import SelectionStep, SelectionSummary

COLUMNS = ("DATE", "RULE", "IDENTIFIER", "QUANTITY", "VALUE", "COMPARISON", "THRESHOLD",
           "PASSED")


def encoded(selections: dict[pd.Timestamp, SelectionSummary]) -> tuple[dict[str, Any],
                                                                        pd.DataFrame]:
    """*selections* as plain data, and their measures as one long table."""
    plain = {date.isoformat(): {"universe": list(summary.universe),
                                "steps": [asdict(step) for step in summary.steps],
                                "exclusions": dict(summary.exclusions)}
             for date, summary in selections.items()}
    columns: dict[str, list[Any]] = {column: [] for column in COLUMNS}

    for date, summary in selections.items():
        for position, measured in summary.measures.items():
            for name, measures in measured.items():
                for measure in measures:
                    for column, value in zip(columns, (
                            date.isoformat(), position, name, measure.quantity, measure.value,
                            measure.comparison, measure.threshold, measure.passed), strict=True):
                        columns[column].append(value)

    table = pd.DataFrame(columns, columns=list(COLUMNS)).astype(
        {"DATE": str, "RULE": "int64", "IDENTIFIER": str, "QUANTITY": str, "VALUE": float,
         "COMPARISON": str, "THRESHOLD": float, "PASSED": bool})

    return plain, table


def decoded(plain: dict[str, Any],
            measures: pd.DataFrame) -> dict[pd.Timestamp, SelectionSummary]:
    """The selections `encoded` gave *plain* and *measures* for."""
    found: dict[str, dict[int, dict[str, list[Measure]]]] = {}

    for date, position, name, quantity, value, comparison, threshold, passed in (
            measures[list(COLUMNS)].itertuples(index=False, name=None)):
        found.setdefault(str(date), {}).setdefault(int(position), {}).setdefault(
            str(name), []).append(Measure(str(quantity), _number(value), str(comparison),
                                          _number(threshold), bool(passed)))

    return {pd.Timestamp(date): SelectionSummary(
        universe=[str(name) for name in kept["universe"]],
        steps=[SelectionStep(**step) for step in kept["steps"]],
        exclusions={str(name): int(position) for name, position in kept["exclusions"].items()},
        measures={position: {name: tuple(listed) for name, listed in measured.items()}
                  for position, measured in found.get(date, {}).items()})
        for date, kept in plain.items()}


def _number(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


__all__ = ["decoded", "encoded"]
