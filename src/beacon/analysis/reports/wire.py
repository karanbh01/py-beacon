# src/beacon/analysis/reports/wire.py
"""
A statistic's value as plain data, ready to send as JSON: figures as a map of
names to numbers, a series as ``{index, name, data}``, a table as
``{index, columns, data}`` (rows), with dates as ``YYYY-MM-DD`` and missing
values as None.
"""
# BN-398. The same shapes as the server's SeriesPayload and TableFrame, made
# here so the library needs nothing of the server.
from typing import Any

import pandas as pd

from .catalogue.build import frame
from .statistics import FIGURES, SERIES, TABLE, Summarised, figures_of, scalar


def series_payload(series: pd.Series) -> dict[str, Any]:
    """A series as ``{index, name, data}``."""
    return {"index": [scalar(label) for label in series.index],
            "name": None if series.name is None else str(series.name),
            "data": [scalar(value) for value in series.to_numpy()]}


def table_payload(table: pd.DataFrame) -> dict[str, Any]:
    """A table as ``{index, columns, data}``, a list of values a row."""
    flat = frame(table)

    return {"index": [scalar(label) for label in flat.index],
            "columns": [str(column) for column in flat.columns],
            "data": [[scalar(value) for value in row] for row in flat.itertuples(index=False)]}


def payload(kind: str,
            value: Any) -> dict[str, Any]:
    """A statistic's value of *kind*: ``{"point": ...}``, with ``summary``
    for a table that has one."""
    if kind == FIGURES:
        return {"point": figures_of(value)}

    if kind == SERIES:
        series = value if isinstance(value, pd.Series) else pd.Series(value, dtype=float)
        return {"point": series_payload(series)}

    if kind == TABLE:
        if isinstance(value, Summarised):
            return {"point": table_payload(value.table), "summary": figures_of(value.summary)}

        return {"point": table_payload(value)}

    raise ValueError(f"Unknown kind of statistic {kind!r}.")
