# src/beacon/server/serialisation.py
"""
Wire formats shared by every router.

pandas objects do not survive a plain JSON encoder intact — timestamps, NaN
and numpy scalars all need handling — so frames and series are converted here
into an explicit, stable shape rather than left to a default encoder.
"""
from typing import Any

import numpy as np
import pandas as pd


def _json_safe(value: Any) -> Any:
    """Convert a single pandas/numpy value into something JSON can carry."""
    if value is None or value is pd.NaT:
        return None

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    # Covers both float('nan') and numpy floats already unwrapped above.
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None

    return value


def dataframe_to_payload(frame: pd.DataFrame) -> dict[str, Any]:
    """Serialise a DataFrame as ``{index, columns, data}``.

    Row-oriented ``data`` keeps the payload compact and preserves column
    order, which a dict-of-columns would not guarantee.

    Args:
        frame: The frame to serialise. An empty frame yields empty lists.

    Returns:
        dict: ``index`` (list of row labels), ``columns`` (list of column
        names), and ``data`` (list of rows, each a list of cell values).
        NaN, NaT and infinities all become None.
    """
    return {
        "index": [_json_safe(label) for label in frame.index],
        "columns": [str(column) for column in frame.columns],
        "data": [[_json_safe(cell) for cell in row] for row in frame.itertuples(index=False)],
    }


def series_to_payload(series: pd.Series) -> dict[str, Any]:
    """Serialise a Series as ``{index, name, data}``.

    Args:
        series: The series to serialise.

    Returns:
        dict: ``index`` (list of labels), ``name`` (the series name, or None),
        and ``data`` (list of values, with NaN/NaT/infinities as None).
    """
    return {
        "index": [_json_safe(label) for label in series.index],
        "name": str(series.name) if series.name is not None else None,
        "data": [_json_safe(value) for value in series.to_numpy()],
    }
