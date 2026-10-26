# src/beacon/server/serialisation.py
"""
Wire formats shared by every router.

pandas objects do not survive a plain JSON encoder intact (timestamps, NaN
and numpy scalars all need handling), so frames and series are converted here
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
        "data": _rows(frame),
    }


def _rows(frame: pd.DataFrame) -> list[list[Any]]:
    """A frame's cells, row by row, as `_json_safe` gives them."""
    kinds = set(frame.dtypes)

    # A frame of one float dtype, as a book's daily weights are, all at once
    # (BN-404): picking out thousands of columns one at a time was most of
    # serving a 6,000-name panel. The same Python floats, and None for NaN.
    if (frame.shape[0] and frame.shape[1] and len(kinds) == 1
            and isinstance(next(iter(kinds)), np.dtype) and next(iter(kinds)).kind == "f"):
        grid = frame.to_numpy()
        cells = grid.astype(object)
        cells[~np.isfinite(grid)] = None

        return list(cells.tolist())

    # Column by column, then into rows (BN-349): a numeric column through
    # numpy's `tolist`, which gives the Python numbers `_json_safe` gave cell
    # by cell, about twelve million calls on a 6,000-name run.
    columns = [_column_values(frame.iloc[:, position]) for position in range(frame.shape[1])]

    return [list(row) for row in zip(*columns, strict=True)]


def _column_values(column: pd.Series) -> list[Any]:
    """One column's cells as `_json_safe` gives them, numeric columns at once."""
    # Plain numpy dtypes only: a pandas extension dtype (a nullable integer,
    # say) becomes floats on the way to numpy, where its cells did not.
    if not isinstance(column.dtype, np.dtype) or column.dtype.kind not in "fiub":
        return [_json_safe(cell) for cell in column]

    values = column.to_numpy()

    cells: list[Any] = values.tolist()

    if values.dtype.kind == "f":
        for position in np.flatnonzero(~np.isfinite(values)):
            cells[position] = None

    return cells


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
