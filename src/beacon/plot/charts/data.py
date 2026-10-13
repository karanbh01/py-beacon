# src/beacon/plot/charts/data.py
"""
What the chart classes take in: series and frames, their dates, and the
limits a caller sets.

Dates become days since 1970-01-01 (beacon.plot.dateticks), the numbers every
renderer draws dates as.
"""
# BN-299.
import math
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from .. import dateticks, ticks

# What a chart takes as data: one series, or a frame of them.
Data: TypeAlias = pd.Series | pd.DataFrame

# A limit on one axis: low and high, either left to the data with None.
Limits = tuple[Any, Any] | None


def frame_of(data: Data,
             name: str = "Series") -> pd.DataFrame:
    """A series as a one-column frame, or a frame as it is.

    Raises:
        TypeError: If *data* is neither.
        ValueError: If it is empty.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name if data.name is not None else name)

    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"a chart takes a pandas Series or DataFrame, not "
                        f"{type(data).__name__}")

    if data.empty:
        raise ValueError("there is nothing to draw: the data is empty")

    return data


def is_dates(index: pd.Index) -> bool:
    """Whether an index holds dates."""
    return isinstance(index, pd.DatetimeIndex)


def positions(index: pd.Index) -> tuple[float, ...]:
    """An index as the numbers it is drawn at: days since 1970-01-01 for
    dates, the values themselves for numbers."""
    if is_dates(index):
        return tuple(number(when) for when in index)

    return tuple(float(value) for value in index)


def number(when: Any) -> float:
    """A date (or anything pandas reads as one) as days since 1970-01-01."""
    stamp = pd.Timestamp(when)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(None)

    return dateticks.to_number(stamp.to_pydatetime())


def values(series: pd.Series) -> tuple[float, ...]:
    """A series' values as floats, missing values as NaN."""
    return tuple(float(value) for value in series.to_numpy(dtype=float, na_value=np.nan))


def extent(*groups: Iterable[float]) -> tuple[float, float]:
    """The lowest and highest of some values, ignoring missing ones.

    Raises:
        ValueError: If there is no value at all.
    """
    found = [value for group in groups for value in group if math.isfinite(value)]

    if not found:
        raise ValueError("there is nothing to draw: every value is missing")

    return min(found), max(found)


def limited(low: float,
            high: float,
            given: Limits,
            dates: bool = False) -> tuple[float, float]:
    """An axis's range from its data and the limits a caller set.

    Limits given the wrong way round are swapped. A single limit leaves the
    other end to the data; where the limit is beyond the data, the automatic
    end keeps the data's span from it.

    Args:
        low: The data's low end.
        high: Its high end.
        given: The caller's (low, high), either None for automatic.
        dates: Whether the limits are dates.
    """
    if given is None:
        return low, high

    start, stop = (_limit(value, dates) for value in given)
    span = high - low

    if start is not None and stop is not None:
        return min(start, stop), max(start, stop)
    if start is not None:
        return start, high if high > start else start + span
    if stop is not None:
        return low if low < stop else stop - span, stop

    return low, high


def _limit(value: Any,
           dates: bool) -> float | None:
    if value is None:
        return None

    return number(value) if dates else float(value)


def percent(value: float,
            decimals: int = 1) -> str:
    """A share as a percentage, with a true minus sign."""
    return ticks.fix_minus(f"{value * 100:.{decimals}f}%")


def plain(value: float,
          decimals: int = 1) -> str:
    """A number with thousands separated, with a true minus sign."""
    return ticks.fix_minus(f"{value:,.{decimals}f}")


def named(columns: Sequence[Any]) -> list[str]:
    """A frame's column names as text."""
    return [str(column) for column in columns]
