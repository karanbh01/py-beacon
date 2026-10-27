# src/beacon/analysis/performance/drawdowns.py
"""
A run's drawdowns: each fall from a peak, how deep it went, how long it took
to reach its trough and to recover, and the figures over all of them.

A drawdown starts on a peak (the level's high so far), reaches its trough
(the lowest level before it recovers) and ends when the level first gets back
to the peak. One still open on the last day has no recovery date: it is the
current drawdown. Lengths are counted in trading days, from the peak.
"""
# BN-374.
from dataclasses import dataclass

import numpy as np
import pandas as pd

# How many of the deepest drawdowns are listed unless asked otherwise.
WORST = 5


@dataclass(frozen=True)
class Drawdown:
    """One fall from a peak.

    Attributes:
        peak: The day of the high it fell from.
        trough: The day of its lowest level.
        recovery: The first day the level got back to the peak; None when it
            had not by the end.
        depth: The fall to the trough, as a fraction of the peak (negative).
        to_trough: Trading days from the peak to the trough.
        to_recover: Trading days from the trough to the recovery; None while
            open.
        length: Trading days from the peak to the recovery, or to the last
            day while open.
    """
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    depth: float
    to_trough: int
    to_recover: int | None
    length: int


@dataclass(frozen=True)
class DrawdownFigures:
    """The figures over a run's drawdowns.

    Attributes:
        worst: The deepest drawdowns, deepest first.
        current: How far below its peak the level ended (0 at a high).
        current_drawdown: The drawdown open on the last day; None at a high.
        average_depth: The mean depth of every drawdown; 0 without any.
        average_length: Their mean length in trading days; 0 without any.
        time_below_peak: The share of days spent below a peak.
        longest_below_peak: The longest drawdown, in trading days.
        recovery_factor: The total return over the size of the deepest
            drawdown; None without a drawdown.
        ulcer_index: The root mean square of the daily drawdown.
        pain_index: The mean size of the daily drawdown.
    """
    worst: list[Drawdown]
    current: float
    current_drawdown: Drawdown | None
    average_depth: float
    average_length: float
    time_below_peak: float
    longest_below_peak: int
    recovery_factor: float | None
    ulcer_index: float
    pain_index: float


def episodes(levels: pd.Series) -> list[Drawdown]:
    """Every drawdown of a dated level series, in order."""
    values = levels.to_numpy(dtype=float)

    if len(values) < 2:
        return []

    days = pd.DatetimeIndex(levels.index)
    peaks = np.maximum.accumulate(values)
    below = values < peaks
    found = []
    position = 1

    while position < len(values):
        if not below[position]:
            position += 1
            continue

        start = position - 1
        end = position

        while end < len(values) and below[end]:
            end += 1

        trough = start + 1 + int(np.argmin(values[start + 1:end]))
        recovered = end < len(values)
        found.append(Drawdown(peak=days[start], trough=days[trough],
                              recovery=days[end] if recovered else None,
                              depth=float(values[trough] / peaks[start] - 1.0),
                              to_trough=trough - start,
                              to_recover=end - trough if recovered else None,
                              length=(end if recovered else len(values) - 1) - start))
        position = end + 1

    return found


def drawdown_figures(levels: pd.Series,
                     total_return: float,
                     worst: int = WORST) -> DrawdownFigures:
    """The figures over a level series' drawdowns.

    Args:
        levels: The level, dated, from its start.
        total_return: The return over the whole series, for the recovery
            factor.
        worst: How many of the deepest drawdowns to list.
    """
    found = episodes(levels)
    values = levels.to_numpy(dtype=float)
    daily = values / np.maximum.accumulate(values) - 1.0 if len(values) else np.array([])
    deepest = sorted(found, key=lambda drawdown: drawdown.depth)
    current = found[-1] if found and found[-1].recovery is None else None

    return DrawdownFigures(
        worst=deepest[:worst],
        current=float(daily[-1]) if len(daily) else 0.0,
        current_drawdown=current,
        average_depth=float(np.mean([drawdown.depth for drawdown in found])) if found else 0.0,
        average_length=(float(np.mean([drawdown.length for drawdown in found]))
                        if found else 0.0),
        time_below_peak=float((daily < 0).mean()) if len(daily) else 0.0,
        longest_below_peak=max((drawdown.length for drawdown in found), default=0),
        recovery_factor=(total_return / abs(deepest[0].depth) if deepest else None),
        ulcer_index=float(np.sqrt(np.mean(daily ** 2))) if len(daily) else 0.0,
        pain_index=float(np.mean(np.abs(daily))) if len(daily) else 0.0)
