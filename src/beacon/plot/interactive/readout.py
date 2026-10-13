# src/beacon/plot/interactive/readout.py
"""
What an interactive chart's popup says: values written with their units, as
the axis they are read on writes its ticks, and dates as a reader says them.
"""
# BN-301.
import datetime as dt
import math

from ..layout import LabelStyle
from ..ticks import fix_minus

EPOCH = dt.date(1970, 1, 1)


def date_text(days: float) -> str:
    """A date, from days since 1970-01-01: 13 Oct 2026."""
    day = EPOCH + dt.timedelta(days=math.floor(days))

    return f"{day.day} {day:%b %Y}"


def value_text(value: float,
               labels: LabelStyle,
               whole: bool = False) -> str:
    """A value as its axis writes it, with its units: shares and percentages
    with a % sign, and an axis's own writing (such as £m) where it has one;
    *whole* for a series of counts, written without decimals."""
    if not math.isfinite(value):
        return "\N{EN DASH}"
    if callable(labels):
        return labels(value)
    if labels in ("share", "percent"):
        return fix_minus(f"{100.0 * value:,.{1 if labels == 'share' else 2}f}%")

    return fix_minus(f"{value:,.0f}") if whole else number_text(value)


def number_text(value: float) -> str:
    """A number with as many decimals as its size reads with."""
    size = abs(value)
    decimals = 0 if size >= 1000 else 1 if size >= 100 else 2

    return fix_minus(f"{value:,.{decimals}f}")
