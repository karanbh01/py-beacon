# src/beacon/plot/drawing.py
"""
What every chart method draws with: the axes it draws on, the line a level
chart is read against, the last value marked, sign colours and truncation.

Shared by `accessors` and `analysis_charts`, which hold the chart methods.
"""
import logging
from typing import Any

import pandas as pd

from .._optional import require
from . import style as beacon_style
from .frame import REFERENCE_ID, finish, ink, new_figure

require("matplotlib", "Charting")

from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

# Bars thinner than this are invisible; a near-zero contribution still needs to
# show that it exists and which way it points.
MINIMUM_BAR = 1e-4

# How many names a weights or contributions chart shows before it stops being
# readable. Beyond this the labels collide and the chart says less than a table.
MAX_BARS = 25

# How many constituents a constituents chart draws. More lines than this are a
# tangle in which no one line can be followed.
MAX_LINES = 8

# Short names, as the methods below used them before the frame moved out.
_ink = ink
_finish = finish


def _axes(ax: Axes | None,
          kind: str) -> Axes:
    """The axes to draw on, creating a framed figure only when none is given."""
    beacon_style.register()

    if ax is not None:
        return ax

    figure = new_figure(beacon_style.FIGSIZE.get(kind, (8.0, 5.0)))

    return figure.add_subplot()


def _given(value: str | None,
           default: str) -> str:
    """A caller's title, subtitle or notes, or the chart's own when None.
    An empty string is kept: it is how a caller turns one off."""
    return default if value is None else value


def _reference_at(ax: Axes,
                  level: float) -> None:
    """A level chart's one horizontal line, at its starting value, in place of
    gridlines: above it the series has gained, below it lost."""
    ax.grid(visible=False)
    ax.axhline(level, color=_ink(ax, "border"),
               linewidth=beacon_style.REFERENCE_WIDTH, zorder=1, gid=REFERENCE_ID)


def _series(payload: pd.Series) -> pd.Series:
    """A result's series as floats."""
    return payload.astype(float)


def _rebase(series: pd.Series,
            base: float = 100.0) -> pd.Series:
    """A level series rescaled to start at *base*."""
    first = float(series.iloc[0])

    return series / first * base if first else series


def _mark_last(ax: Axes,
               series: pd.Series,
               colour_name: str = "accent") -> None:
    """Dot and label the final value.

    The number a reader actually wants from a level chart is where it ended,
    and making them read it off an axis is a small tax on every glance.
    """
    if series.empty:
        return

    ink = _ink(ax, colour_name)
    ax.plot([series.index[-1]], [series.iloc[-1]], marker="o", markersize=4.5,
            color=ink, zorder=5)
    ax.annotate(f"{series.iloc[-1]:,.1f}",
                xy=(series.index[-1], series.iloc[-1]),
                xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=8, color=ink, fontweight="bold")


def _signed_colours(ax: Axes,
                    values: Any) -> list[str]:
    """Green for positive, red for negative — the application's own signs."""
    up, down = _ink(ax, "success"), _ink(ax, "danger")

    return [up if float(value) >= 0 else down for value in values]


def _truncate(labels: list[str],
              values: list[float],
              limit: int = MAX_BARS) -> tuple[list[str], list[float], int]:
    """Keep the largest *limit* entries by magnitude, reporting what was cut.

    Silently dropping the tail would make a chart of thirty names look like a
    chart of twenty-five. The count comes back so the caller can say so on the
    page.
    """
    if len(labels) <= limit:
        return labels, values, 0

    order = sorted(range(len(values)), key=lambda i: abs(values[i]), reverse=True)
    kept = sorted(order[:limit], key=lambda i: values[i], reverse=True)

    return ([labels[i] for i in kept], [values[i] for i in kept],
            len(labels) - limit)
