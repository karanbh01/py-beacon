# src/beacon/plot/legend.py
"""
A framed chart's legend, in the heading band.

The legend sits at the right of the heading band, right-aligned with the end
of the x axis and its bottom level with the subtitle's, in a row of as many
entries as fit beside the title and subtitle; it wraps onto more rows, upward,
when they do not. A legend taller than
the heading pushes the plot down rather than running into it, and one that
cannot fit beside the heading at all goes under it.

Widths are measured from the fonts themselves, as text paths, so the layout
needs no renderer and comes out the same on every backend.
"""
from dataclasses import dataclass
from typing import Any, Literal

from .._optional import require

require("matplotlib", "Charting")

import matplotlib  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

# Spacing inside the legend, in multiples of its font size.
HANDLE_LENGTH = 1.6
HANDLE_GAP = 0.5
COLUMN_GAP = 1.2
ROW_HEIGHT = 1.35

# The least room, in points, kept between the heading's text and the legend.
CLEARANCE = 12.0


@dataclass(frozen=True)
class LegendPlan:
    """Where a framed chart's legend goes and how it is laid out."""

    handles: list[Any]
    labels: list[str]
    columns: int
    height: float  # inches
    beside: bool   # beside the heading, or under it


def text_width(text: str,
               size: float,
               family: str,
               style: Literal["normal", "italic", "oblique"] = "normal",
               weight: Any = "normal") -> float:
    """How wide *text* is set, in inches."""
    if not text:
        return 0.0

    font = FontProperties(family=family, style=style, weight=weight)

    return float(TextPath((0, 0), text, size=size, prop=font).get_extents().width) / 72.0


def plan(ax: Axes,
         heading_width: float,
         room: float) -> LegendPlan | None:
    """Lay out an axes' legend for the heading band.

    Args:
        ax: The chart's axes.
        heading_width: How far the heading's text reaches from the plot's
            left edge, in inches.
        room: The plot's width, in inches.

    Returns:
        LegendPlan | None: None when the axes has no legend.
    """
    legend = ax.get_legend()

    if legend is None:
        return None

    handles = list(getattr(legend, "legend_handles", []))
    labels = [text.get_text() for text in legend.get_texts()]

    if not labels:
        return None

    size = _font_size()
    family = matplotlib.rcParams["font.sans-serif"][0]
    weight = matplotlib.rcParams["font.weight"]
    entry = (max(text_width(label, size, family, weight=weight) for label in labels)
             + (HANDLE_LENGTH + HANDLE_GAP) * size / 72.0)
    gap = COLUMN_GAP * size / 72.0

    beside = room - heading_width - CLEARANCE / 72.0
    columns = _columns(len(labels), entry, gap, beside)
    fits_beside = columns > 0

    if not fits_beside:
        columns = max(1, _columns(len(labels), entry, gap, room))

    rows = -(-len(labels) // columns)

    return LegendPlan(handles=handles,
                      labels=labels,
                      columns=columns,
                      height=rows * ROW_HEIGHT * size / 72.0,
                      beside=fits_beside)


def place(ax: Axes,
          layout: LegendPlan,
          right: float,
          bottom: float) -> None:
    """Redraw the axes' legend at the bottom right of the heading band.

    Args:
        ax: The chart's axes.
        layout: From plan().
        right: The plot's right edge, as a figure fraction.
        bottom: Where the legend's bottom sits, as a figure fraction.
    """
    figure = ax.get_figure()
    assert figure is not None

    ax.legend(layout.handles, layout.labels, ncol=layout.columns,
              loc="lower right", bbox_to_anchor=(right, bottom),
              bbox_transform=figure.transFigure, frameon=False,
              borderaxespad=0.0, borderpad=0.0,
              handlelength=HANDLE_LENGTH, handletextpad=HANDLE_GAP,
              columnspacing=COLUMN_GAP, labelspacing=ROW_HEIGHT - 1.0)


def _columns(count: int,
             entry: float,
             gap: float,
             width: float) -> int:
    """The most columns of *entry*-wide entries that fit in *width*, or 0."""
    for columns in range(count, 0, -1):
        if columns * entry + (columns - 1) * gap <= width:
            return columns

    return 0


def _font_size() -> float:
    """The legend's font size, in points."""
    from matplotlib.font_manager import font_scalings

    size = matplotlib.rcParams["legend.fontsize"]

    if isinstance(size, str):
        return float(matplotlib.rcParams["font.size"]) * font_scalings.get(size, 1.0)

    return float(size)
