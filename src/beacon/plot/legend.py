# src/beacon/plot/legend.py
"""
A framed chart's legend: in the heading band by default, or, as the style
says (legend.position), under the plot or in one column to its right.

The legend sits at the right of the heading band, right-aligned with the end
of the x axis and its bottom level with the subtitle's, in a row of as many
entries as fit beside the title and subtitle; it wraps onto more rows, upward,
when they do not. A legend taller than the heading pushes the plot down rather
than running into it, and one that cannot fit beside the heading at all goes
under it.

Widths are measured from the fonts themselves (beacon.plot.metrics), so the
layout needs no renderer and comes out the same on every backend.
"""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import metrics
from .layout import Box, LegendEntry, LegendItem, Text
from .metrics import Font

# Spacing inside the legend, in multiples of its font size.
HANDLE_LENGTH = 1.6
HANDLE_GAP = 0.5
COLUMN_GAP = 1.2
ROW_HEIGHT = 1.35
# A key's height, and where its line sits above the label's baseline.
KEY_HEIGHT = 0.7

# The least room, in points, kept between the heading's text and the legend.
CLEARANCE = 12.0


@dataclass(frozen=True)
class LegendPlan:
    """How a framed chart's legend is laid out."""

    labels: tuple[str, ...]
    columns: int
    height: float  # inches
    beside: bool   # beside the heading, or under it
    width: float = 0.0  # inches


def text_width(text: str,
               size: float,
               family: str,
               italic: bool = False,
               weight: int = 400) -> float:
    """How wide *text* is set, in inches."""
    return metrics.width(text, size, Font(family, weight, italic))


def plan(labels: Sequence[str],
         size: float,
         font: Font,
         heading_width: float,
         room: float,
         under: bool = False,
         columns: int | None = None) -> LegendPlan | None:
    """Lay out a legend for the heading band, or under the plot, or in a set
    number of columns.

    Args:
        labels: The entries' labels.
        size: Their size, in points.
        font: Their face.
        heading_width: How far the heading's text reaches from the plot's
            left edge, in inches.
        room: The plot's width, in inches.
        under: Lay it out across the plot's whole width, under it.
        columns: A set number of columns, such as one for a legend beside the
            plot.

    Returns:
        LegendPlan | None: None when there are no entries.
    """
    if not labels:
        return None

    entry = (max(metrics.width(label, size, font) for label in labels)
             + (HANDLE_LENGTH + HANDLE_GAP) * size / 72.0)
    gap = COLUMN_GAP * size / 72.0

    if columns is not None or under:
        fits_beside = False
        columns = columns or max(1, _columns(len(labels), entry, gap, room))
    else:
        beside = room - heading_width - CLEARANCE / 72.0
        columns = _columns(len(labels), entry, gap, beside)
        fits_beside = columns > 0

        if not fits_beside:
            columns = max(1, _columns(len(labels), entry, gap, room))

    rows = -(-len(labels) // columns)

    return LegendPlan(labels=tuple(labels),
                      columns=columns,
                      height=rows * ROW_HEIGHT * size / 72.0,
                      beside=fits_beside,
                      width=columns * entry + (columns - 1) * gap)


def _columns(count: int,
             entry: float,
             gap: float,
             width: float) -> int:
    """The most columns of *entry*-wide entries that fit in *width*, or 0."""
    for columns in range(count, 0, -1):
        if columns * entry + (columns - 1) * gap <= width:
            return columns

    return 0


def entries(items: Sequence[LegendItem],
            columns: int,
            size: float,
            font: Font,
            anchor: tuple[float, float],
            corner: str) -> tuple[Box, tuple[LegendEntry, ...]]:
    """Each entry's key and label, with the legend's *corner* (such as
    "lower right") at *anchor*.

    Entries fill the columns in turn, top to bottom, each column as wide as
    its widest entry, as matplotlib sets a legend.
    """
    em = size / 72.0
    line = metrics.box(metrics.REFERENCE_TEXT, size, font)
    above = max(KEY_HEIGHT * em, line.ascent)
    pitch = above + line.descent + (ROW_HEIGHT - 1.0) * em
    split = [list(part) for part in np.array_split(np.arange(len(items)), columns) if len(part)]
    widths = [max(metrics.width(items[i].label, size, font) for i in column)
              + (HANDLE_LENGTH + HANDLE_GAP) * em for column in split]
    rows = max(len(column) for column in split)
    wide = sum(widths) + (len(split) - 1) * COLUMN_GAP * em
    tall = rows * pitch - (ROW_HEIGHT - 1.0) * em
    left = anchor[0] - (wide if "right" in corner else 0.0)
    bottom = anchor[1] - (tall if "upper" in corner else 0.0)
    placed: list[LegendEntry] = []
    x = left

    for column, column_width in zip(split, widths, strict=True):
        for row, index in enumerate(column):
            item = items[index]
            baseline = bottom + tall - above - row * pitch
            key = Box(x, baseline, HANDLE_LENGTH * em, KEY_HEIGHT * em)
            text = Text(item.label, key.right + HANDLE_GAP * em, baseline, size, font,
                        "text-primary", ha="left", va="baseline")
            placed.append(LegendEntry(item.label, item.colour, item.key, key, text))
        x += column_width + COLUMN_GAP * em

    order = [index for column in split for index in column]
    by_item = dict(zip(order, placed, strict=True))

    return Box(left, bottom, wide, tall), tuple(by_item[i] for i in range(len(items)))
