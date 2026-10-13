# src/beacon/plot/gridding.py
"""
Panels laid out in rows and columns, under one heading.

Each column is given the room its panels' labels take on either side, each
row the room its panels' titles take above and their tick labels below, and
the plots share what is left by each row's and column's share (equal unless
set), so the labels between panels never run into the next one. A panel may
take several rows or columns. The room each needs is measured from the panels
as drawn in the last pass, so the layout settles over its passes.
"""
# BN-300.
from dataclasses import dataclass, field

from . import metrics
from .layout import Box, Description

# Between columns and between rows, in inches at the chart's scale, beyond
# the room the labels take.
COLUMN_GAP = 0.2
ROW_GAP = 0.12

# Between a panel's title and its plot, in points at the chart's scale.
TITLE_GAP = 4.0

# The room assumed before a pass has measured any, in inches.
FIRST_GUESS = {"left": 0.35, "right": 0.05, "below": 0.2}


@dataclass
class Needs:
    """The room each column takes beside its plots and each row above and
    below them, in inches."""

    left: list[float] = field(default_factory=list)
    right: list[float] = field(default_factory=list)
    above: list[float] = field(default_factory=list)
    below: list[float] = field(default_factory=list)


def shape(chart: Description) -> tuple[int, int]:
    """A grid's rows and columns."""
    rows = max(panel.cell[0] + panel.span[0] for panel in chart.panels)
    columns = max(panel.cell[1] + panel.span[1] for panel in chart.panels)

    return rows, max(columns, chart.columns)


def _shares(weights: tuple[float, ...],
            count: int) -> list[float]:
    """Each row's or column's share, from its weight; equal without any."""
    found = list(weights[:count]) + [1.0] * max(count - len(weights), 0)
    total = sum(found)

    return [weight / total for weight in found]


class Grid:
    """Where a grid's panels go, from the room each row and column needs."""

    def __init__(self,
                 chart: Description,
                 zoom: float,
                 title_size: float,
                 title_font: metrics.Font):
        self.chart = chart
        self.zoom = zoom
        self.rows, self.columns = shape(chart)
        self.title_size = title_size
        self.title_font = title_font
        self.heights = _shares(chart.heights, self.rows)
        self.widths = _shares(chart.widths, self.columns)
        titled = [any(panel.title for panel in chart.panels if panel.cell[0] == row)
                  for row in range(self.rows)]
        self.needs = Needs(left=[FIRST_GUESS["left"]] * self.columns,
                           right=[FIRST_GUESS["right"]] * self.columns,
                           above=[self.title_room() if has else 0.0 for has in titled],
                           below=[FIRST_GUESS["below"]] * self.rows)

    def title_room(self) -> float:
        """The room a panel's title takes above its plot, in inches."""
        shape_of = metrics.box("Title", self.title_size, self.title_font)

        return shape_of.height + TITLE_GAP * self.zoom * 0.8 / 72.0

    def cells(self,
              bottom: float,
              top: float,
              margin: float) -> list[tuple[float, float, float, float]]:
        """Each panel's cell as figure fractions (left, bottom, right, top),
        between *bottom* and *top* inches up the figure and *margin* inches in
        from its sides. A panel taking several rows or columns runs from the
        first one's edge to the last one's."""
        width, height = self.chart.size
        needs = self.needs
        across = max(width - 2.0 * margin - sum(needs.left) - sum(needs.right)
                     - (self.columns - 1) * COLUMN_GAP * self.zoom, 0.0)
        up = max(top - bottom - sum(needs.above) - sum(needs.below)
                 - (self.rows - 1) * ROW_GAP * self.zoom, 0.0)
        lefts, rights, tops, bottoms = [], [], [], []
        x, y = margin, top

        for column, share in enumerate(self.widths):
            x += needs.left[column]
            lefts.append(x)
            x += across * share
            rights.append(x)
            x += needs.right[column] + COLUMN_GAP * self.zoom

        for row, share in enumerate(self.heights):
            y -= needs.above[row]
            tops.append(y)
            y -= up * share
            bottoms.append(y)
            y -= needs.below[row] + ROW_GAP * self.zoom

        found = []
        for panel in self.chart.panels:
            row, column = panel.cell
            last_row, last_column = row + panel.span[0] - 1, column + panel.span[1] - 1
            found.append((lefts[column] / width, bottoms[last_row] / height,
                          rights[last_column] / width, tops[row] / height))

        return found

    def measure(self,
                placed: list[Box],
                spans: list[tuple[float, float]],
                lows: list[float]) -> None:
        """Take each column's and row's room from the panels as drawn: a
        panel's labels at its left need room in its first column, at its
        right in its last, and below it in its last row.

        Args:
            placed: Each panel's plot.
            spans: How far left and right each panel's drawing reaches.
            lows: How far down each panel's drawing reaches.
        """
        left = [0.0] * self.columns
        right = [0.0] * self.columns
        below = [0.0] * self.rows

        for panel, plot, (low, high), bottom in zip(self.chart.panels, placed, spans, lows,
                                                    strict=True):
            row, column = panel.cell
            last_row, last_column = row + panel.span[0] - 1, column + panel.span[1] - 1
            left[column] = max(left[column], plot.left - low)
            right[last_column] = max(right[last_column], high - plot.right)
            below[last_row] = max(below[last_row], plot.bottom - bottom)

        self.needs.left, self.needs.right, self.needs.below = left, right, below

    def outer(self,
              placed: list[Box]) -> tuple[float, float]:
        """The left edge of the leftmost plots and the right edge of the
        rightmost, which the heading and the legend line up with."""
        return min(plot.left for plot in placed), max(plot.right for plot in placed)

    def title_at(self,
                 plot: Box) -> tuple[float, float]:
        """Where a panel's title sits: its left edge and its bottom."""
        return plot.left, plot.top + TITLE_GAP * self.zoom * 0.8 / 72.0
