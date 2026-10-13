# src/beacon/plot/charts/grid.py
"""
Several charts in one figure, under one heading and one notes line: `Grid`.
"""
# BN-300.
import dataclasses
from collections.abc import Sequence

from .. import design
from ..design import StyleArgument
from ..layout import Description, LegendItem, Panel
from ..marks import Mark
from ..style import sized
from .base import Chart, Plan

# A grid's designed size: as wide as every chart, and this tall for its
# heading and notes plus this much for each row, in inches.
WIDTH = 6.0
FRAME_HEIGHT = 2.0
ROW_HEIGHT = 1.75


class Grid(Chart):
    """Charts laid out in rows and columns under one heading and one notes
    line, each panel with its chart's title above it.

    The same chart named in neighbouring cells takes them all, so a tall
    chart can run beside two short ones. Each column is given the room its
    labels take, and the heading and legend line up with the outer panels'
    edges.

    Args:
        charts: Rows of charts, top to bottom, each row left to right; None
            leaves a cell empty, and one chart in a rectangle of cells spans
            them. Each chart draws one panel, titled with its own title; its
            subtitle and notes are not drawn.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        heights: The rows' shares of the height, such as (3, 1) for a tall
            row over a short one; equal by default.
        widths: The columns' shares of the width; equal by default.
        sharex: Whether every panel on dates runs over the same dates, even
            where one series starts later. A panel on dates with another
            directly beneath it then leaves its dates to the one beneath.
        style: Settings for this chart alone: a Style, or settings by name.

    Raises:
        ValueError: If there is no chart, one draws more than one panel, or
            one is named in cells that do not make a rectangle.
    """

    def __init__(self,
                 charts: Sequence[Sequence[Chart | None]],
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 heights: Sequence[float] | None = None,
                 widths: Sequence[float] | None = None,
                 sharex: bool = False,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.rows = [list(row) for row in charts]
        self.heights = tuple(float(share) for share in heights or ())
        self.widths = tuple(float(share) for share in widths or ())
        self.sharex = sharex
        self.cells = _cells(self.rows)

        if not self.cells:
            raise ValueError("a grid needs at least one chart")

        self.columns = max(len(row) for row in self.rows)

    def _size(self) -> tuple[float, float]:
        with design.scoped(self.style):
            return sized(WIDTH, FRAME_HEIGHT + ROW_HEIGHT * len(self.rows))

    def _build(self) -> Plan:
        panels: list[Panel] = []
        marks: list[Mark] = []
        legends: list[tuple[LegendItem, ...]] = []

        for chart, (cell, span) in self.cells:
            plan = chart.plan
            if len(plan.description.panels) != 1:
                raise ValueError(f"a grid's charts draw one panel each, and "
                                 f"{type(chart).__name__} draws "
                                 f"{len(plan.description.panels)}")
            index = len(panels)
            panels.append(dataclasses.replace(plan.description.panels[0], cell=cell, span=span,
                                              title=chart.title, weight=1.0))
            marks.extend(dataclasses.replace(mark, panel=index) for mark in plan.marks)
            legends.append(plan.description.legend)

        if self.sharex:
            panels = _one_calendar(panels)

        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=tuple(panels),
                                  legend=_shared_legend(legends), columns=self.columns,
                                  grid=True, heights=self.heights, widths=self.widths)

        return Plan(description, tuple(marks))


Place = tuple[tuple[int, int], tuple[int, int]]


def _cells(rows: list[list[Chart | None]]) -> list[tuple[Chart, Place]]:
    """Each chart with its first cell and how many rows and columns it
    takes, in the order it is first met, row by row.

    Raises:
        ValueError: If a chart's cells do not make a rectangle.
    """
    seen: dict[int, tuple[Chart, list[tuple[int, int]]]] = {}

    for row, charts in enumerate(rows):
        for column, chart in enumerate(charts):
            if chart is not None:
                seen.setdefault(id(chart), (chart, []))[1].append((row, column))

    found = []
    for chart, cells in seen.values():
        top, left = min(row for row, _ in cells), min(column for _, column in cells)
        bottom, right = max(row for row, _ in cells), max(column for _, column in cells)
        if len(cells) != (bottom - top + 1) * (right - left + 1):
            raise ValueError(f"{type(chart).__name__} {chart.title!r} is in cells that do "
                             f"not make a rectangle")
        found.append((chart, ((top, left), (bottom - top + 1, right - left + 1))))

    return found


def _one_calendar(panels: list[Panel]) -> list[Panel]:
    """Panels on dates made to share the first one's axis, over every one's
    dates; one with another on dates directly beneath it leaves its dates
    to that one."""
    dated = [index for index, panel in enumerate(panels) if panel.x.kind == "date"]

    if len(dated) < 2:
        return panels

    first = dated[0]
    low = min(panels[index].x.low for index in dated)
    high = max(panels[index].x.high for index in dated)
    shared = list(panels)

    for index in dated:
        x = panels[index].x
        x = (dataclasses.replace(x, low=low, high=high) if index == first
             else dataclasses.replace(x, shared=first))
        if any(_beneath(panels[index], panels[other]) for other in dated):
            x = dataclasses.replace(x, show_labels=False, title="")
        shared[index] = dataclasses.replace(panels[index], x=x)

    return shared


def _beneath(upper: Panel,
             lower: Panel) -> bool:
    """Whether *lower* starts in the row just below *upper* and shares a
    column with it."""
    row = upper.cell[0] + upper.span[0]
    columns = range(upper.cell[1], upper.cell[1] + upper.span[1])
    below = range(lower.cell[1], lower.cell[1] + lower.span[1])

    return lower.cell[0] == row and bool(set(columns) & set(below))


def _shared_legend(legends: list[tuple[LegendItem, ...]]) -> tuple[LegendItem, ...]:
    """One legend for the figure, where every panel with a legend names the
    same series; otherwise none, since one legend could not say which panel
    a series is in."""
    named = [legend for legend in legends if legend]

    if not named:
        return ()

    labels = {tuple(item.label for item in legend) for legend in named}

    return named[0] if len(labels) == 1 else ()
