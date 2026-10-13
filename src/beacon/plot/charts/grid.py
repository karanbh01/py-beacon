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

    The plots are all the same size, each column given the room its labels
    take, and the heading and legend line up with the outer panels' edges.

    Args:
        charts: Rows of charts, top to bottom, each row left to right; None
            leaves a cell empty. Each chart draws one panel, titled with its
            own title; its subtitle and notes are not drawn.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        sharex: Whether every panel on dates runs over the same dates, even
            where one series starts later.
        style: Settings for this chart alone: a Style, or settings by name.

    Raises:
        ValueError: If there is no chart, or one draws more than one panel.
    """

    def __init__(self,
                 charts: Sequence[Sequence[Chart | None]],
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 sharex: bool = False,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.rows = [list(row) for row in charts]
        self.sharex = sharex
        placed = [chart for row in self.rows for chart in row if chart is not None]

        if not placed:
            raise ValueError("a grid needs at least one chart")

        self.columns = max(len(row) for row in self.rows)

    def _size(self) -> tuple[float, float]:
        with design.scoped(self.style):
            return sized(WIDTH, FRAME_HEIGHT + ROW_HEIGHT * len(self.rows))

    def _build(self) -> Plan:
        panels: list[Panel] = []
        marks: list[Mark] = []
        legends: list[tuple[LegendItem, ...]] = []

        for row, charts in enumerate(self.rows):
            for column, chart in enumerate(charts):
                if chart is None:
                    continue
                plan = chart.plan
                if len(plan.description.panels) != 1:
                    raise ValueError(f"a grid's charts draw one panel each, and "
                                     f"{type(chart).__name__} draws "
                                     f"{len(plan.description.panels)}")
                index = len(panels)
                panels.append(dataclasses.replace(plan.description.panels[0], cell=(row, column),
                                                  title=chart.title, weight=1.0))
                marks.extend(dataclasses.replace(mark, panel=index) for mark in plan.marks)
                legends.append(plan.description.legend)

        if self.sharex:
            panels = _one_calendar(panels)

        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=tuple(panels),
                                  legend=_shared_legend(legends), columns=self.columns)

        return Plan(description, tuple(marks))


def _one_calendar(panels: list[Panel]) -> list[Panel]:
    """Panels on dates made to share the first one's axis, over every one's
    dates."""
    dated = [index for index, panel in enumerate(panels) if panel.x.kind == "date"]

    if len(dated) < 2:
        return panels

    first = dated[0]
    low = min(panels[index].x.low for index in dated)
    high = max(panels[index].x.high for index in dated)
    shared = list(panels)
    shared[first] = dataclasses.replace(panels[first],
                                        x=dataclasses.replace(panels[first].x, low=low,
                                                              high=high))

    for index in dated[1:]:
        shared[index] = dataclasses.replace(panels[index],
                                            x=dataclasses.replace(panels[index].x,
                                                                  shared=first))

    return shared


def _shared_legend(legends: list[tuple[LegendItem, ...]]) -> tuple[LegendItem, ...]:
    """One legend for the figure, where every panel with a legend names the
    same series; otherwise none, since one legend could not say which panel
    a series is in."""
    named = [legend for legend in legends if legend]

    if not named:
        return ()

    labels = {tuple(item.label for item in legend) for legend in named}

    return named[0] if len(labels) == 1 else ()
