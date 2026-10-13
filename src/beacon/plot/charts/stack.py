# src/beacon/plot/charts/stack.py
"""
Charts stacked in one column under one heading, sharing their horizontal
axis: a level over its drawdown.

Each chart is built on its own and its one panel taken into the stack; the
panels below the first measure the same dates, so only the lowest writes its
tick labels.
"""
# BN-299: the performance chart's two panels. Grid (BN-300) lays out charts
# in rows and columns.
import dataclasses
from collections.abc import Sequence

from .. import design
from ..design import StyleArgument
from ..layout import Description, LegendItem
from ..marks import Mark
from ..style import figure_size
from .base import Chart, Plan

# The space between stacked panels, as a share of their average height.
GAP = 0.12


class Stacked(Chart):
    """Charts stacked top to bottom under one frame.

    Args:
        charts: The charts, top to bottom; each draws one panel.
        weights: Each panel's share of the height.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        gap: The space between panels, as a share of their average height.
        kind: The designed size to draw at (beacon.plot.style.FIGSIZE).
        style: Settings for this chart alone: a Style, or settings by name.
    """

    def __init__(self,
                 charts: Sequence[Chart],
                 weights: Sequence[float],
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 gap: float = GAP,
                 kind: str = "stacked",
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        if len(charts) != len(weights) or not charts:
            raise ValueError("a stack needs one weight per chart, and at least one chart")
        self.charts = list(charts)
        self.weights = list(weights)
        self.gap = gap
        self.size_kind = kind

    def _size(self) -> tuple[float, float]:
        with design.scoped(self.style):
            return figure_size(self.size_kind)

    def _build(self) -> Plan:
        panels = []
        marks: list[Mark] = []
        legend: list[LegendItem] = []
        last = len(self.charts) - 1

        for index, (chart, weight) in enumerate(zip(self.charts, self.weights, strict=True)):
            plan = chart.plan
            panel = plan.description.panels[0]
            x = panel.x if index == 0 else dataclasses.replace(panel.x, shared=0)
            if index < last:
                x = dataclasses.replace(x, show_labels=False, title="")
            panels.append(dataclasses.replace(panel, x=x, weight=weight))
            marks.extend(dataclasses.replace(mark, panel=index) for mark in plan.marks)
            legend.extend(plan.description.legend)

        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=tuple(panels), gap=self.gap,
                                  legend=tuple(legend))

        return Plan(description, tuple(marks), front=self.charts[0].plan.front)
