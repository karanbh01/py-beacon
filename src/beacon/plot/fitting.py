# src/beacon/plot/fitting.py
"""
A chart's panels fitted between its heading and its notes, pass by pass:
placed, their measuring axes ended on their ticks at the size each has, and
their sides widened until every label they draw is inside the figure.

Used by beacon.plot.arrange, which runs the passes.
"""
# BN-298: the rules beacon.plot.frame applied inside matplotlib (axes ended on
# their ticks, the corner label blanked, the sides fitted), without it.
import dataclasses
from collections.abc import Iterable

from . import metrics, panels
from .design import Style
from .layout import Axis, AxisLayout, Box, Chart, Label, Panel, PanelLayout, Text, Tick
from .panels import AxisState, Sizes

# Where the plot's sides start, as shares of the figure's width, before they
# widen to fit what the chart draws.
SIDES = (0.10, 0.92)


def _states(chart: Chart) -> list[tuple[AxisState, AxisState]]:
    """Each panel's axes as the layout moves them; a shared axis is one
    state for the panels that share it."""
    states: list[tuple[AxisState, AxisState]] = []

    for panel in chart.panels:
        x = (states[panel.x.shared][0] if panel.x.shared is not None
             else AxisState(panel.x, panel.x.low, panel.x.high))
        states.append((x, AxisState(panel.y, panel.y.low, panel.y.high)))

    return states


class Passes:
    """The panels' geometry as it is fitted, pass by pass."""

    def __init__(self,
                 chart: Chart,
                 style: Style):
        self.chart = chart
        self.style = style
        self.sizes = Sizes.of(style)
        self.states = _states(chart)
        self.sides = list(SIDES)
        self.blanks: list[float | None] = [None] * len(chart.panels)
        self.placed: list[tuple[Box, Box | None]] = []

    def place(self,
              bottom: float,
              top: float) -> None:
        """Put the panels between *bottom* and *top* (figure fractions)."""
        chart = self.chart
        grid = panels.cells([panel.weight for panel in chart.panels], chart.gap,
                            self.sides[0], self.sides[1], bottom, top)
        self.placed = [panels.boxes(panel, cell, state, chart.size)
                       for panel, cell, state in zip(chart.panels, grid, self.states, strict=True)]

    def end_axes(self) -> None:
        """End each panel's measuring axes on their ticks at the size each
        panel has now, and blank the corner label where both measure."""
        axes = self.style.axes

        for index, ((plot, _), (x, y)) in enumerate(zip(self.placed, self.states, strict=True)):
            self.blanks[index] = None
            if not axes.end_on_ticks:
                continue
            x_ended = panels.end(x, plot.width, self.sizes, True)
            y_ended = panels.end(y, plot.height, self.sizes, False)
            if x_ended and y_ended and axes.blank_lowest_label:
                self.blanks[index] = min(y.low, y.high)

    def axes(self) -> list[tuple[AxisLayout, AxisLayout, AxisLayout | None]]:
        """Each panel's axes as they would be drawn now: x, y and the colour
        bar's."""
        drawn = []

        for index, (panel, (plot, bar), (x, y)) in enumerate(
                zip(self.chart.panels, self.placed, self.states, strict=True)):
            x_ticks, x_offset = panels.axis_ticks(x, plot.width, self.sizes, True)
            y_ticks, y_offset = panels.axis_ticks(y, plot.height, self.sizes, False,
                                                  self.blanks[index])
            bar_axis = None
            if panel.colour_bar is not None and bar is not None:
                bar_axis = self._bar_axis(panel.colour_bar.low, panel.colour_bar.high, bar)
            drawn.append((_axis(x, x_ticks, x_offset), _axis(y, y_ticks, y_offset), bar_axis))

        return drawn

    def _bar_axis(self,
                  low: float,
                  high: float,
                  bar: Box) -> AxisLayout:
        state = AxisState(Axis("number", low, high), low, high)
        found, offset = panels.axis_ticks(state, bar.height, self.sizes, False)

        return _axis(state, found, offset)

    def titles(self,
               drawn: list[tuple[AxisLayout, AxisLayout, AxisLayout | None]]
               ) -> list[tuple[Text | None, Text | None]]:
        """Each panel's axis titles: the vertical ones in one column, clear of
        every panel's tick labels."""
        sizes = self.sizes
        y_ticks = self.style.axes.y_ticks
        lefts = [panels.y_label_left(y, plot, sizes, sizes.label_pad(state[1].spec, y_ticks))
                 for (plot, _), (_, y, _), state in zip(self.placed, drawn, self.states,
                                                        strict=True)]
        column = min(lefts) - sizes.title_pad / 72.0
        found = []

        for panel, (plot, _), (x, _, _) in zip(self.chart.panels, self.placed, drawn, strict=True):
            y_title = panels.y_title(panel.y.title, column, plot, sizes) if panel.y.title else None
            x_title = None
            if panel.x.title:
                pad = sizes.label_pad(panel.x, self.style.axes.x_ticks)
                top = panels.x_label_bottom(x, plot, sizes, pad) - sizes.title_pad / 72.0
                x_title = panels.x_title(panel.x.title, top, plot, sizes)
            found.append((x_title, y_title))

        return found

    def reach(self) -> tuple[float, float]:
        """How far left and right anything the panels draw reaches, in
        inches."""
        drawn = self.axes()
        titles = self.titles(drawn)
        sizes = self.sizes
        spans: list[tuple[float, float]] = []

        for panel, (plot, bar), (x, y, bar_axis), (x_title, y_title), state in zip(
                self.chart.panels, self.placed, drawn, titles, self.states, strict=True):
            spans.append((plot.left, plot.right))
            spans.extend(panels.tick_label_reach(x, state[0], plot, sizes, self.style.axes.x_ticks))
            pad = sizes.label_pad(panel.y, self.style.axes.y_ticks) / 72.0
            spans.extend((plot.left - pad - metrics.width(tick.label, sizes.tick_label, sizes.font),
                          plot.left - pad) for tick in y.ticks if tick.label and y.show_labels)
            spans.extend(_title_span(title) for title in (x_title, y_title) if title)
            spans.extend(_offset_spans(x, y, plot, sizes))
            spans.extend(_drawn(panels.label_reach(label, plot, state) for label in panel.labels))
            if bar is not None and bar_axis is not None and panel.colour_bar is not None:
                spans.extend(_bar_spans(bar, bar_axis, panel.colour_bar.labels, sizes))

        return min(span[0] for span in spans), max(span[1] for span in spans)

    def fit_sides(self,
                  margin: float) -> None:
        """Widen the sides until nothing the panels draw runs past *margin*
        inches from either edge."""
        width = self.chart.size[0]
        left, right = self.reach()
        self.sides[0] += max(margin - left, 0.0) / width
        self.sides[1] -= max(right - (width - margin), 0.0) / width


def _axis(state: AxisState,
          found: tuple[Tick, ...],
          offset: str) -> AxisLayout:
    spec = state.spec

    return AxisLayout(kind=spec.kind, low=min(state.low, state.high),
                      high=max(state.low, state.high), ticks=found, offset=offset,
                      ended=state.ended, show_labels=spec.show_labels,
                      rotation=spec.rotation, align=spec.align, inverted=spec.inverted)


def _title_span(title: Text) -> tuple[float, float]:
    wide, _ = panels.label_span(title.text, title.size, title.font, title.rotation)
    if title.rotation:
        return title.x - wide, title.x

    return panels.across_from(title.x, wide, title.ha)


def _offset_spans(x: AxisLayout,
                  y: AxisLayout,
                  plot: Box,
                  sizes: Sizes) -> list[tuple[float, float]]:
    """The offsets written at the axes' ends: under the x axis's right end,
    over the y axis's top."""
    spans = []
    if x.offset:
        wide = metrics.width(x.offset, sizes.tick_label, sizes.font)
        spans.append((plot.right - wide, plot.right))
    if y.offset:
        spans.append((plot.left, plot.left + metrics.width(y.offset, sizes.tick_label, sizes.font)))

    return spans


def _bar_spans(bar: Box,
               axis: AxisLayout,
               labels: tuple[Label, ...],
               sizes: Sizes) -> list[tuple[float, float]]:
    """A colour bar's box, its tick labels at its right, and its labels."""
    start = bar.right + sizes.pad / 72.0
    spans = [(bar.left, bar.right)]
    spans.extend((start, start + metrics.width(tick.label, sizes.tick_label, sizes.font))
                 for tick in axis.ticks if tick.label)
    spans.extend(_drawn(panels.label_reach(label, bar, (_unit(), _unit())) for label in labels))

    return spans


def _drawn(spans: Iterable[tuple[float, float] | None]) -> list[tuple[float, float]]:
    return [span for span in spans if span is not None]


def _unit() -> AxisState:
    """An axis from 0 to 1, for text placed at fractions of a box."""
    return AxisState(Axis("fixed", 0.0, 1.0), 0.0, 1.0)


def lay_out_panel(panel: Panel,
                  box: Box,
                  style: Style,
                  tick_label_size: float | None = None) -> PanelLayout:
    """One panel on a figure the caller laid out: its measuring axes ended on
    their ticks at the size it has, with nothing framed around it.

    Args:
        panel: What it draws.
        box: Where it is, in inches.
        style: The settings it is drawn in.
        tick_label_size: Its tick labels' size, where the caller's figure
            sets its own.
    """
    sizes = Sizes.of(style)
    if tick_label_size is not None:
        sizes = dataclasses.replace(sizes, tick_label=tick_label_size)

    x = AxisState(panel.x, panel.x.low, panel.x.high)
    y = AxisState(panel.y, panel.y.low, panel.y.high)
    blank = None

    if style.axes.end_on_ticks:
        x_ended = panels.end(x, box.width, sizes, True)
        y_ended = panels.end(y, box.height, sizes, False)
        if x_ended and y_ended and style.axes.blank_lowest_label:
            blank = min(y.low, y.high)

    x_ticks, x_offset = panels.axis_ticks(x, box.width, sizes, True)
    y_ticks, y_offset = panels.axis_ticks(y, box.height, sizes, False, blank)

    return PanelLayout(box, _axis(x, x_ticks, x_offset), _axis(y, y_ticks, y_offset))
