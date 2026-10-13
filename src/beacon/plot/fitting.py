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
from dataclasses import dataclass

from . import metrics, panels
from .design import Style
from .layout import Axis, AxisLayout, Box, Description, Label, Panel, PanelLayout, Text, Tick
from .panels import AxisState, Sizes

# Where the plot's sides start, as shares of the figure's width, before they
# widen to fit what the chart draws.
SIDES = (0.10, 0.92)

# The tick labels left blank on each axis of a panel, by axis: "x", "y", "y2".
Blanks = dict[str, float]


@dataclass
class PanelState:
    """A panel's axes as the layout moves them."""

    x: AxisState
    y: AxisState
    y2: AxisState | None = None


@dataclass(frozen=True)
class Drawn:
    """A panel's axes and titles as they would be drawn now."""

    x: AxisLayout
    y: AxisLayout
    y2: AxisLayout | None
    bar: AxisLayout | None
    x_title: Text | None = None
    y_title: Text | None = None
    y2_title: Text | None = None


def _states(chart: Description) -> list[PanelState]:
    """Each panel's axes as the layout moves them; a shared axis is one
    state for the panels that share it."""
    states: list[PanelState] = []

    for panel in chart.panels:
        x = (states[panel.x.shared].x if panel.x.shared is not None
             else AxisState(panel.x, panel.x.low, panel.x.high))
        y2 = AxisState(panel.y2, panel.y2.low, panel.y2.high) if panel.y2 is not None else None
        states.append(PanelState(x, AxisState(panel.y, panel.y.low, panel.y.high), y2))

    return states


class Passes:
    """The panels' geometry as it is fitted, pass by pass."""

    def __init__(self,
                 chart: Description,
                 style: Style):
        self.chart = chart
        self.style = style
        self.sizes = Sizes.of(style)
        self.states = _states(chart)
        self.sides = list(SIDES)
        self.blanks: list[Blanks] = [{} for _ in chart.panels]
        self.placed: list[tuple[Box, Box | None]] = []

    def place(self,
              bottom: float,
              top: float) -> None:
        """Put the panels between *bottom* and *top* (figure fractions)."""
        chart = self.chart
        grid = panels.cells([panel.weight for panel in chart.panels], chart.gap,
                            self.sides[0], self.sides[1], bottom, top)
        self.placed = [panels.boxes(panel, cell, (state.x, state.y), chart.size)
                       for panel, cell, state in zip(chart.panels, grid, self.states,
                                                     strict=True)]

    def end_axes(self) -> None:
        """End each panel's measuring axes on their ticks at the size each
        panel has now, and blank the labels that would collide."""
        axes = self.style.axes

        for index, (panel, (plot, _), state) in enumerate(
                zip(self.chart.panels, self.placed, self.states, strict=True)):
            if axes.end_on_ticks:
                panels.end(state.x, plot.width, self.sizes, True)
                panels.end(state.y, plot.height, self.sizes, False)
                if state.y2 is not None:
                    panels.end(state.y2, plot.height, self.sizes, False)
            cornered = {"x": self._cornered(state.x, plot.width, True),
                        "y": self._cornered(state.y, plot.height, False)}
            if state.y2 is not None:
                cornered["y2"] = self._cornered(state.y2, plot.height, False)
            self.blanks[index] = _blanks(panel, state, cornered, axes.blank_lowest_label)

    def _cornered(self,
                  state: AxisState,
                  length: float,
                  across: bool) -> bool:
        """Whether a measuring axis has a tick at its low end, where it meets
        the other axis: always so for one ended on its ticks."""
        if not state.spec.measures:
            return False
        if state.ended:
            return True

        found, _ = panels.axis_ticks(state, length, self.sizes, across)
        low, high = sorted((state.low, state.high))

        return bool(found) and abs(found[0].value - low) <= (high - low) * 1e-9

    def axes(self) -> list[Drawn]:
        """Each panel's axes and their titles as they would be drawn now."""
        drawn = []

        for index, (panel, (plot, bar), state) in enumerate(
                zip(self.chart.panels, self.placed, self.states, strict=True)):
            blanks = self.blanks[index]
            x = self._axis(state.x, plot.width, True, blanks.get("x"))
            y = self._axis(state.y, plot.height, False, blanks.get("y"))
            y2 = (self._axis(state.y2, plot.height, False, blanks.get("y2"))
                  if state.y2 is not None else None)
            bar_axis = None
            if panel.colour_bar is not None and bar is not None:
                low, high = panel.colour_bar.low, panel.colour_bar.high
                bar_axis = self._axis(AxisState(Axis("number", low, high), low, high),
                                      bar.height, False, None)
            drawn.append(Drawn(x, y, y2, bar_axis))

        return self._titled(drawn)

    def _axis(self,
              state: AxisState,
              length: float,
              across: bool,
              blank: float | None) -> AxisLayout:
        found, offset = panels.axis_ticks(state, length, self.sizes, across, blank)

        return _axis(state, found, offset)

    def _titled(self,
                drawn: list[Drawn]) -> list[Drawn]:
        """Each panel's axis titles: the left ones in one column, clear of
        every panel's tick labels, and the right ones beyond their own."""
        sizes, axes = self.sizes, self.style.axes
        lefts = [panels.y_label_left(laid.y, plot, sizes,
                                     sizes.label_pad(state.y.spec, axes.y_ticks))
                 for (plot, _), laid, state in zip(self.placed, drawn, self.states, strict=True)]
        column = min(lefts) - sizes.title_pad / 72.0
        titled = []

        for panel, (plot, _), laid in zip(self.chart.panels, self.placed, drawn, strict=True):
            y_title = panels.y_title(panel.y.title, column, plot, sizes) if panel.y.title else None
            x_title = y2_title = None
            if panel.x.title:
                pad = sizes.label_pad(panel.x, axes.x_ticks)
                top = panels.x_label_bottom(laid.x, plot, sizes, pad) - sizes.title_pad / 72.0
                x_title = panels.x_title(panel.x.title, top, plot, sizes)
            if panel.y2 is not None and laid.y2 is not None and panel.y2.title:
                pad = sizes.label_pad(panel.y2, axes.y_ticks)
                right = panels.y_label_right(laid.y2, plot, sizes, pad) + sizes.title_pad / 72.0
                y2_title = panels.y2_title(panel.y2.title, right, plot, sizes)
            titled.append(dataclasses.replace(laid, x_title=x_title, y_title=y_title,
                                              y2_title=y2_title))

        return titled

    def reach(self) -> tuple[float, float]:
        """How far left and right anything the panels draw reaches, in
        inches."""
        spans: list[tuple[float, float]] = []

        for panel, placed, laid, state in zip(self.chart.panels, self.placed, self.axes(),
                                              self.states, strict=True):
            spans.extend(self._spans(panel, placed, laid, state))

        return min(span[0] for span in spans), max(span[1] for span in spans)

    def _spans(self,
               panel: Panel,
               placed: tuple[Box, Box | None],
               laid: Drawn,
               state: PanelState) -> list[tuple[float, float]]:
        """How far across each thing one panel draws reaches."""
        sizes, axes = self.sizes, self.style.axes
        plot, bar = placed
        spans = [(plot.left, plot.right)]
        spans.extend(_title_span(title)
                     for title in (laid.x_title, laid.y_title, laid.y2_title) if title)
        spans.extend(_drawn(panels.label_reach(label, plot, (state.x, state.y))
                            for label in panel.labels))

        if bar is not None and laid.bar is not None and panel.colour_bar is not None:
            spans.extend(_bar_spans(bar, laid.bar, panel.colour_bar.labels, sizes))

        if panel.origin:
            # The tick labels sit along the axes through zero, inside the plot.
            return spans

        spans.extend(panels.tick_label_reach(laid.x, state.x, plot, sizes, axes.x_ticks))
        if laid.y.show_labels:
            pad = sizes.label_pad(panel.y, axes.y_ticks) / 72.0
            spans.extend((plot.left - pad - _wide(tick.label, sizes), plot.left - pad)
                         for tick in laid.y.ticks if tick.label)
        if panel.y2 is not None and laid.y2 is not None:
            start = plot.right + sizes.label_pad(panel.y2, axes.y_ticks) / 72.0
            spans.extend((start, start + _wide(tick.label, sizes))
                         for tick in laid.y2.ticks if tick.label)
        spans.extend(_offset_spans(laid.x, laid.y, plot, sizes))

        return spans

    def fit_sides(self,
                  margin: float) -> None:
        """Widen the sides until nothing the panels draw runs past *margin*
        inches from either edge."""
        width = self.chart.size[0]
        left, right = self.reach()
        self.sides[0] += max(margin - left, 0.0) / width
        self.sides[1] -= max(right - (width - margin), 0.0) / width


def _blanks(panel: Panel,
            state: PanelState,
            cornered: dict[str, bool],
            lowest: bool) -> Blanks:
    """The tick labels left blank so they do not collide: zero on both axes
    where they cross there; otherwise the lowest label of each vertical axis
    that shares its corner with a tick on the horizontal one."""
    if panel.origin:
        return {name: 0.0 for name, axis in (("x", state.x), ("y", state.y))
                if min(axis.low, axis.high) < 0.0 < max(axis.low, axis.high)}

    if not lowest or not cornered.get("x"):
        return {}

    blanks = {}
    if cornered.get("y"):
        blanks["y"] = min(state.y.low, state.y.high)
    if state.y2 is not None and cornered.get("y2"):
        blanks["y2"] = min(state.y2.low, state.y2.high)

    return blanks


def _wide(text: str,
          sizes: Sizes) -> float:
    return metrics.width(text, sizes.tick_label, sizes.font)


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
        # Turned about its anchor: a title reading up the left sits left of
        # it, and one on the right sits right of it.
        return (title.x, title.x + wide) if title.va == "top" else (title.x - wide, title.x)

    return panels.across_from(title.x, wide, title.ha)


def _offset_spans(x: AxisLayout,
                  y: AxisLayout,
                  plot: Box,
                  sizes: Sizes) -> list[tuple[float, float]]:
    """The offsets written at the axes' ends: under the x axis's right end,
    over the y axis's top."""
    spans = []
    if x.offset:
        spans.append((plot.right - _wide(x.offset, sizes), plot.right))
    if y.offset:
        spans.append((plot.left, plot.left + _wide(y.offset, sizes)))

    return spans


def _bar_spans(bar: Box,
               axis: AxisLayout,
               labels: tuple[Label, ...],
               sizes: Sizes) -> list[tuple[float, float]]:
    """A colour bar's box, its tick labels at its right, and its labels."""
    start = bar.right + sizes.pad / 72.0
    spans = [(bar.left, bar.right)]
    spans.extend((start, start + _wide(tick.label, sizes)) for tick in axis.ticks if tick.label)
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
