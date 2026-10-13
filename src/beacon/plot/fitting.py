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
from .gridding import Grid
from .layout import (
    Axis,
    AxisLayout,
    Box,
    ColourBar,
    Description,
    Label,
    Panel,
    Text,
    Tick,
)
from .metrics import Font
from .panels import AxisState, Sizes

# The weight a grid's panel titles are set in, the heading title's.
TITLE_WEIGHT = 400

# Names closer than this, in multiples of their size, do not fit side by
# side. Ordered categories are then labelled every so many, by the first of
# these steps that fits (every other, every third: a year's quarters, ...);
# a table's names are slanted, or stood upright.
NAME_GAP = 0.6
STEPS = (2, 3, 4, 6, 12)
SLANT = 45.0
UPRIGHT = 90.0

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
                 style: Style,
                 margin: float = 0.0):
        self.chart = chart
        self.style = style
        self.margin = margin
        self.sizes = Sizes.of(style)
        self.states = _states(chart)
        self.sides = list(SIDES)
        self.blanks: list[Blanks] = [{} for _ in chart.panels]
        self.placed: list[tuple[Box, Box | None]] = []
        self.grid = (Grid(chart, style.size.scale, self.sizes.axis_title,
                          Font(style.heading.title_font, TITLE_WEIGHT))
                     if chart.columns > 1 else None)

    def place(self,
              bottom: float,
              top: float) -> None:
        """Put the panels between *bottom* and *top* (figure fractions)."""
        chart = self.chart
        if self.grid is not None:
            height = chart.size[1]
            grid = self.grid.cells(bottom * height, top * height, self.margin)
        else:
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
            x = _slanted(self._axis(state.x, plot.width, True, blanks.get("x")), plot,
                         self.sizes)
            y = self._axis(state.y, plot.height, False, blanks.get("y"))
            y2 = (self._axis(state.y2, plot.height, False, blanks.get("y2"))
                  if state.y2 is not None else None)
            bar_axis = None
            if panel.colour_bar is not None and bar is not None:
                bar_axis = self._axis(_bar_state(panel.colour_bar), bar.height, False, None)
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
        # The left titles line up within each column of panels.
        columns = {panel.cell[1]: min(left for other, left in zip(self.chart.panels, lefts,
                                                                    strict=True)
                                      if other.cell[1] == panel.cell[1])
                   for panel in self.chart.panels}
        titled = []

        for panel, (plot, _), laid in zip(self.chart.panels, self.placed, drawn, strict=True):
            column = columns[panel.cell[1]] - sizes.title_pad / 72.0
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
        inches from either edge; in a grid, give each row and column the room
        its panels take."""
        if self.grid is not None:
            self._measure_grid()
            return

        width = self.chart.size[0]
        left, right = self.reach()
        self.sides[0] += max(margin - left, 0.0) / width
        self.sides[1] -= max(right - (width - margin), 0.0) / width


    def _measure_grid(self) -> None:
        assert self.grid is not None
        spans, lows = [], []

        for panel, placed, laid, state in zip(self.chart.panels, self.placed, self.axes(),
                                              self.states, strict=True):
            found = self._spans(panel, placed, laid, state)
            spans.append((min(span[0] for span in found), max(span[1] for span in found)))
            lows.append(self._lowest(panel, placed[0], laid))

        self.grid.measure([plot for plot, _ in self.placed], spans, lows)

    def _lowest(self,
                panel: Panel,
                plot: Box,
                laid: Drawn) -> float:
        """How far down a panel's drawing reaches: its axis title, or its tick
        labels."""
        if panel.origin:
            return plot.bottom
        if laid.x_title is not None:
            title = laid.x_title
            return title.y - metrics.box(title.text, title.size, title.font).height

        pad = self.sizes.label_pad(panel.x, self.style.axes.x_ticks)

        return panels.x_label_bottom(laid.x, plot, self.sizes, pad)

    def outer(self) -> tuple[float, float]:
        """The plots' outer edges, which the heading and the legend line up
        with: the top panel's, or a grid's outermost."""
        if self.grid is not None:
            return self.grid.outer([plot for plot, _ in self.placed])

        main = self.placed[0][0]

        return main.left, main.right

    def panel_titles(self) -> list[Text | None]:
        """Each panel's title above it, in a grid."""
        found: list[Text | None] = []

        for panel, (plot, _) in zip(self.chart.panels, self.placed, strict=True):
            if self.grid is None or not panel.title:
                found.append(None)
                continue
            x, y = self.grid.title_at(plot)
            found.append(Text(panel.title, x, y, self.sizes.axis_title,
                              Font(self.style.heading.title_font, TITLE_WEIGHT), "text-primary",
                              ha="left", va="bottom", id="beacon-panel-title"))

        return found


def _slanted(axis: AxisLayout,
             plot: Box,
             sizes: Sizes) -> AxisLayout:
    """An axis whose names do not fit side by side, made to read: categories
    in order (a column chart's) labelled every few, the smallest step that
    fits, so they stay level; a table's (a heatmap's), where every name
    matters, slanted, or stood upright where a slant does not fit either.
    Names that fit, and every other axis, are left as they are."""
    if axis.kind not in ("names", "fixed") or axis.rotation or len(axis.ticks) < 2:
        return axis

    slot = plot.width / len(axis.ticks)
    gap = NAME_GAP * sizes.tick_label / 72.0
    widest = max(panels.label_span(tick.label, sizes.tick_label, sizes.font, 0.0)[0]
                 for tick in axis.ticks)

    if widest + gap <= slot:
        return axis

    if axis.kind == "names":
        step = next((step for step in STEPS if widest + gap <= slot * step),
                    len(axis.ticks))
        thinned = tuple(Tick(tick.value, tick.label if index % step == 0 else "")
                        for index, tick in enumerate(axis.ticks))
        return dataclasses.replace(axis, ticks=thinned)

    slanted = max(panels.label_span(tick.label, sizes.tick_label, sizes.font, SLANT)[0]
                  for tick in axis.ticks)
    if slanted + gap <= slot:
        return dataclasses.replace(axis, rotation=SLANT, align="right")

    return dataclasses.replace(axis, rotation=UPRIGHT, align="center")


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


def _bar_state(bar: ColourBar) -> AxisState:
    """A colour bar's scale: ticks along it, or labels at its two ends."""
    if bar.ends is None:
        spec = Axis("number", bar.low, bar.high)
    else:
        spec = Axis("fixed", bar.low, bar.high, ticks=(bar.low, bar.high), names=bar.ends)

    return AxisState(spec, bar.low, bar.high)


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
