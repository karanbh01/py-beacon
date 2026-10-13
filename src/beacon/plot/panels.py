# src/beacon/plot/panels.py
"""
A chart's panels, laid out: where each sits, where its axes end, what its
ticks say and how far its text reaches past its sides.

Used by beacon.plot.arrange, which fits the panels between the heading and
the notes. Boxes here are figure fractions, as a grid of panels divides the
figure, until arrange turns them into inches.
"""
# BN-298: the panel grid, a heatmap's squares and its colour bar follow
# matplotlib's gridspec and aspect rules, so the charts keep their shapes.
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import dateticks, metrics, ticks
from .design import Style
from .layout import Axis, AxisLayout, Box, ColourBar, Label, Panel, Text, Tick
from .metrics import Font

# Every chart but its heading is drawn at this share of its designed size
# (beacon.plot.style.SCALE, which this repeats without importing matplotlib).
SCALE = 0.8

# Between a tick and its label, and between the labels and the axis title,
# in points before scaling.
TICK_PAD = 3.5
TITLE_PAD = 4.0

# The weight the chart's own text is set in.
CHART_WEIGHT = 300

# Where a heatmap sits in its cell beside a colour bar, and the bar in its own:
# against each other, centred up.
BESIDE_BAR = (1.0, 0.5)
BAR_ANCHOR = (0.0, 0.5)


@dataclass
class AxisState:
    """An axis's range as the layout moves it: where it ends, and whether it
    has been ended on its ticks."""

    spec: Axis
    low: float
    high: float
    ended: bool = False


@dataclass(frozen=True)
class Sizes:
    """The sizes a style sets for the panels, in points as drawn."""

    tick_label: float
    axis_title: float
    pad: float
    title_pad: float
    tick_length: float
    direction: str
    font: Font

    @classmethod
    def of(cls, style: Style) -> "Sizes":
        """The sizes in a style."""
        k = SCALE * style.size.scale
        zoom = style.size.scale * style.text.scale

        return cls(tick_label=style.axes.tick_label_size * zoom,
                   axis_title=style.axes.title_size * zoom,
                   pad=TICK_PAD * k,
                   title_pad=TITLE_PAD * k,
                   tick_length=style.axes.tick_length * k,
                   direction=style.axes.tick_direction,
                   font=Font(style.text.font, CHART_WEIGHT))

    def label_pad(self,
                  axis: Axis,
                  ticks_on: bool) -> float:
        """From the axis line to its tick labels, in points."""
        length = axis.tick_length if axis.tick_length is not None else self.tick_length
        length = length if ticks_on else 0.0

        return self.pad + {"out": length, "inout": length / 2.0}.get(self.direction, 0.0)


def cells(weights: Sequence[float],
          gap: float,
          left: float,
          right: float,
          bottom: float,
          top: float) -> list[tuple[float, float, float, float]]:
    """Panels stacked in one column, as figure fractions: left, bottom,
    right, top for each, top to bottom."""
    count = len(weights)
    cell = (top - bottom) / (count + gap * (count - 1))
    norm = cell * count / sum(weights)
    heights = [weight * norm for weight in weights]
    edges = np.cumsum(np.column_stack([[0.0] + [gap * cell] * (count - 1), heights]).flat)
    tops, bottoms = (top - edges).reshape((-1, 2)).T

    return [(left, float(low), right, float(high)) for high, low in zip(tops, bottoms, strict=True)]


def split_for_bar(cell: tuple[float, float, float, float],
                  bar: ColourBar) -> tuple[tuple[float, float, float, float],
                                           tuple[float, float, float, float]]:
    """A panel's cell divided between its plot and its colour bar."""
    left, bottom, right, top = cell
    space = 2 * bar.pad / (1 - bar.pad)
    ratios = np.array([1 - bar.fraction - bar.pad, bar.fraction])
    each = (right - left) / (2 + space)
    widths = ratios * each * 2 / ratios.sum()
    edges = np.cumsum(np.column_stack([[0.0, space * each], widths]).flat)
    lefts, rights = (left + edges).reshape((-1, 2)).T

    return ((float(lefts[0]), bottom, float(rights[0]), top),
            (float(lefts[1]), bottom, float(rights[1]), top))


def shrink(cell: tuple[float, float, float, float],
           aspect: float,
           figure_aspect: float,
           anchor: tuple[float, float]) -> tuple[float, float, float, float]:
    """The largest box of *aspect* (height to width) in a cell, placed at
    *anchor* within it."""
    left, bottom, right, top = cell
    width, height = right - left, top - bottom
    tall = width * aspect / figure_aspect

    if tall <= height:
        wide = width
    else:
        wide, tall = height * figure_aspect / aspect, height

    x = left + anchor[0] * (width - wide)
    y = bottom + anchor[1] * (height - tall)

    return (x, y, x + wide, y + tall)


def boxes(panel: Panel,
          cell: tuple[float, float, float, float],
          state: tuple[AxisState, AxisState],
          figure: tuple[float, float]) -> tuple[Box, Box | None]:
    """A panel's plot box and its colour bar's, in inches."""
    width, height = figure
    plot, bar = cell, None

    if panel.colour_bar is not None:
        plot, bar = split_for_bar(cell, panel.colour_bar)
        bar = shrink(bar, panel.colour_bar.aspect, height / width, BAR_ANCHOR)

    if panel.equal:
        x, y = state
        ratio = abs(y.high - y.low) / abs(x.high - x.low)
        anchor = BESIDE_BAR if panel.colour_bar is not None else (0.5, 0.5)
        plot = shrink(plot, ratio, height / width, anchor)

    return _inches(plot, figure), (_inches(bar, figure) if bar else None)


def _inches(box: tuple[float, float, float, float],
            figure: tuple[float, float]) -> Box:
    left, bottom, right, top = box

    return Box(left * figure[0], bottom * figure[1],
               (right - left) * figure[0], (top - bottom) * figure[1])


def end(state: AxisState,
        length: float,
        sizes: Sizes,
        across: bool) -> bool:
    """End a measuring axis on its outermost ticks. Returns whether it was
    one."""
    spec = state.spec
    low, high = sorted((state.low, state.high))

    if not spec.measures or not spec.end or not high > low:
        return False

    if spec.kind == "date":
        low, high, ended = dateticks.ends(low, high)
    else:
        room = ticks.room_for(length, sizes.tick_label, across)
        (low, high), ended = ticks.ends(low, high, spec.spacing, room), True

    state.low, state.high = low, high
    state.ended = ended

    return ended


def axis_ticks(state: AxisState,
               length: float,
               sizes: Sizes,
               across: bool,
               blank: float | None = None) -> tuple[tuple[Tick, ...], str]:
    """An axis's ticks inside its range, with their labels, and the offset
    written at its end."""
    spec = state.spec
    low, high = sorted((state.low, state.high))

    if spec.kind in ("names", "fixed"):
        shown = [Tick(value, name) for value, name in zip(spec.ticks, spec.names, strict=True)]
        return tuple(tick for tick in shown if low - 1e-9 <= tick.value <= high + 1e-9), ""

    if spec.kind == "date":
        every = dateticks.tick_values(low, high)
        written = dateticks.labels(every, spec.inverted)
        labels, offset = written.labels, written.offset
    else:
        every = ticks.tick_values(low, high, spec.spacing,
                                  ticks.room_for(length, sizes.tick_label, across))
        labels, offset = _numbers(every, spec, low, high)

    inside = set(ticks.within(every, low, high))
    found = []

    for value, label in zip(every, labels, strict=True):
        if value in inside:
            hidden = blank is not None and math.isclose(value, blank, rel_tol=1e-9, abs_tol=1e-12)
            found.append(Tick(value, "" if hidden else label))

    return tuple(found), offset


def _numbers(every: list[float],
             spec: Axis,
             low: float,
             high: float) -> tuple[tuple[str, ...], str]:
    if spec.labels == "percent":
        return ticks.percent_labels(every, low, high).labels, ""

    if callable(spec.labels):
        return tuple(str(spec.labels(value)) for value in every), ""

    written = ticks.number_labels(every, low, high)

    return written.labels, written.offset


def at(value: float,
       state: AxisState,
       start: float,
       length: float) -> float:
    """Where a value falls along an axis drawn from *start* for *length*
    inches."""
    low, high = state.low, state.high
    if state.spec.inverted:
        low, high = high, low

    return start + (value - low) / (high - low) * length


def label_span(text: str,
               size: float,
               font: Font,
               rotation: float) -> tuple[float, float]:
    """How wide and how tall a label's box is once turned, in inches."""
    shape = metrics.box(text, size, font)
    turn = math.radians(rotation)
    wide = abs(shape.width * math.cos(turn)) + abs(shape.height * math.sin(turn))
    tall = abs(shape.width * math.sin(turn)) + abs(shape.height * math.cos(turn))

    return wide, tall


def across_from(x: float,
                wide: float,
                ha: str) -> tuple[float, float]:
    """A box's left and right edges when its *ha* side sits at x."""
    if ha == "right":
        return x - wide, x
    if ha == "center":
        return x - wide / 2.0, x + wide / 2.0

    return x, x + wide


def shows(label: Label,
          box: Box,
          state: tuple[AxisState, AxisState]) -> bool:
    """Whether a label is drawn: one clipped to its panel is not when its
    point falls outside it."""
    if not label.clip or label.fraction:
        return True

    x = at(label.x, state[0], box.left, box.width)
    y = at(label.y, state[1], box.bottom, box.height)
    slack = 1e-9

    return (box.left - slack <= x <= box.right + slack
            and box.bottom - slack <= y <= box.top + slack)


def label_reach(label: Label,
                box: Box,
                state: tuple[AxisState, AxisState]) -> tuple[float, float] | None:
    """How far a label written on a panel reaches, left and right; None when
    it is not drawn."""
    if not shows(label, box, state):
        return None

    if label.fraction:
        x = box.left + label.x * box.width
    else:
        x = at(label.x, state[0], box.left, box.width)

    wide, _ = label_span(label.text, label.size, label.font, label.rotation)

    return across_from(x + label.dx / 72.0, wide, label.ha)


def tick_label_reach(axis: AxisLayout,
                     state: AxisState,
                     box: Box,
                     sizes: Sizes,
                     ticks_on: bool) -> list[tuple[float, float]]:
    """How far a horizontal axis's tick labels reach, left and right."""
    if not axis.show_labels:
        return []

    reach = []
    for tick in axis.ticks:
        if not tick.label:
            continue
        x = at(tick.value, state, box.left, box.width)
        wide, _ = label_span(tick.label, sizes.tick_label, sizes.font, axis.rotation)
        reach.append(across_from(x, wide, axis.align))

    return reach


def y_label_left(axis: AxisLayout,
                 box: Box,
                 sizes: Sizes,
                 pad: float) -> float:
    """Where a vertical axis's tick labels start, at the left: the axis line
    where there are none."""
    widths = [metrics.width(tick.label, sizes.tick_label, sizes.font)
              for tick in axis.ticks if tick.label] if axis.show_labels else []

    return box.left - pad / 72.0 - max(widths) if widths else box.left


def y_label_right(axis: AxisLayout,
                  box: Box,
                  sizes: Sizes,
                  pad: float) -> float:
    """Where a right-hand axis's tick labels end, at the right: the axis line
    where there are none."""
    widths = [metrics.width(tick.label, sizes.tick_label, sizes.font)
              for tick in axis.ticks if tick.label] if axis.show_labels else []

    return box.right + pad / 72.0 + max(widths) if widths else box.right


def x_label_bottom(axis: AxisLayout,
                   box: Box,
                   sizes: Sizes,
                   pad: float) -> float:
    """Where a horizontal axis's tick labels end, at the bottom: the axis
    line where there are none."""
    talls = [label_span(tick.label, sizes.tick_label, sizes.font, axis.rotation)[1]
             for tick in axis.ticks if tick.label] if axis.show_labels else []

    return box.bottom - pad / 72.0 - max(talls) if talls else box.bottom


def y_title(text: str,
            right: float,
            box: Box,
            sizes: Sizes) -> Text:
    """A vertical axis's title, its right edge at *right*, centred up the
    panel, reading upward."""
    return Text(text, right, box.bottom + box.height / 2.0, sizes.axis_title, sizes.font,
                "text-primary", ha="center", va="bottom", rotation=90.0,
                id="beacon-axis-title")


def y2_title(text: str,
             left: float,
             box: Box,
             sizes: Sizes) -> Text:
    """A right-hand axis's title, its left edge at *left*, centred up the
    panel, reading upward."""
    return Text(text, left, box.bottom + box.height / 2.0, sizes.axis_title, sizes.font,
                "text-primary", ha="center", va="top", rotation=90.0,
                id="beacon-axis-title")


def x_title(text: str,
            top: float,
            box: Box,
            sizes: Sizes) -> Text:
    """A horizontal axis's title, its top at *top*, centred under the
    panel."""
    return Text(text, box.left + box.width / 2.0, top, sizes.axis_title, sizes.font,
                "text-primary", ha="center", va="top", id="beacon-axis-title")
