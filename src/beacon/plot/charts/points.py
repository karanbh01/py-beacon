# src/beacon/plot/charts/points.py
"""
One measure against another, and how a measure is spread: `Scatter` and
`Histogram`.
"""
# BN-299.
import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .. import style as sizes
from ..colours import palette
from ..design import StyleArgument
from ..layout import Axis, Description, LabelStyle, LegendItem, Panel
from ..marks import BarMark, LineMark, Mark, PointMark, RuleMark, TextMark
from .base import Chart, Plan
from .data import Data, Limits, extent, frame_of, limited, percent, plain
from .data import values as floats
from .lines import labels_of

# A point's size, in points before scaling: larger where there are few to
# read one by one, smaller and fainter where there are many.
POINT = 4.0
MANY_POINTS = 2.8
MANY = 60
MANY_ALPHA = 0.55

# How far a point's name sits from it, and a quadrant's name from the plot's
# corner, in points before scaling.
NAME_GAP = 3.0

# Where the fitted line's summary is written, as fractions of the plot.
SUMMARY_AT = (0.02, 0.97)

# A histogram's bars as a share of each bin, and its default number of bins.
BIN_FILL = 0.86
BINS = 50

# Which axes a percentage applies to.
Percent = bool | Literal["x", "y"]


def _formats(choice: Percent) -> tuple[LabelStyle, LabelStyle]:
    """How each axis is written: (x, y)."""
    x: LabelStyle = "share" if choice in (True, "x") else "number"
    y: LabelStyle = "share" if choice in (True, "y") else "number"

    return x, y


class Scatter(Chart):
    """Points, one measure against another.

    Args:
        data: A DataFrame with a column for each measure, or None when x and
            y are given as series.
        x: The horizontal measure: a column's name, or a series.
        y: The vertical measure: a column's name, or a series.
        labels: Names written beside the points: a column's name, or a
            series.
        name: What the points are, in the legend when there is one.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        xlabel: The horizontal axis's title; the x measure's name by default.
        ylabel: The vertical axis's title; the y measure's name by default.
        fit: "linear" draws a fitted line and writes its slope (beta) and
            fit (R squared) on the chart.
        curve: A curve through the same axes, such as a frontier: a series
            of y indexed by x.
        curve_label: The curve's name in the legend.
        axes: "ends" puts the axes at the plot's edges; "origin" draws them
            through zero, for a chart read as four quadrants.
        quadrants: Names for the quadrants, from the top right anticlockwise.
        signed: Whether points are coloured by whether x and y share a sign.
        percent: Whether values are shares, written as percentages: True for
            both axes, or "x" or "y" for one.
        xlim: The horizontal axis's (low, high); either may be None.
        ylim: The vertical axis's (low, high).
        color: The points' colour.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "scatter"

    def __init__(self,
                 data: pd.DataFrame | None = None,
                 x: str | pd.Series | None = None,
                 y: str | pd.Series | None = None,
                 labels: str | pd.Series | None = None,
                 name: str = "Points",
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 xlabel: str | None = None,
                 ylabel: str | None = None,
                 fit: Literal["linear"] | None = None,
                 curve: pd.Series | None = None,
                 curve_label: str | None = None,
                 axes: Literal["ends", "origin"] = "ends",
                 quadrants: Sequence[str] | None = None,
                 signed: bool = False,
                 percent: Percent = False,
                 xlim: Limits = None,
                 ylim: Limits = None,
                 color: str | None = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.x = _measure(data, x, "x")
        self.y = _measure(data, y, "y")
        self.names = _measure(data, labels, "labels") if labels is not None else None
        self.name = name
        self.xlabel = xlabel if xlabel is not None else str(self.x.name or "")
        self.ylabel = ylabel if ylabel is not None else str(self.y.name or "")
        self.fit = fit
        self.curve = curve
        self.curve_label = curve_label
        self.origin = axes == "origin"
        self.quadrants = list(quadrants) if quadrants is not None else []
        self.signed = signed
        self.percent = percent
        self.xlim = xlim
        self.ylim = ylim
        self.color = color

        if axes not in ("ends", "origin"):
            raise ValueError(f"axes must be 'ends' or 'origin', not {axes!r}")
        if fit not in (None, "linear"):
            raise ValueError(f"fit must be 'linear' or None, not {fit!r}")

    def _build(self) -> Plan:
        x, y = floats(self.x), floats(self.y)
        many = len(x) > MANY
        point_colour = self.color or ("text-secondary" if self.curve is not None
                                      else palette(0))
        marks: list[Mark] = []
        legend: list[LegendItem] = []
        xs, ys = list(x), list(y)

        if self.curve is not None:
            curve_x = tuple(float(value) for value in self.curve.index)
            curve_y = floats(self.curve)
            marks.append(LineMark(curve_x, curve_y, palette(0), sizes.series_width(), layer=3.0))
            xs.extend(curve_x)
            ys.extend(curve_y)
            legend.append(LegendItem(self.curve_label or str(self.curve.name or "Curve"),
                                     palette(0)))

        marks.extend(self._points(x, y, point_colour, many))
        if self.curve is not None or self.fit:
            legend.insert(0, LegendItem(self.name, point_colour, "marker"))

        if self.fit == "linear":
            marks.extend(self._fitted(x, y))
            legend.append(LegendItem("Fitted line", palette(1)))

        marks.extend(self._names(x, y))
        x_low, x_high, y_low, y_high = self._limits(xs, ys)
        marks.extend(self._quadrants())
        x_format, y_format = _formats(self.percent)

        if self.origin:
            marks.extend(self._inline_titles())

        panel = Panel(x=Axis("number", x_low, x_high, labels=x_format,
                             title="" if self.origin else self.xlabel),
                      y=Axis("number", y_low, y_high, labels=y_format,
                             title="" if self.origin else self.ylabel),
                      labels=labels_of(marks, self.style.text.font), origin=self.origin)
        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=(panel,), legend=tuple(legend),
                                  bottom_room=_room(self.origin, bool(self.xlabel)))

        return Plan(description, tuple(marks))

    def _points(self,
                x: tuple[float, ...],
                y: tuple[float, ...],
                colour: str,
                many: bool) -> list[Mark]:
        size = sizes.scaled(MANY_POINTS if many else POINT)
        alpha = MANY_ALPHA if many else 1.0

        names = (tuple(str(name) for name in self.names) if self.names is not None
                 else ("",) * len(x))
        measures = (self.xlabel, self.ylabel)

        if not self.signed:
            return [PointMark(x, y, colour, size, alpha=alpha, names=names, name=self.name,
                              measures=measures)]

        agree = [a * b > 0 for a, b in zip(x, y, strict=True)]
        marks: list[Mark] = []
        for keep, ink in ((True, "success"), (False, "danger")):
            chosen = [(a, b, name) for a, b, name, same in zip(x, y, names, agree, strict=True)
                      if same == keep]
            if chosen:
                marks.append(PointMark(tuple(a for a, _, _ in chosen),
                                       tuple(b for _, b, _ in chosen), ink, size, alpha=alpha,
                                       names=tuple(name for _, _, name in chosen),
                                       name=self.name, measures=measures))

        return marks

    def _fitted(self,
                x: tuple[float, ...],
                y: tuple[float, ...]) -> list[Mark]:
        """A least-squares line, with its slope and fit written on the
        chart."""
        pairs = [(a, b) for a, b in zip(x, y, strict=True) if math.isfinite(a) and math.isfinite(b)]
        xs, ys = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
        slope, intercept = np.polyfit(xs, ys, 1)
        r2 = float(np.corrcoef(xs, ys)[0, 1] ** 2)
        low, high = float(xs.min()), float(xs.max())

        return [LineMark((low, high), (intercept + slope * low, intercept + slope * high),
                         palette(1), sizes.secondary_width(), layer=4.0),
                TextMark(f"\N{GREEK SMALL LETTER BETA} {slope:.2f}, R\N{SUPERSCRIPT TWO} {r2:.2f}",
                         SUMMARY_AT[0], SUMMARY_AT[1], "text-primary", sizes.label_size(),
                         weight=400, ha="left", va="top", coords="axes")]

    def _names(self,
               x: tuple[float, ...],
               y: tuple[float, ...]) -> list[Mark]:
        if self.names is None:
            return []

        return [TextMark(str(name), a, b, "text-secondary", sizes.label_size(),
                         dx=sizes.scaled(NAME_GAP), ha="left", va="center")
                for name, a, b in zip(self.names, x, y, strict=True)
                if math.isfinite(a) and math.isfinite(b)]

    def _limits(self,
                xs: list[float],
                ys: list[float]) -> tuple[float, float, float, float]:
        """The axes' ranges: through zero and even about it when the axes
        cross there."""
        x_low, x_high = extent(xs)
        y_low, y_high = extent(ys)

        if self.origin:
            x_far, y_far = max(abs(x_low), abs(x_high)), max(abs(y_low), abs(y_high))
            x_low, x_high, y_low, y_high = -x_far, x_far, -y_far, y_far

        return (*limited(x_low, x_high, self.xlim), *limited(y_low, y_high, self.ylim))

    def _quadrants(self) -> list[Mark]:
        corners: tuple[tuple[tuple[float, float], Literal["left", "right"],
                             Literal["top", "bottom"]], ...] = (
            ((0.98, 0.98), "right", "top"), ((0.02, 0.98), "left", "top"),
            ((0.02, 0.02), "left", "bottom"), ((0.98, 0.02), "right", "bottom"))

        return [TextMark(name, x, y, "text-muted", sizes.label_size(), ha=ha, va=va,
                         coords="axes")
                for name, ((x, y), ha, va) in zip(self.quadrants, corners, strict=False)]

    def _inline_titles(self) -> list[Mark]:
        """Axis titles along the axes through zero: over the horizontal one's
        right end and beside the vertical one's top, each clear of the tick
        marks, which point into the plot."""
        clear = self._clear_of_ticks()
        size = sizes.axis_title_size()

        return [TextMark(self.xlabel, 1.0, 0.0, "text-primary", size, dy=clear, ha="right",
                         va="bottom", coords="ydata"),
                TextMark(self.ylabel, 0.0, 1.0, "text-primary", size, dx=clear, ha="left",
                         va="top", coords="xdata")]

    def _clear_of_ticks(self) -> float:
        """How far an axis title sits from its axis, in points: past the tick
        marks and a gap."""
        ticks = sizes.scaled(self.style.axes.tick_length)

        return ticks + sizes.scaled(NAME_GAP)


def _measure(data: pd.DataFrame | None,
             chosen: str | pd.Series | None,
             what: str) -> pd.Series:
    """One of a scatter's measures, from a column's name or a series."""
    if chosen is None:
        raise ValueError(f"a scatter needs {what}=: a column's name or a series")

    if isinstance(chosen, pd.Series):
        return chosen

    if data is None or chosen not in data.columns:
        raise ValueError(f"{what}={chosen!r} names no column of the data")

    found: pd.Series = data[chosen]

    return found


def _room(origin: bool,
          titled: bool) -> float | None:
    """Room under the plot: one row of tick labels, and a row for the
    horizontal axis's title when it has one under it."""
    if origin or not titled:
        return None

    return sizes.scaled(0.3) + sizes.axis_title_size() * 1.6 / 72.0


class Histogram(Chart):
    """How a measure is spread: counts in equal bins.

    Args:
        data: The values, as a Series.
        bins: How many equal bins.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        xlabel: The horizontal axis's title; the series' name by default.
        ylabel: The vertical axis's title.
        normal: Whether to draw the normal curve with the values' mean and
            standard deviation.
        mark: Values marked with a line and named, such as {"95% VaR": -0.02}.
        percent: Whether values are shares, written as percentages.
        color: The bars' colour.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "histogram"

    def __init__(self,
                 data: Data,
                 bins: int = BINS,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 xlabel: str | None = None,
                 ylabel: str = "Count",
                 normal: bool = False,
                 mark: Mapping[str, float] | None = None,
                 percent: bool = False,
                 color: str | None = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.data = frame_of(data).iloc[:, 0].dropna()
        self.bins = bins
        self.xlabel = xlabel if xlabel is not None else str(self.data.name or "")
        self.ylabel = ylabel
        self.normal = normal
        self.marked = dict(mark or {})
        self.percent = percent
        self.color = color

    def _bound(self, value: float) -> str:
        """A bin's edge, as the popup writes it."""
        return percent(value, 2) if self.percent else plain(value, 2)

    def _build(self) -> Plan:
        values = self.data.to_numpy(dtype=float)
        counts, edges = np.histogram(values, bins=self.bins)
        centres = (edges[:-1] + edges[1:]) / 2.0
        width = float(edges[1] - edges[0])
        colour = self.color or palette(0)
        bins = tuple(f"{self._bound(float(low))} to {self._bound(float(high))}"
                     for low, high in itertools.pairwise(edges))
        marks: list[Mark] = [BarMark(tuple(float(at) for at in centres),
                                     tuple(float(count) for count in counts),
                                     (width * BIN_FILL,) * len(counts), (colour,) * len(counts),
                                     name="Count", labels=bins)]
        legend: list[LegendItem] = []
        tops = [0.0, float(counts.max())]

        if self.normal:
            curve_x, curve_y = _normal(values, float(edges[0]), float(edges[-1]), width)
            marks.append(LineMark(curve_x, curve_y, "text-secondary", sizes.secondary_width(),
                                  layer=4.0))
            tops.append(max(curve_y))
            legend = [LegendItem(self.xlabel or "Values", colour, "patch"),
                      LegendItem("Normal, same mean and deviation", "text-secondary")]

        for name, value in self.marked.items():
            words = f"{name} {percent(value, 2) if self.percent else plain(value, 2)}"
            marks.append(RuleMark(float(value), "danger", sizes.scaled(0.8), vertical=True,
                                  dashed=True, layer=4.0))
            marks.append(TextMark(words, float(value), 1.0, "danger", sizes.label_size(),
                                  dx=-sizes.scaled(NAME_GAP), dy=-sizes.scaled(NAME_GAP),
                                  ha="right", va="top", coords="xdata"))

        x_format: LabelStyle = "share" if self.percent else "number"
        low, high = extent(edges.tolist(), list(self.marked.values()))
        panel = Panel(x=Axis("number", low, high, labels=x_format, title=self.xlabel),
                      y=Axis("number", *extent(tops), title=self.ylabel),
                      labels=labels_of(marks, self.style.text.font))
        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=(panel,), legend=tuple(legend),
                                  bottom_room=_room(False, bool(self.xlabel)))

        return Plan(description, tuple(marks))


def _normal(values: np.ndarray,
            low: float,
            high: float,
            width: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The normal curve with the values' mean and deviation, scaled to the
    bins' counts."""
    mean, deviation = float(values.mean()), float(values.std())
    grid = np.linspace(low, high, 161)

    if deviation == 0:
        return tuple(grid.tolist()), (0.0,) * len(grid)

    density = (len(values) * width * np.exp(-0.5 * ((grid - mean) / deviation) ** 2)
               / (deviation * math.sqrt(2.0 * math.pi)))

    return tuple(grid.tolist()), tuple(density.tolist())
