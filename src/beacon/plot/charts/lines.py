# src/beacon/plot/charts/lines.py
"""
Series over time or over a measure: `Line` and `Area`.
"""
# BN-299.
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .. import style as sizes
from ..colours import palette
from ..design import StyleArgument
from ..layout import Axis, Description, Label, LabelStyle, LegendItem, Panel
from ..marks import AreaMark, LineMark, Mark, PointMark, RuleMark, Side, SpanMark, TextMark
from ..metrics import Font
from .base import Chart, Plan
from .data import (
    Data,
    Limits,
    extent,
    frame_of,
    is_dates,
    limited,
    named,
    number,
    percent,
    plain,
    positions,
    values,
)

# The id on a chart's reference line, so a caller can find it.
REFERENCE_ID = "beacon-reference"

# How far the last value's label sits from its dot, and how big the dot is, in
# points before scaling.
LAST_GAP = 6.0
LAST_DOT = 3.0

# How far a marked period's label sits below the plot's top, in points before
# scaling.
PERIOD_LABEL_DROP = 3.0

# A filled series' fill, and a stacked area's.
FILL_ALPHA = 0.28
STACK_ALPHA = 0.85

# Marked periods: {label: (start, end)}, or a list of (start, end) or
# (start, end, label).
Periods = Mapping[str, tuple[Any, Any]] | Sequence[tuple[Any, ...]] | None


def _periods(periods: Periods) -> list[tuple[float, float, str]]:
    """Marked periods as (start, end, label), in date order."""
    if periods is None:
        return []

    if isinstance(periods, Mapping):
        found = [(number(start), number(end), str(label))
                 for label, (start, end) in periods.items()]
    else:
        found = [(number(period[0]), number(period[1]),
                  str(period[2]) if len(period) > 2 else "") for period in periods]

    return sorted((min(start, end), max(start, end), label) for start, end, label in found)


def _last(series: pd.Series) -> tuple[Any, float] | None:
    """A series' last value that is there, and where."""
    present = series.dropna()

    if present.empty:
        return None

    return present.index[-1], float(present.iloc[-1])


def _chosen(columns: list[str],
            choice: bool | str | Sequence[str]) -> list[str]:
    """The columns an option such as label_last= names."""
    if choice is True:
        return columns[:1]
    if not choice:
        return []
    if isinstance(choice, str):
        return [choice]

    return [str(name) for name in choice]


class Line(Chart):
    """One or more series as lines, the first in the accent colour and drawn
    heavier.

    Args:
        data: A Series, or a DataFrame of one series per column. Its index
            is the horizontal axis: dates, or numbers.
        secondary: Series measured on a second axis, on the right, drawn
            lighter.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        ylabel: The left axis's title.
        ylabel_secondary: The right axis's title.
        xlabel: The horizontal axis's title.
        reference: A value the chart is read against, such as 100, drawn as
            a line across it.
        label_last: Dot and label the last value: True for the first series,
            or the names of the series to label.
        periods: Marked periods, as boxes from the bottom of the plot to the
            top behind the series: {label: (start, end)}, or a list of
            (start, end) or (start, end, label).
        period_labels: Whether each period's label is written at the top of
            its box.
        period_legend: Names the periods once in the legend.
        xlim: The horizontal axis's (low, high); either may be None.
        ylim: The left axis's (low, high).
        ylim_secondary: The right axis's (low, high).
        invert: Whether the left axis runs from high at the bottom to low at
            the top.
        percent: Whether values are shares, written as percentages.
        colors: Each series' colour, by token name or hex, in order (the
            secondary series after the others); the palette by default.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "line"

    def __init__(self,
                 data: Data,
                 secondary: Data | None = None,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 ylabel: str = "",
                 ylabel_secondary: str = "",
                 xlabel: str = "",
                 reference: float | None = None,
                 label_last: bool | str | Sequence[str] = False,
                 periods: Periods = None,
                 period_labels: bool = True,
                 period_legend: str | None = None,
                 xlim: Limits = None,
                 ylim: Limits = None,
                 ylim_secondary: Limits = None,
                 invert: bool = False,
                 percent: bool = False,
                 colors: Sequence[str] | None = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.data = frame_of(data)
        self.secondary = frame_of(secondary, "Secondary") if secondary is not None else None
        self.ylabel = ylabel
        self.ylabel_secondary = ylabel_secondary
        self.xlabel = xlabel
        self.reference = reference
        self.label_last = label_last
        self.periods = _periods(periods)
        self.period_labels = period_labels
        self.period_legend = period_legend
        self.xlim = xlim
        self.ylim = ylim
        self.ylim_secondary = ylim_secondary
        self.invert = invert
        self.percent = percent
        self.colors = list(colors) if colors is not None else None

    def _colour(self, index: int) -> str:
        if self.colors is not None and index < len(self.colors):
            return self.colors[index]

        return palette(index)

    def _value(self, value: float) -> str:
        return percent(value) if self.percent else plain(value)

    def _build(self) -> Plan:
        dates = is_dates(self.data.index)
        x = positions(self.data.index)
        left = named(self.data.columns)
        right: list[str] = named(self.secondary.columns) if self.secondary is not None else []
        marks: list[Mark] = [*self._period_marks()]

        if self.reference is not None and sizes.reference_lines():
            marks.append(RuleMark(float(self.reference), "border", sizes.reference_width(),
                                  id=REFERENCE_ID))

        # The first series last, so it is drawn over the others.
        marks.extend(LineMark(x, values(self.data.iloc[:, index]), self._colour(index),
                              sizes.series_width() if index == 0 else sizes.secondary_width(),
                              layer=4.0 if index == 0 else 3.0)
                     for index in reversed(range(len(left))))

        if self.secondary is not None:
            x_right = positions(self.secondary.index)
            for index, _ in enumerate(right):
                marks.append(LineMark(x_right, values(self.secondary.iloc[:, index]),
                                      self._colour(len(left) + index), sizes.secondary_width(),
                                      side="right"))

        marks.extend(self._last_marks(left, right))

        return Plan(self._description(dates, x, left, right, marks), tuple(marks))

    def _period_marks(self) -> list[Mark]:
        marks: list[Mark] = [SpanMark(start, end, "period") for start, end, _ in self.periods]

        if self.period_labels:
            marks.extend(TextMark(label, (start + end) / 2.0, 1.0, "period-label",
                                  sizes.label_size(), dy=-sizes.scaled(PERIOD_LABEL_DROP),
                                  ha="center", va="top", coords="xdata")
                         for start, end, label in self.periods if label)

        return marks

    def _last_marks(self,
                    left: list[str],
                    right: list[str]) -> list[Mark]:
        marks: list[Mark] = []

        for name in _chosen(left + right, self.label_last):
            side: Side = "right" if name in right and name not in left else "left"
            frame = self.secondary if side == "right" else self.data
            assert frame is not None
            found = _last(frame[name] if name in frame.columns else frame.iloc[:, 0])
            if found is None:
                continue
            when, value = found
            at = number(when) if is_dates(frame.index) else float(when)
            colour = self._colour((left + right).index(name))
            # Unclipped: the last value sits on the axis's end, and clipped
            # there the dot would be cut in half.
            marks.append(PointMark((at,), (value,), colour, sizes.scaled(LAST_DOT),
                                   clip=False, side=side))
            if right:
                # The right axis's labels are beside the plot, so the value
                # is written inside it: above its dot on the left's scale,
                # below it on the right's, so the two stay apart.
                above = side == "left"
                marks.append(TextMark(self._value(value), at, value, colour,
                                      sizes.label_size(), weight=400,
                                      dx=-sizes.scaled(LAST_GAP / 2),
                                      dy=sizes.scaled(LAST_GAP / 2) * (1 if above else -1),
                                      ha="right", va="bottom" if above else "top",
                                      halo=True, side=side))
            else:
                marks.append(TextMark(self._value(value), at, value, colour, sizes.label_size(),
                                      weight=400, dx=sizes.scaled(LAST_GAP), ha="left",
                                      va="center", side=side))

        return marks

    def _description(self,
                     dates: bool,
                     x: tuple[float, ...],
                     left: list[str],
                     right: list[str],
                     marks: list[Mark]) -> Description:
        reference = [float(self.reference)] if self.reference is not None else []
        x_low, x_high = limited(*extent(x), self.xlim, dates)
        y_low, y_high = limited(*extent(*(values(self.data[name]) for name in self.data),
                                        reference), self.ylim)
        labels: LabelStyle = "percent" if self.percent else "number"
        y2 = None

        if self.secondary is not None:
            low, high = limited(*extent(*(values(self.secondary[name])
                                          for name in self.secondary)), self.ylim_secondary)
            y2 = Axis("number", low, high, labels=labels, title=self.ylabel_secondary)

        panel = Panel(x=Axis("date" if dates else "number", x_low, x_high, title=self.xlabel),
                      y=Axis("number", y_low, y_high, labels=labels, title=self.ylabel,
                             inverted=self.invert),
                      labels=labels_of(marks, self.style.text.font),
                      y2=y2)

        return Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                           notes=self._notes(), panels=(panel,),
                           legend=self._legend(left, right))

    def _legend(self,
                left: list[str],
                right: list[str]) -> tuple[LegendItem, ...]:
        items = []

        if len(left) + len(right) > 1:
            items = [LegendItem(name, self._colour(index)) for index, name in enumerate(left)]
            items += [LegendItem(f"{name} (right)" if left else name,
                                 self._colour(len(left) + index))
                      for index, name in enumerate(right)]

        if self.period_legend and self.periods:
            items.append(LegendItem(self.period_legend, "period", "period"))

        return tuple(items)


class Area(Chart):
    """Filled series: stacked parts of a whole, or one filled series such as
    a drawdown. The horizontal axis runs from the first value to the last,
    so the fill meets the axes with no gap.

    Args:
        data: A Series, or a DataFrame of one series per column, indexed by
            dates or numbers.
        stacked: Whether several series are stacked one on another; one
            series is filled from zero.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        ylabel: The vertical axis's title.
        xlabel: The horizontal axis's title.
        ylim: The vertical axis's (low, high); either may be None.
        percent: Whether values are shares, written as percentages.
        colors: Each series' colour, by token name or hex, in order.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "area"

    def __init__(self,
                 data: Data,
                 stacked: bool = True,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 ylabel: str = "",
                 xlabel: str = "",
                 ylim: Limits = None,
                 percent: bool = False,
                 colors: Sequence[str] | None = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.data = frame_of(data)
        self.stacked = stacked
        self.ylabel = ylabel
        self.xlabel = xlabel
        self.ylim = ylim
        self.percent = percent
        self.colors = list(colors) if colors is not None else None

    def _colour(self, index: int) -> str:
        if self.colors is not None and index < len(self.colors):
            return self.colors[index]

        return palette(index)

    def _build(self) -> Plan:
        dates = is_dates(self.data.index)
        x = positions(self.data.index)
        names = named(self.data.columns)
        filled = self.data.fillna(0.0)
        marks: list[Mark] = []
        lower = [0.0] * len(x)
        tops: list[float] = [0.0]

        if len(names) == 1 or not self.stacked:
            for index, name in enumerate(names):
                line = values(self.data[name])
                marks.append(AreaMark(x, tuple(lower), values(filled[name]),
                                      self._colour(index), alpha=FILL_ALPHA))
                marks.append(LineMark(x, line, self._colour(index), sizes.secondary_width()))
                tops.extend(line)
        else:
            for index, name in enumerate(names):
                upper = [low + value for low, value in zip(lower, values(filled[name]),
                                                           strict=True)]
                marks.append(AreaMark(x, tuple(lower), tuple(upper), self._colour(index),
                                      alpha=STACK_ALPHA))
                lower = upper
                tops.extend(upper)

        y_low, y_high = limited(*extent(tops), self.ylim)
        legend = (tuple(LegendItem(name, self._colour(index), "patch")
                        for index, name in enumerate(names)) if len(names) > 1 else ())
        panel = Panel(x=Axis("date" if dates else "number", *extent(x), title=self.xlabel,
                             end=False),
                      y=Axis("number", y_low, y_high,
                             labels="percent" if self.percent else "number", title=self.ylabel))
        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=(panel,), legend=legend)

        return Plan(description, tuple(marks))


def labels_of(marks: Sequence[Mark],
              font: str,
              panel: int = 0) -> tuple[Label, ...]:
    """The text a panel's marks write in its data, for the layout to keep
    inside the figure."""
    return tuple(Label(mark.text, mark.x, mark.y, mark.dx, mark.dy, mark.size,
                       Font(font, mark.weight), mark.ha, mark.va, clip=mark.clip)
                 for mark in marks
                 if isinstance(mark, TextMark) and mark.coords == "data" and mark.panel == panel
                 and math.isfinite(mark.y))
