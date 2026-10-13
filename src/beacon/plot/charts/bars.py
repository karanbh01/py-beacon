# src/beacon/plot/charts/bars.py
"""
Values by category as bars: `Column` (vertical), `Bar` (horizontal), and
`Combo`, columns on the left axis with a line on the right.

Several series are grouped by category. A single series takes the signed
colours, gains in green and losses in red, unless `signed=False`, when every
bar takes one colour.
"""
# BN-299.
import dataclasses
import itertools
from collections.abc import Sequence

import pandas as pd

from .. import style as sizes
from ..colours import palette
from ..design import StyleArgument
from ..layout import Axis, Description, LabelStyle, LegendItem, Panel
from ..marks import BarMark, LineMark, Mark, RuleMark, TextMark
from ..themes import GREY_4
from .base import Chart, Plan
from .data import Data, Limits, extent, frame_of, is_dates, limited, named, percent, plain
from .data import positions as numbers
from .data import values as floats
from .lines import labels_of

# The space between categories, as a share of each category's slot, and
# between the columns in one category, as a share of a column.
GAP = 0.32
INNER_GAP = 0.08

# How far a value's label sits from its bar's end, in points before scaling.
VALUE_GAP = 3.0

# The room kept beyond the longest bar for its value's label, as a share of
# the axis's span.
LABEL_ROOM = 0.08

# A cap's line weight, before scaling.
CAP_WIDTH = 0.8

# The colour bars take when they are not signed.
UNSIGNED = palette(GREY_4)


class _Bars(Chart):
    """What columns and bars share: categories, grouping, signs, value
    labels and a cap."""

    horizontal = False

    def __init__(self,
                 data: Data,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 xlabel: str = "",
                 ylabel: str = "",
                 percent: bool = False,
                 decimals: int = 1,
                 gap: float = GAP,
                 inner_gap: float = INNER_GAP,
                 value_labels: bool = False,
                 cap: float | None = None,
                 signed: bool | None = None,
                 color: str | None = None,
                 up_color: str | None = None,
                 down_color: str | None = None,
                 colors: Sequence[str] | None = None,
                 lim: Limits = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.data = frame_of(data)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.percent = percent
        self.decimals = decimals
        self.gap = gap
        self.inner_gap = inner_gap
        self.value_labels = value_labels
        self.cap = cap
        self.signed = signed if signed is not None else len(self.data.columns) == 1
        self.color = color
        self.up_color = up_color
        self.down_color = down_color
        self.colors = list(colors) if colors is not None else None
        self.lim = lim

    def _value(self, value: float) -> str:
        return (percent(value, self.decimals) if self.percent
                else plain(value, self.decimals))

    def _colours(self,
                 index: int,
                 column: tuple[float, ...]) -> tuple[str, ...]:
        """One series' bar colours: by sign, or the series' own."""
        if self.signed and len(self.data.columns) == 1:
            up, down = self.up_color or "success", self.down_color or "danger"
            return tuple(up if value >= 0 else down for value in column)

        if self.colors is not None and index < len(self.colors):
            return (self.colors[index],) * len(column)
        if len(self.data.columns) == 1:
            return (self.color or UNSIGNED,) * len(column)

        return (palette(index),) * len(column)

    def _build(self) -> Plan:
        names = [str(name) for name in self.data.index]
        count, series = len(names), named(self.data.columns)
        slot = (1.0 - self.gap) / len(series)
        marks: list[Mark] = []
        everything: list[float] = [0.0]

        if any(value < 0 for name in series for value in floats(self.data[name])) \
                and sizes.reference_lines():
            marks.append(RuleMark(0.0, "border", sizes.reference_width(),
                                  vertical=self.horizontal))

        for index, name in enumerate(series):
            column = tuple(0.0 if value != value else value for value in floats(self.data[name]))
            offset = (index - (len(series) - 1) / 2.0) * slot
            places = tuple(self._place(at, count) + offset for at in range(count))
            marks.append(BarMark(places, column, (slot * (1.0 - self.inner_gap),) * count,
                                 self._colours(index, column), horizontal=self.horizontal))
            everything.extend(column)
            if self.value_labels:
                marks.extend(self._label(place, value) for place, value in zip(places, column,
                                                                               strict=True))

        if self.cap is not None:
            marks.extend(self._cap(count))
            everything.append(self.cap)

        if self.value_labels:
            everything.extend(_room(everything))

        return Plan(self._description(names, everything, marks, series), tuple(marks))

    def _place(self,
               at: int,
               count: int) -> float:
        """Where a category sits: left to right, or top to bottom."""
        return float(count - 1 - at) if self.horizontal else float(at)

    def _label(self,
               place: float,
               value: float) -> TextMark:
        """A value written at its bar's end, beyond it."""
        gap = sizes.scaled(VALUE_GAP) * (1 if value >= 0 else -1)

        if self.horizontal:
            return TextMark(self._value(value), value, place, "text-primary", sizes.label_size(),
                            dx=gap, ha="left" if value >= 0 else "right", va="center")

        return TextMark(self._value(value), place, value, "text-primary", sizes.label_size(),
                        dy=gap, ha="center", va="bottom" if value >= 0 else "top")

    def _cap(self, count: int) -> list[Mark]:
        """A marked limit: a dashed line, named at its end."""
        assert self.cap is not None
        words = f"cap {self._value(self.cap)}"
        rule = RuleMark(self.cap, "danger", sizes.scaled(CAP_WIDTH), vertical=self.horizontal,
                        dashed=True, layer=4.0)

        if self.horizontal:
            return [rule, TextMark(words, self.cap, count - 0.5, "danger", sizes.label_size(),
                                   dy=sizes.scaled(1.5), ha="center", va="bottom")]

        return [rule, TextMark(words, count - 0.5, self.cap, "danger", sizes.label_size(),
                               dy=sizes.scaled(1.5), ha="right", va="bottom")]

    def _description(self,
                     names: list[str],
                     everything: list[float],
                     marks: list[Mark],
                     series: list[str]) -> Description:
        count = len(names)
        places = tuple(self._place(at, count) for at in range(count))
        labels: LabelStyle = "percent" if self.percent else "number"
        low, high = limited(*extent(everything), self.lim)
        categories = Axis("names", -0.5, count - 0.5, ticks=places, names=tuple(names))

        if self.horizontal:
            panel = Panel(x=Axis("number", low, high, labels=labels, title=self.xlabel),
                          y=categories, labels=labels_of(marks, self.style.text.font))
        else:
            panel = Panel(x=_titled(categories, self.xlabel),
                          y=Axis("number", low, high, labels=labels, title=self.ylabel),
                          labels=labels_of(marks, self.style.text.font))

        legend = (tuple(LegendItem(name, self._colours(index, (1.0,))[0], "patch")
                        for index, name in enumerate(series)) if len(series) > 1 else ())

        return Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                           notes=self._notes(), panels=(panel,), legend=legend)


def _room(values: list[float]) -> list[float]:
    """Values just beyond the longest bars, so their labels have room: on
    each side where there are bars."""
    low, high = min(values), max(values)
    room = (high - low) * LABEL_ROOM

    return [high + room if high > 0 else high, low - room if low < 0 else low]


def _titled(axis: Axis,
            title: str) -> Axis:
    return dataclasses.replace(axis, title=title)


class Column(_Bars):
    """Vertical bars, one per category, grouped where there are several
    series.

    Args:
        data: A Series, or a DataFrame of one series per column; its index
            names the categories, left to right.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        ylabel: The vertical axis's title.
        xlabel: The categories' title.
        percent: Whether values are shares, written as percentages.
        decimals: Decimals in the value labels.
        gap: The space between categories, as a share of each category's
            slot; 0 closes it.
        inner_gap: The space between the columns in one category, as a
            share of a column; 0 closes it.
        value_labels: Whether each column's value is written at its end.
        cap: A marked limit, drawn as a dashed line across.
        signed: Whether a single series is coloured by sign; by default it
            is, and several series take their own colours.
        color: The colour of unsigned columns; grey by default.
        up_color: Gains' colour, for this chart alone.
        down_color: Losses' colour, for this chart alone.
        colors: Each series' colour, in order.
        ylim: The vertical axis's (low, high); either may be None.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "column"

    def __init__(self,
                 data: Data,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 ylabel: str = "",
                 xlabel: str = "",
                 percent: bool = False,
                 decimals: int = 1,
                 gap: float = GAP,
                 inner_gap: float = INNER_GAP,
                 value_labels: bool = False,
                 cap: float | None = None,
                 signed: bool | None = None,
                 color: str | None = None,
                 up_color: str | None = None,
                 down_color: str | None = None,
                 colors: Sequence[str] | None = None,
                 ylim: Limits = None,
                 style: StyleArgument = None):
        super().__init__(data, title, subtitle, notes, source, xlabel, ylabel, percent,
                         decimals, gap, inner_gap, value_labels, cap, signed, color, up_color,
                         down_color, colors, ylim, style)


class Bar(_Bars):
    """Horizontal bars, one per category from the top down, so long names
    read across; grouped where there are several series.

    Args:
        data: A Series, or a DataFrame of one series per column; its index
            names the categories, top to bottom.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        xlabel: The horizontal axis's title.
        ylabel: The categories' title.
        percent: Whether values are shares, written as percentages.
        decimals: Decimals in the value labels.
        gap: The space between categories, as a share of each category's
            slot; 0 closes it.
        inner_gap: The space between the bars in one category, as a share of
            a bar; 0 closes it.
        value_labels: Whether each bar's value is written at its end.
        cap: A marked limit, drawn as a dashed line down.
        signed: Whether a single series is coloured by sign; by default it
            is, and several series take their own colours.
        color: The colour of unsigned bars; grey by default.
        up_color: Gains' colour, for this chart alone.
        down_color: Losses' colour, for this chart alone.
        colors: Each series' colour, in order.
        xlim: The horizontal axis's (low, high); either may be None.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "bar"
    horizontal = True

    def __init__(self,
                 data: Data,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 xlabel: str = "",
                 ylabel: str = "",
                 percent: bool = False,
                 decimals: int = 1,
                 gap: float = GAP,
                 inner_gap: float = INNER_GAP,
                 value_labels: bool = False,
                 cap: float | None = None,
                 signed: bool | None = None,
                 color: str | None = None,
                 up_color: str | None = None,
                 down_color: str | None = None,
                 colors: Sequence[str] | None = None,
                 xlim: Limits = None,
                 style: StyleArgument = None):
        super().__init__(data, title, subtitle, notes, source, xlabel, ylabel, percent,
                         decimals, gap, inner_gap, value_labels, cap, signed, color, up_color,
                         down_color, colors, xlim, style)


class Combo(Chart):
    """Columns on the left axis and a line on the right, for a flow and a
    stock measured on one calendar.

    On dates, each column is as wide as the step to the next date less the
    gap; with `gap=0` each runs to the next date, so daily columns meet edge
    to edge.

    Args:
        columns: The columns' series, indexed by dates or categories.
        line: The line's series, on the same index, measured on the right.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        ylabel: The left axis's title.
        ylabel_secondary: The right axis's title.
        xlabel: The horizontal axis's title.
        gap: The space between columns, as a share of each one's slot.
        signed: Whether the columns are coloured by sign; they are by
            default.
        color: The colour of unsigned columns; grey by default.
        up_color: Gains' colour, for this chart alone.
        down_color: Losses' colour, for this chart alone.
        line_color: The line's colour; the accent by default.
        ylim: The left axis's (low, high); either may be None.
        ylim_secondary: The right axis's (low, high).
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "combo"

    def __init__(self,
                 columns: pd.Series,
                 line: pd.Series,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 ylabel: str = "",
                 ylabel_secondary: str = "",
                 xlabel: str = "",
                 gap: float = GAP,
                 signed: bool = True,
                 color: str | None = None,
                 up_color: str | None = None,
                 down_color: str | None = None,
                 line_color: str | None = None,
                 ylim: Limits = None,
                 ylim_secondary: Limits = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        self.columns = frame_of(columns, "Columns").iloc[:, 0]
        self.line = frame_of(line, "Line").iloc[:, 0]
        self.ylabel = ylabel
        self.ylabel_secondary = ylabel_secondary
        self.xlabel = xlabel
        self.gap = gap
        self.signed = signed
        self.color = color
        self.up_color = up_color
        self.down_color = down_color
        self.line_color = line_color
        self.ylim = ylim
        self.ylim_secondary = ylim_secondary

    def _build(self) -> Plan:
        dates = is_dates(self.columns.index)
        heights = tuple(0.0 if value != value else value for value in floats(self.columns))
        up, down = self.up_color or "success", self.down_color or "danger"
        colours = (tuple(up if value >= 0 else down for value in heights) if self.signed
                   else (self.color or UNSIGNED,) * len(heights))
        marks: list[Mark] = []

        if any(value < 0 for value in heights) and sizes.reference_lines():
            marks.append(RuleMark(0.0, "border", sizes.reference_width()))

        if dates:
            places, widths, edge = self._dated()
            x = Axis("date", places[0], places[-1] + widths[-1], title=self.xlabel)
            line_x = numbers(self.line.index)
        else:
            count = len(heights)
            places = tuple(float(at) for at in range(count))
            widths, edge = (1.0 - self.gap,) * count, False
            x = Axis("names", -0.5, count - 0.5, ticks=places,
                     names=tuple(str(name) for name in self.columns.index), title=self.xlabel)
            line_x = tuple(float(self.columns.index.get_loc(when)) for when in self.line.index)

        line_colour = self.line_color or palette(0)
        marks.append(BarMark(places, heights, widths, colours, edge=edge,
                             seamless=dates and self.gap == 0))
        marks.append(LineMark(line_x, floats(self.line), line_colour, sizes.series_width(),
                              side="right"))

        y_low, y_high = limited(*extent(heights, [0.0]), self.ylim)
        r_low, r_high = limited(*extent(floats(self.line)), self.ylim_secondary)
        panel = Panel(x=x, y=Axis("number", y_low, y_high, title=self.ylabel),
                      y2=Axis("number", r_low, r_high, title=self.ylabel_secondary))
        legend = (LegendItem(str(self.columns.name or "Columns"), colours[0] if not self.signed
                             else up, "patch"),
                  LegendItem(f"{self.line.name or 'Line'} (right)", line_colour))
        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=(panel,), legend=legend)

        return Plan(description, tuple(marks), front="right")

    def _dated(self) -> tuple[tuple[float, ...], tuple[float, ...], bool]:
        """Columns on dates: where each starts and how wide it is. Each runs
        towards the next date, less the gap; the last as far as the one
        before it."""
        at = numbers(self.columns.index)
        steps = [after - before for before, after in itertools.pairwise(at)]
        steps.append(steps[-1] if steps else 1.0)
        widths = tuple(step * (1.0 - self.gap) for step in steps)

        if self.gap == 0:
            return at, widths, True

        starts = tuple(place + step * self.gap / 2.0
                       for place, step in zip(at, steps, strict=True))

        return starts, widths, True
