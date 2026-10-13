# src/beacon/plot/interactive/traces.py
"""
A chart's marks as plotly draws them: lines, areas, bars, points and cells
as traces, the lines a chart is read against and its marked periods as
shapes, and its text as annotations, each on its panel's axes.

Each mark that is data also gets a readout for the popup: for every point,
what it is (a date, a category, a point's name), and a row per value with
the colour it is drawn in. Lines, areas, bars and points carry it on an
invisible trace of markers at their points, so the popup and its crosshair
snap to the data; the drawn traces themselves take no part in hovering.
"""
# BN-301.
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..colormaps import colour_at
from ..layout import Axis, LabelStyle, Layout, Panel
from ..marks import (
    AreaMark,
    BarMark,
    CellsMark,
    LineMark,
    Mark,
    PointMark,
    RuleMark,
    SpanMark,
    TextMark,
)
from ..metrics import Font
from .axes import Names
from .baseline import shift
from .readout import date_text, value_text
from .units import DAY_MS, PT, PX, Paint, rgba

# A dashed line's dash and gap, in multiples of its width.
DASH = (2.0, 1.6)


@dataclass
class Drawn:
    """What a chart's marks become: traces, shapes and annotations."""

    traces: list[dict[str, Any]]
    shapes: list[dict[str, Any]]
    annotations: list[dict[str, Any]]


class Marks:
    """Turns marks into plotly's traces, shapes and annotations."""

    def __init__(self,
                 panels: Sequence[Panel],
                 names: Sequence[Names],
                 paint: Paint,
                 layout: Layout,
                 legend: Sequence[str] = ()):
        self.panels = panels
        self.legend = list(legend)
        self.names = names
        self.paint = paint
        self.layout = layout
        self.drawn = Drawn([], [], [])
        self.order = 0

    def draw(self,
             marks: Sequence[Mark]) -> Drawn:
        """Every mark, by layer as matplotlib stacks them."""
        for mark in sorted(marks, key=lambda mark: getattr(mark, "layer", 0.5)):
            self.mark(mark)

        return self.drawn

    def mark(self, mark: Mark) -> None:
        if isinstance(mark, LineMark):
            self.line(mark)
        elif isinstance(mark, AreaMark):
            self.area(mark)
        elif isinstance(mark, BarMark):
            self.bars(mark)
        elif isinstance(mark, PointMark):
            self.points(mark)
        elif isinstance(mark, RuleMark):
            self.rule(mark)
        elif isinstance(mark, SpanMark):
            self.span(mark)
        elif isinstance(mark, TextMark):
            self.text(mark)
        elif isinstance(mark, CellsMark):
            self.cells(mark)

    def refs(self,
             panel: int,
             side: str = "left") -> tuple[str, str]:
        """A mark's axes. Marks on a second axis are drawn on the main one,
        their values carried onto its scale (lift), so plotly draws every
        mark in one subplot in the order the image does; on axes of its own
        a second axis's marks would all be drawn over the main one's."""
        names = self.names[panel]

        return names.x, names.y

    def lift(self,
             panel: int,
             side: str,
             values: Sequence[float]) -> list[float]:
        """Values on the second axis's scale carried onto the main axis's."""
        laid = self.layout.panels[panel]
        if side != "right" or laid.y2 is None:
            return list(values)

        main, second = laid.y, laid.y2
        ratio = (main.high - main.low) / ((second.high - second.low) or 1.0)

        return [main.low + (value - second.low) * ratio for value in values]

    def x_of(self,
             panel: int,
             values: Sequence[float]) -> list[float]:
        """Positions across, as the panel's axis counts them."""
        scale = DAY_MS if self.panels[panel].x.kind == "date" else 1.0

        return [value * scale for value in values]

    def line(self, mark: LineMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        width = mark.width * PT
        trace: dict[str, Any] = {
            "type": "scatter", "mode": "lines", "x": self.x_of(mark.panel, mark.x),
            "y": self.lift(mark.panel, mark.side, mark.y), "xaxis": x, "yaxis": y,
            "hoverinfo": "skip", "line": {"color": self.paint.ink(mark.colour), "width": width},
            "opacity": mark.alpha, "showlegend": False}
        if mark.dashed:
            trace["line"]["dash"] = f"{DASH[0] * width:.2f}px,{DASH[1] * width:.2f}px"
        self.drawn.traces.append(trace)

        if mark.name:
            rows = [[[self.paint.ink(mark.colour), mark.name, self.value(mark, value)]]
                    for value in mark.y]
            self.readout(mark.panel, mark.side, mark.x, mark.y, rows, "line", mark.name)

    def area(self, mark: AreaMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        across = self.x_of(mark.panel, mark.x)
        self.drawn.traces.append({
            "type": "scatter", "mode": "lines", "x": across + across[::-1],
            "y": self.lift(mark.panel, mark.side, [*mark.upper, *mark.lower[::-1]]),
            "fill": "toself",
            "fillcolor": rgba(self.paint.ink(mark.colour), mark.alpha),
            "line": {"width": 0}, "xaxis": x, "yaxis": y, "hoverinfo": "skip",
            "showlegend": False})

        if mark.name:
            rows = [[[self.paint.ink(mark.colour), mark.name,
                      self.value(mark, upper - lower)]]
                    for lower, upper in zip(mark.lower, mark.upper, strict=True)]
            self.readout(mark.panel, mark.side, mark.x, mark.upper, rows, "area", mark.name)

    def bars(self, mark: BarMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        colours = [self.paint.ink(name) for name in mark.colours]
        places = [place + (width / 2.0 if mark.edge else 0.0)
                  for place, width in zip(mark.positions, mark.thickness, strict=True)]
        dated = not mark.horizontal and self.panels[mark.panel].x.kind == "date"
        scale = DAY_MS if dated else 1.0
        trace: dict[str, Any] = {
            "type": "bar", "xaxis": x, "yaxis": y, "hoverinfo": "skip", "showlegend": False,
            "width": [width * scale for width in mark.thickness],
            "marker": {"color": colours, "line": {"width": 0}},
            "orientation": "h" if mark.horizontal else "v"}
        if mark.seamless:
            trace["marker"]["line"] = {"width": 0.25 * PT, "color": colours}
        if mark.base:
            trace["base"] = list(mark.base) if mark.horizontal else self.lift(
                mark.panel, mark.side, mark.base)
        positions = [place * scale for place in places]
        values = (list(mark.values) if mark.horizontal
                  else self.lift(mark.panel, mark.side, mark.values))
        trace.update({"x": values, "y": positions} if mark.horizontal
                     else {"x": positions, "y": values})
        self.drawn.traces.append(trace)

        if mark.name:
            ends = [value + (mark.base[index] if mark.base else 0.0)
                    for index, value in enumerate(mark.values)]
            rows = [[[colour, mark.name, self.value(mark, value)]]
                    for colour, value in zip(colours, mark.values, strict=True)]
            if mark.horizontal:
                self.readout(mark.panel, mark.side, ends, places, rows, "bar", mark.name,
                             labels=mark.labels, along="y")
            else:
                self.readout(mark.panel, mark.side, places, ends, rows, "bar", mark.name,
                             labels=mark.labels)

    def points(self, mark: PointMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        ink = self.paint.ink(mark.colour)
        self.drawn.traces.append({
            "type": "scatter", "mode": "markers", "x": self.x_of(mark.panel, mark.x),
            "y": self.lift(mark.panel, mark.side, mark.y), "xaxis": x, "yaxis": y,
            "hoverinfo": "skip", "showlegend": False, "cliponaxis": mark.clip,
            "opacity": mark.alpha,
            "marker": {"color": ink, "size": mark.size * PT, "line": {"width": 0}}})

        if mark.name:
            panel = self.panels[mark.panel]
            across, up = mark.measures
            rows = [[[ink, across or "x", value_text(a, panel.x.labels)],
                     [ink, up or "y", value_text(b, panel.y.labels)]]
                    for a, b in zip(mark.x, mark.y, strict=True)]
            labels = tuple(name or mark.name for name in mark.names) or (mark.name,) * len(mark.x)
            self.readout(mark.panel, mark.side, mark.x, mark.y, rows, "point", mark.name,
                         labels=labels)

    def value(self,
              mark: LineMark | AreaMark | BarMark,
              value: float) -> str:
        """A value as the axis it is read on writes it; a series of whole
        numbers, such as counts, without decimals."""
        values = mark.values if isinstance(mark, BarMark) else (
            mark.y if isinstance(mark, LineMark) else mark.upper)
        whole = all(float(item).is_integer() for item in values)

        return value_text(value, self.reading(mark), whole)

    def reading(self, mark: LineMark | AreaMark | BarMark) -> LabelStyle:
        panel = self.panels[mark.panel]
        if isinstance(mark, BarMark) and mark.horizontal:
            return panel.x.labels
        if mark.side == "right" and panel.y2 is not None:
            return panel.y2.labels

        return panel.y.labels

    def readout(self,
                panel: int,
                side: str,
                across: Sequence[float],
                up: Sequence[float],
                rows: list[list[list[str]]],
                kind: str,
                name: str,
                labels: Sequence[str] = (),
                along: str = "x") -> None:
        """An invisible trace of markers at a mark's points, carrying each
        point's heading and rows for the popup, and plain text for plotly's
        own labels where the popup cannot run."""
        x, y = self.refs(panel, side)
        axis = self.panels[panel].x if along == "x" else self.panels[panel].y
        places = across if along == "x" else up
        heads = [labels[index] if index < len(labels) and labels[index]
                 else heading(axis, place) for index, place in enumerate(places)]
        data = [[head, "<br>".join(f"{label}: {value}" if label else value
                                   for _, label, value in row), json.dumps(row)]
                for head, row in zip(heads, rows, strict=True)]
        self.drawn.traces.append({
            "type": "scatter", "mode": "markers", "x": self.x_of(panel, across),
            "y": list(up) if along == "y" else self.lift(panel, side, up), "xaxis": x,
            "yaxis": y, "name": name, "showlegend": False,
            "marker": {"size": 6, "opacity": 0}, "customdata": data,
            "hovertemplate": "%{customdata[1]}<extra></extra>",
            "meta": {"beacon": {"kind": kind, "order": self.rank(name)}}})
        self.order += 1

    def rank(self, name: str) -> int:
        """Where a series' row comes in the popup: in the legend's order, then
        in the order the marks are drawn."""
        if name in self.legend:
            return self.legend.index(name)

        return len(self.legend) + self.order

    def rule(self, mark: RuleMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        line: dict[str, Any] = {"color": self.paint.ink(mark.colour), "width": mark.width * PT}
        if mark.dashed:
            width = mark.width * PT
            line["dash"] = f"{DASH[0] * width:.2f}px,{DASH[1] * width:.2f}px"
        if mark.vertical:
            at = self.x_of(mark.panel, [mark.value])[0]
            shape = {"x0": at, "x1": at, "xref": x, "y0": 0, "y1": 1, "yref": f"{y} domain"}
        else:
            at = self.lift(mark.panel, mark.side, [mark.value])[0]
            shape = {"x0": 0, "x1": 1, "xref": f"{x} domain", "y0": at, "y1": at, "yref": y}
        # Under the series, or over them; plotly's "between" layer fails to
        # draw a line placed by the plot's domain, and with it every shape.
        self.drawn.shapes.append({"type": "line", "line": line,
                                  "layer": "below" if mark.layer < 2.0 else "above",
                                  **shape})

    def span(self, mark: SpanMark) -> None:
        x, y = self.refs(mark.panel)
        start, end = self.x_of(mark.panel, [mark.start, mark.end])
        self.drawn.shapes.append({"type": "rect", "x0": start, "x1": end, "xref": x,
                                  "y0": 0, "y1": 1, "yref": f"{y} domain", "layer": "below",
                                  "fillcolor": self.paint.ink(mark.colour),
                                  "line": {"width": 0}})

    def text(self, mark: TextMark) -> None:
        x, y = self.refs(mark.panel, mark.side)
        across_data = mark.coords in ("data", "xdata")
        up_data = mark.coords in ("data", "ydata")
        if mark.clip and not self.inside(mark, across_data, up_data):
            return
        font: dict[str, Any] = {"family": self.paint.family, "size": mark.size * PT,
                                "color": self.paint.ink(mark.colour), "weight": mark.weight}
        canvas = self.paint.canvas
        if mark.halo and canvas is not None:
            font["shadow"] = ", ".join([f"{canvas} 0 0 1px"] * 4)
        self.drawn.annotations.append({
            "text": mark.text, "showarrow": False, "font": font, "borderpad": 0,
            "x": self.x_of(mark.panel, [mark.x])[0] if across_data else mark.x,
            "y": self.lift(mark.panel, mark.side, [mark.y])[0] if up_data else mark.y,
            "xref": x if across_data else f"{x} domain",
            "yref": y if up_data else f"{y} domain",
            "xanchor": mark.ha, "yanchor": "bottom", "xshift": mark.dx * PT,
            "yshift": mark.dy * PT + shift(mark.text, mark.size,
                                           Font(self.paint.style.text.font, mark.weight),
                                           mark.va)})

    def inside(self,
               mark: TextMark,
               across_data: bool,
               up_data: bool) -> bool:
        laid = self.layout.panels[mark.panel]
        up = laid.y2 if mark.side == "right" and laid.y2 is not None else laid.y
        checks = [(across_data, mark.x, laid.x), (up_data, mark.y, up)]

        return all(min(axis.low, axis.high) <= value <= max(axis.low, axis.high)
                   for data, value, axis in checks if data)

    def cells(self, mark: CellsMark) -> None:
        x, y = self.refs(mark.panel)
        stops = [self.paint.ink(stop) for stop in mark.stops]
        last = max(len(stops) - 1, 1)
        panel = self.panels[mark.panel]
        rows = list(panel.y.names) or [str(index) for index in range(len(mark.values))]
        columns = list(panel.x.names) or [str(index) for index in range(len(mark.values[0]))]
        span = (mark.high - mark.low) or 1.0
        box = self.layout.panels[mark.panel].box
        wide = box.width * PX / max(len(columns), 1)
        tall = box.height * PX / max(len(rows), 1)
        data = [[self.cell(mark, stops, span, i, j, rows, columns)
                 for j in range(len(row))] for i, row in enumerate(mark.values)]

        self.drawn.traces.append({
            "type": "heatmap", "xaxis": x, "yaxis": y, "showscale": False,
            "z": [[None if value is None else value for value in row] for row in mark.values],
            "x": list(range(len(columns))), "y": list(range(len(rows))),
            "zmin": mark.low, "zmax": mark.high, "zsmooth": "best" if mark.blended else False,
            "colorscale": [[index / last, stop] for index, stop in enumerate(stops)],
            "xgap": mark.gap * wide, "ygap": mark.gap * tall,
            "customdata": data, "hovertemplate": "%{customdata[1]}<extra></extra>",
            "hoverongaps": False, "meta": {"beacon": {"kind": "cell", "order": 0}}})

    def cell(self,
             mark: CellsMark,
             stops: list[str],
             span: float,
             i: int,
             j: int,
             rows: list[str],
             columns: list[str]) -> list[str]:
        value = mark.values[i][j]
        if value is None:
            return ["", "", "[]"]
        written = (mark.written[i][j] if mark.written else value_text(value, "number"))
        row = [[colour_at(stops, (value - mark.low) / span), "", written]]

        return [f"{rows[i]} \N{MIDDLE DOT} {columns[j]}", written, json.dumps(row)]


def heading(axis: Axis,
            place: float) -> str:
    """What a point is, from where it sits across: a date, a category's name,
    or a number."""
    if axis.kind == "date":
        return date_text(place)
    if axis.names and axis.ticks:
        nearest = min(range(len(axis.ticks)), key=lambda index: abs(axis.ticks[index] - place))
        return axis.names[nearest]

    return value_text(place, axis.labels)
