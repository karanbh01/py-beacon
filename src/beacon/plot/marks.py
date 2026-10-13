# src/beacon/plot/marks.py
"""
What a chart draws inside its plot, as values every renderer can draw.

A chart class turns its data into marks: lines, filled areas, bars, points,
the lines a chart is read against, marked periods, text and grids of
coloured cells. Positions are in the panel's data, with dates as days since
1970-01-01 (beacon.plot.dateticks). Colours are names (beacon.plot.colours),
and line widths and sizes are in points as drawn.

Each mark names the panel it is drawn in and, where a panel has a second
vertical axis, the side whose scale it is measured on.

Standard library only, like the layout, so the plotly renderer draws the
same marks without importing matplotlib.
"""
# BN-299.
from dataclasses import dataclass
from typing import Literal

from .layout import Colour

Side = Literal["left", "right"]


@dataclass(frozen=True)
class LineMark:
    """A line through points, in order."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    colour: Colour
    width: float
    dashed: bool = False
    alpha: float = 1.0
    layer: float = 3.0
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class AreaMark:
    """The space between two lines, filled."""

    x: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    colour: Colour
    alpha: float = 1.0
    layer: float = 2.0
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class BarMark:
    """Bars from a base to their values.

    Attributes:
        positions: Where each bar sits along its category axis: its centre,
            or its left edge when *edge*.
        values: How far each reaches.
        thickness: Each bar's thickness, in the category axis's units.
        colours: Each bar's colour.
        horizontal: Whether the bars run across rather than up.
        base: Where each bar starts; zero when empty.
        edge: Whether a position is the bar's left edge rather than its centre.
        seamless: Whether bars that meet are drawn without a seam between.
    """

    positions: tuple[float, ...]
    values: tuple[float, ...]
    thickness: tuple[float, ...]
    colours: tuple[Colour, ...]
    horizontal: bool = False
    base: tuple[float, ...] = ()
    edge: bool = False
    seamless: bool = False
    layer: float = 2.0
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class PointMark:
    """Dots."""

    x: tuple[float, ...]
    y: tuple[float, ...]
    colour: Colour
    size: float
    alpha: float = 1.0
    clip: bool = True
    layer: float = 5.0
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class RuleMark:
    """A line across the whole plot at one value: up it when *vertical*,
    otherwise across it."""

    value: float
    colour: Colour
    width: float
    vertical: bool = False
    dashed: bool = False
    layer: float = 1.0
    id: str = ""
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class SpanMark:
    """A marked period: a box from the bottom of the plot to the top, over
    a range of the horizontal axis, behind everything else."""

    start: float
    end: float
    colour: Colour
    panel: int = 0


@dataclass(frozen=True)
class TextMark:
    """Text on the plot.

    Attributes:
        text: What it says.
        x: Across, in data, or a fraction of the plot when *coords* says so.
        y: Up, likewise.
        colour: By name.
        size: In points.
        weight: The font's weight.
        dx: Offset across, in points.
        dy: Offset up, in points.
        ha: Which side of its box sits on the point.
        va: Which side of it sits on the point.
        coords: "data" for both in data, "axes" for both as fractions of the
            plot, "xdata" for x in data and y as a fraction, "ydata" for x as a
            fraction and y in data.
        clip: Whether it is left out when its point falls outside the plot.
        halo: Whether it is outlined in the background's colour, so it reads
            over the lines it sits on.
    """

    text: str
    x: float
    y: float
    colour: Colour
    size: float
    weight: int = 300
    dx: float = 0.0
    dy: float = 0.0
    ha: Literal["left", "center", "right"] = "left"
    va: Literal["top", "center", "bottom", "baseline"] = "center"
    coords: Literal["data", "axes", "xdata", "ydata"] = "data"
    clip: bool = False
    halo: bool = False
    panel: int = 0
    side: Side = "left"


@dataclass(frozen=True)
class CellsMark:
    """A grid of values drawn as coloured cells, row by row from the top.

    Attributes:
        values: One row of values per row of cells; None leaves a cell empty.
        stops: The colour map's colours from low to high.
        low: The value at the map's low end.
        high: The value at its high end.
        blended: Whether each colour blends into the next rather than each
            cell being its own square.
        gap: The space between cells, as a share of a cell.
    """

    values: tuple[tuple[float | None, ...], ...]
    stops: tuple[Colour, ...]
    low: float
    high: float
    blended: bool = False
    gap: float = 0.0
    panel: int = 0


Mark = LineMark | AreaMark | BarMark | PointMark | RuleMark | SpanMark | TextMark | CellsMark
