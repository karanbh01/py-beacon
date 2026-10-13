# src/beacon/plot/layout.py
"""
A chart's layout as a value: where everything in its frame goes, worked out
once, without a renderer, for every renderer to draw from.

A renderer describes what it will draw as a `Chart`: the figure's size, the
heading and notes, each panel's axes (their data ranges, how their ticks are
spaced and written) and the labels it writes on them, and the legend's
entries. `beacon.plot.arrange.lay_out()` turns that and a `Style` into a
`Layout`: the figure's size; each panel's box; each axis's range, ticks and
tick labels (a blanked corner label included) and its title; the heading,
accent bar and mark; the legend's entries with their keys; and the notes'
lines.

Positions are in inches from the figure's bottom left. Text is anchored as
matplotlib anchors it: `ha` and `va` place the text's box on (x, y), and a
rotated title turns about that point. Colours are names, never resolved
colours, so one layout draws in any theme: a token name such as "accent" or
"text-primary", "accent-bar" for the bar beside the title, "palette-N" for
the Nth series colour, or a literal "#rrggbb" a caller chose.

Standard library and numpy only: the plotly renderer lays charts out without
importing matplotlib.
"""
# BN-298.
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .metrics import Font
from .ticks import Spacing

# A colour by name (see above).
Colour = str

# How an axis measures: numbers, dates (days since 1970-01-01), names at
# 0, 1, 2..., or ticks a chart fixed itself (a heatmap's rows).
AxisKind = Literal["number", "date", "names", "fixed"]

# How a number axis writes its ticks: plainly, as a percentage, or as a
# chart's own function of the value.
LabelStyle = Literal["number", "percent"] | Callable[[float], str]

HorizontalAlignment = Literal["left", "center", "right"]
VerticalAlignment = Literal["top", "center", "bottom", "baseline", "center_baseline"]
Key = Literal["line", "dashed", "marker", "patch"]


@dataclass(frozen=True)
class Box:
    """A rectangle, in inches from the figure's bottom left."""

    left: float
    bottom: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """Its right edge."""
        return self.left + self.width

    @property
    def top(self) -> float:
        """Its top edge."""
        return self.bottom + self.height


@dataclass(frozen=True)
class Text:
    """Text placed on the figure.

    Attributes:
        text: What it says; lines are separated by newlines.
        x: Its anchor across, in inches.
        y: Its anchor up, in inches.
        size: In points.
        font: The face.
        colour: By name.
        ha: Which side of the text's box sits on x.
        va: Which side of it sits on y.
        rotation: Degrees anticlockwise, about the anchor.
        line_height: Between lines, as a multiple of the size.
        id: Names the piece, such as "beacon-title".
    """

    text: str
    x: float
    y: float
    size: float
    font: Font
    colour: Colour
    ha: HorizontalAlignment = "left"
    va: VerticalAlignment = "baseline"
    rotation: float = 0.0
    line_height: float = 1.2
    id: str = ""


# What a chart asks for.


@dataclass(frozen=True)
class Axis:
    """One axis of a panel, as the chart sees it.

    Attributes:
        kind: How it measures.
        low: The low end of its data, or of its range for names and fixed
            ticks.
        high: The high end.
        ticks: For names and fixed ticks, where they are.
        names: For names and fixed ticks, what they say.
        labels: For numbers, how they are written.
        spacing: For numbers, how the ticks are spaced.
        title: The axis's title, or empty.
        show_labels: Whether its tick labels are drawn (a panel above another
            shares the lower one's).
        rotation: Its tick labels' angle.
        align: Their alignment across.
        inverted: Whether it runs high to low (a heatmap's rows).
        tick_length: Its tick marks' length in points as drawn, or None for
            the style's.
        shared: Another panel whose axis this one is: the same range and
            ticks.
    """

    kind: AxisKind
    low: float
    high: float
    ticks: tuple[float, ...] = ()
    names: tuple[str, ...] = ()
    labels: LabelStyle = "number"
    spacing: Spacing = field(default_factory=Spacing)
    title: str = ""
    show_labels: bool = True
    rotation: float = 0.0
    align: HorizontalAlignment = "center"
    inverted: bool = False
    tick_length: float | None = None
    shared: int | None = None

    @property
    def measures(self) -> bool:
        """Whether the axis measures something, and so ends on its ticks."""
        return self.kind in ("number", "date")


@dataclass(frozen=True)
class Label:
    """Text a chart writes on a panel, such as the value at a line's end,
    which the figure's sides are widened to keep inside.

    Attributes:
        text: What it says.
        x: Where, in the panel's data.
        y: Where, in the panel's data.
        dx: Offset across, in points.
        dy: Offset up, in points.
        size: In points.
        font: The face.
        ha: Which side of its box sits on the point.
        va: Which side of it sits on the point.
        rotation: Degrees anticlockwise.
        fraction: Whether x and y are fractions of the panel's box rather
            than data.
        clip: Whether it is left out when its point falls outside the
            panel.
    """

    text: str
    x: float
    y: float
    dx: float
    dy: float
    size: float
    font: Font
    ha: HorizontalAlignment = "left"
    va: VerticalAlignment = "baseline"
    rotation: float = 0.0
    fraction: bool = False
    clip: bool = False


@dataclass(frozen=True)
class ColourBar:
    """A colour scale beside a panel, taking a share of its width.

    Attributes:
        low: The scale's low end.
        high: Its high end.
        fraction: Its share of the panel's width.
        pad: The gap between, as a share of the panel's width.
        aspect: Its height to its width.
        labels: Text written beside it, such as what its ends mean.
    """

    low: float
    high: float
    fraction: float = 0.046
    pad: float = 0.04
    aspect: float = 20.0
    labels: tuple[Label, ...] = ()


@dataclass(frozen=True)
class Panel:
    """One plot of a chart.

    Attributes:
        x: Its horizontal axis.
        y: Its vertical axis.
        weight: Its share of the chart's height, against the other panels.
        equal: Whether a unit is as long across as up (a heatmap's squares).
        labels: Text written on it.
        colour_bar: A colour scale beside it.
    """

    x: Axis
    y: Axis
    weight: float = 1.0
    equal: bool = False
    labels: tuple[Label, ...] = ()
    colour_bar: ColourBar | None = None


@dataclass(frozen=True)
class LegendItem:
    """One legend entry: its label, its colour and the kind of key it has."""

    label: str
    colour: Colour
    key: Key = "line"


@dataclass(frozen=True)
class Chart:
    """What a chart asks the layout for.

    Attributes:
        size: The figure's width and height, in inches.
        title: The heading.
        subtitle: Beneath it; empty for none.
        notes: The notes line, without its "Notes:" prefix; empty for none.
        panels: Top to bottom, in one column.
        gap: Between panels, as a share of their average height.
        legend: Its entries; none for no legend.
        bottom_room: Inches kept free under the plot for its tick labels; by
            default one row of them.
    """

    size: tuple[float, float]
    title: str
    subtitle: str = ""
    notes: str = ""
    panels: tuple[Panel, ...] = ()
    gap: float = 0.2
    legend: tuple[LegendItem, ...] = ()
    bottom_room: float | None = None


# What the layout gives back.


@dataclass(frozen=True)
class Tick:
    """A tick: where it is and what it says (empty where it is blanked)."""

    value: float
    label: str


@dataclass(frozen=True)
class AxisLayout:
    """An axis as drawn.

    Attributes:
        kind: How it measures.
        low: The low end shown.
        high: The high end shown.
        ticks: Every tick inside the range, with its label.
        offset: Written once at the axis's end, such as "2024-Feb"; usually
            empty.
        title: Its title, placed.
        ended: Whether it ends on its outermost ticks.
        show_labels: Whether its tick labels are drawn.
        rotation: Its tick labels' angle.
        align: Their alignment across.
        inverted: Whether it runs high to low.
    """

    kind: AxisKind
    low: float
    high: float
    ticks: tuple[Tick, ...]
    offset: str = ""
    title: Text | None = None
    ended: bool = False
    show_labels: bool = True
    rotation: float = 0.0
    align: HorizontalAlignment = "center"
    inverted: bool = False


@dataclass(frozen=True)
class ColourBarLayout:
    """A colour scale as drawn: its box, its axis and the text beside it."""

    box: Box
    axis: AxisLayout
    labels: tuple[Text, ...] = ()


@dataclass(frozen=True)
class PanelLayout:
    """A panel as drawn: the box its plot fills and its two axes."""

    box: Box
    x: AxisLayout
    y: AxisLayout
    colour_bar: ColourBarLayout | None = None


@dataclass(frozen=True)
class LegendEntry:
    """A legend entry as drawn: its key's box and its label."""

    label: str
    colour: Colour
    key: Key
    key_box: Box
    text: Text


@dataclass(frozen=True)
class LegendLayout:
    """The legend as drawn.

    Attributes:
        position: "heading", "bottom" or "right".
        columns: How many columns its entries run in.
        box: The box its entries fill.
        entries: Each entry, in the order given.
        size: Its labels' size, in points.
        anchor: The point the legend is pinned to, in inches.
        corner: Which of its corners is pinned there, such as "lower right".
    """

    position: str
    columns: int
    box: Box
    entries: tuple[LegendEntry, ...]
    size: float
    anchor: tuple[float, float]
    corner: str


@dataclass(frozen=True)
class Layout:
    """A chart's whole frame, laid out.

    Attributes:
        width: The figure's width, in inches.
        height: Its height.
        panels: Each panel, top to bottom.
        title: The heading.
        subtitle: Beneath it, or None.
        notes: The notes, wrapped to the plot's width, or None.
        accent_bar: The bar beside the heading, or None.
        mark: The box the beta mark fills, or None.
        legend: The legend, or None.
    """

    width: float
    height: float
    panels: tuple[PanelLayout, ...]
    title: Text
    subtitle: Text | None = None
    notes: Text | None = None
    accent_bar: Box | None = None
    mark: Box | None = None
    legend: LegendLayout | None = None
