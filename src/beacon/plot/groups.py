# src/beacon/plot/groups.py
"""
The groups of chart settings a `Style` holds (beacon.plot.design): size,
heading, text, lines, axes, legend, notes and colours, each with Beacon's
design as its default and each checking its own values.

Standard library only, so `import beacon.plot` stays free of matplotlib.
"""
# BN-303.
import dataclasses
import re
from dataclasses import dataclass
from typing import Any

HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
ASPECT = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")

TICK_DIRECTIONS = ("in", "out")
GRIDS = ("none", "x", "y", "both")
LEGEND_POSITIONS = ("heading", "bottom", "right", "none")


def _positive(name: str,
              value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not value > 0:
        raise ValueError(f"{name} must be a number above zero, got {value!r}.")


def _choice(name: str,
            value: Any,
            options: tuple[str, ...]) -> None:
    if value not in options:
        raise ValueError(f"{name} must be one of {', '.join(options)}, got {value!r}.")


def _colour(name: str,
            value: str | None) -> None:
    if value is not None and not (isinstance(value, str) and HEX.match(value)):
        raise ValueError(f"{name} must be a hex colour such as #4a88c7, got {value!r}.")


def _flag(name: str,
          value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, got {value!r}.")


@dataclass(frozen=True)
class SizeSettings:
    """The chart's size.

    Attributes:
        scale: Zooms the chart, its figure, text and lines together; 1.0 is
            the size charts are drawn at today. The heading keeps its own
            scale.
        width: The designed width in inches, giving more room at the same
            text size; None keeps the chart's own, 6in.
        aspect: Width to height, such as "16:9"; None keeps the chart's own.
        dpi: Dots per inch when a chart is saved as an image.
    """
    scale: float = 1.0
    width: float | None = None
    aspect: str | None = None
    dpi: int = 300

    def __post_init__(self) -> None:
        _positive("size.scale", self.scale)
        if self.width is not None:
            _positive("size.width", self.width)
        if self.aspect is not None and not (isinstance(self.aspect, str)
                                            and ASPECT.match(self.aspect)):
            raise ValueError(f"size.aspect must be width:height, such as 16:9, "
                             f"got {self.aspect!r}.")
        _positive("size.dpi", self.dpi)

    def ratio(self) -> float | None:
        """The aspect as width over height, or None."""
        if self.aspect is None:
            return None
        match = ASPECT.match(self.aspect)
        assert match is not None
        return float(match.group(1)) / float(match.group(2))


@dataclass(frozen=True)
class HeadingSettings:
    """The title, subtitle, accent bar and beta mark.

    Attributes:
        mark: The beta mark left of the accent bar.
        accent_bar: The bar beside the title and subtitle.
        scale: Scales the mark, accent bar, title and subtitle together.
        title_font: A font name, or the path of a font file (.ttf or .otf).
        title_size: In points.
        subtitle_font: A font name, or the path of a font file.
        subtitle_size: In points.
        subtitle_italic: Whether the subtitle is set in italics.
    """
    mark: bool = True
    accent_bar: bool = True
    scale: float = 1.0
    title_font: str = "Inter Tight"
    title_size: float = 9.0
    subtitle_font: str = "Source Serif 4"
    subtitle_size: float = 6.3
    subtitle_italic: bool = True

    def __post_init__(self) -> None:
        _flag("heading.mark", self.mark)
        _flag("heading.accent_bar", self.accent_bar)
        _flag("heading.subtitle_italic", self.subtitle_italic)
        for name in ("scale", "title_size", "subtitle_size"):
            _positive(f"heading.{name}", getattr(self, name))


@dataclass(frozen=True)
class TextSettings:
    """The chart's own text.

    Attributes:
        font: A font name, or the path of a font file (.ttf or .otf).
        scale: The chart's small text alone, at any chart scale.
        label_size: Legend entries and labels on the chart, in points.
    """
    font: str = "Inter Tight"
    scale: float = 1.0
    label_size: float = 5.5

    def __post_init__(self) -> None:
        _positive("text.scale", self.scale)
        _positive("text.label_size", self.label_size)


@dataclass(frozen=True)
class LineSettings:
    """Line weights, in points as designed.

    Attributes:
        series_width: The first, leading series.
        secondary_width: Every other series, such as a benchmark.
        axis_width: The axis lines and ticks.
        reference_width: Lines a chart is read against, such as 100 or zero.
        grid_width: Gridlines, where a chart draws them.
    """
    series_width: float = 1.5
    secondary_width: float = 1.1
    axis_width: float = 0.6
    reference_width: float = 0.5
    grid_width: float = 0.6

    def __post_init__(self) -> None:
        for name in ("series_width", "secondary_width", "axis_width",
                     "reference_width", "grid_width"):
            _positive(f"lines.{name}", getattr(self, name))


@dataclass(frozen=True)
class AxesSettings:
    """The axes: their lines, ticks, labels and ends.

    Attributes:
        x_line: The x axis line.
        y_line: The y axis line.
        x_ticks: The x axis's tick marks.
        y_ticks: The y axis's tick marks.
        tick_direction: "in" or "out".
        tick_length: In points as designed.
        tick_label_size: In points.
        title_size: Axis titles, in points.
        grid: "none", "x", "y" or "both"; None keeps each chart's own.
        end_on_ticks: End each measuring axis on its outermost ticks.
        blank_lowest_label: Leave the lowest y label blank where both axes
            measure, so it does not sit on the first x label.
        reference: Draw the lines a chart is read against, such as 100.
    """
    x_line: bool = True
    y_line: bool = True
    x_ticks: bool = True
    y_ticks: bool = True
    tick_direction: str = "in"
    tick_length: float = 3.5
    tick_label_size: float = 7.8
    title_size: float = 8.0
    grid: str | None = None
    end_on_ticks: bool = True
    blank_lowest_label: bool = True
    reference: bool = True

    def __post_init__(self) -> None:
        for name in ("x_line", "y_line", "x_ticks", "y_ticks", "end_on_ticks",
                     "blank_lowest_label", "reference"):
            _flag(f"axes.{name}", getattr(self, name))
        _choice("axes.tick_direction", self.tick_direction, TICK_DIRECTIONS)
        if self.grid is not None:
            _choice("axes.grid", self.grid, GRIDS)
        for name in ("tick_length", "tick_label_size", "title_size"):
            _positive(f"axes.{name}", getattr(self, name))


@dataclass(frozen=True)
class LegendSettings:
    """The legend.

    Attributes:
        position: "heading" (right of the title), "bottom", "right" or
            "none".
    """
    position: str = "heading"

    def __post_init__(self) -> None:
        _choice("legend.position", self.position, LEGEND_POSITIONS)


@dataclass(frozen=True)
class NotesSettings:
    """The notes line under the chart.

    Attributes:
        show: Whether to draw it.
        size: In points.
        italic: Whether it is set in italics.
    """
    show: bool = True
    size: float = 4.5
    italic: bool = True

    def __post_init__(self) -> None:
        _flag("notes.show", self.show)
        _flag("notes.italic", self.italic)
        _positive("notes.size", self.size)


@dataclass(frozen=True)
class ColorSettings:
    """Colours, each a hex code; None follows the theme.

    Attributes:
        series: The series colours in order, replacing the palette.
        up: Gains.
        down: Losses.
        accent_bar: The bar beside the heading.
        period: Marked periods' boxes.
        period_label: Marked periods' labels.
        text: Text, axes and the mark.
        muted: The subtitle.
        background: The canvas.
    """
    series: tuple[str, ...] | None = None
    up: str | None = None
    down: str | None = None
    accent_bar: str | None = None
    period: str | None = None
    period_label: str | None = None
    text: str | None = None
    muted: str | None = None
    background: str | None = None

    def __post_init__(self) -> None:
        if self.series is not None:
            if isinstance(self.series, str) or not len(self.series):
                raise ValueError("colors.series must be a list of hex colours.")
            object.__setattr__(self, "series", tuple(self.series))
            for i, value in enumerate(self.series):
                _colour(f"colors.series[{i}]", value)
        for f in dataclasses.fields(self):
            if f.name != "series":
                _colour(f"colors.{f.name}", getattr(self, f.name))


GROUPS: dict[str, type] = {
    "size": SizeSettings, "heading": HeadingSettings, "text": TextSettings, "lines": LineSettings,
    "axes": AxesSettings, "legend": LegendSettings, "notes": NotesSettings, "colors": ColorSettings,
}

# Settings by their short names, as configure() and a chart's style= take
# them: the name, and where it lives in the tree.
FLAT: dict[str, tuple[str, str]] = {
    "scale": ("size", "scale"), "width": ("size", "width"), "aspect": ("size", "aspect"),
    "dpi": ("size", "dpi"),
    "mark": ("heading", "mark"), "accent_bar": ("heading", "accent_bar"),
    "heading_scale": ("heading", "scale"),
    "title_font": ("heading", "title_font"), "title_size": ("heading", "title_size"),
    "subtitle_font": ("heading", "subtitle_font"), "subtitle_size": ("heading", "subtitle_size"),
    "subtitle_italic": ("heading", "subtitle_italic"),
    "font": ("text", "font"), "text_scale": ("text", "scale"), "label_size": ("text", "label_size"),
    "series_width": ("lines", "series_width"), "secondary_width": ("lines", "secondary_width"),
    "axis_width": ("lines", "axis_width"), "reference_width": ("lines", "reference_width"),
    "grid_width": ("lines", "grid_width"),
    "x_axis_line": ("axes", "x_line"), "y_axis_line": ("axes", "y_line"),
    "x_ticks": ("axes", "x_ticks"), "y_ticks": ("axes", "y_ticks"),
    "tick_direction": ("axes", "tick_direction"), "tick_length": ("axes", "tick_length"),
    "tick_label_size": ("axes", "tick_label_size"), "axis_title_size": ("axes", "title_size"),
    "grid": ("axes", "grid"), "end_on_ticks": ("axes", "end_on_ticks"),
    "blank_lowest_label": ("axes", "blank_lowest_label"), "reference_lines": ("axes", "reference"),
    "legend": ("legend", "position"),
    "notes": ("notes", "show"), "notes_size": ("notes", "size"),
    "notes_italic": ("notes", "italic"),
    "series_colors": ("colors", "series"), "up_color": ("colors", "up"),
    "down_color": ("colors", "down"),
    "accent_bar_color": ("colors", "accent_bar"), "period_color": ("colors", "period"),
    "period_label_color": ("colors", "period_label"), "text_color": ("colors", "text"),
    "muted_color": ("colors", "muted"), "background": ("colors", "background"),
}
