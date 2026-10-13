# src/beacon/plot/arrange.py
"""
A chart's frame, laid out from what it draws and a `Style`.

`lay_out()` places the heading (the mark, the accent bar, the title and the
subtitle), fits the plot between it and the notes, ends each measuring axis
on its ticks, widens the sides to keep every label inside the figure, and
puts the legend where the style says. Every size, weight, font and colour
comes from the style.

The plot's height depends on how many lines the notes take and how tall the
legend is, their widths on where the plot's sides end up, and a heatmap
narrows as it shortens, so the layout is worked out in passes.
"""
# BN-298: the rules beacon.plot.frame applied inside matplotlib, moved here so
# every renderer draws one layout.
import dataclasses
import re
from dataclasses import dataclass

from . import metrics, panels
from .design import Style
from .fitting import Passes
from .layout import (
    AxisLayout,
    Box,
    ColourBarLayout,
    Description,
    Label,
    Layout,
    LegendLayout,
    PanelLayout,
    Text,
)
from .legend import LegendPlan, entries
from .legend import plan as legend_plan
from .metrics import Font
from .panels import SCALE

# Distances in points, at the heading's own scale.
BAR_WIDTH = 2.5
BAR_GAP = 5.0
LINE_GAP = 0.5

# Distances in inches, at the chart's scale.
MARGIN_INCHES = 0.25
PLOT_GAP_INCHES = 0.12
TICK_ROOM_INCHES = 0.3 * SCALE
NOTES_GAP_INCHES = 0.04 * SCALE
# Between the plot and a legend placed under it or beside it.
LEGEND_GAP_INCHES = 0.12

# Line heights as multiples of the font size: tight in the heading, so its
# two lines sit together beside the mark, and easier in the notes.
HEADING_LINE_HEIGHT = 1.1
NOTES_LINE_HEIGHT = 1.2

TITLE_WEIGHT = 400
SUBTITLE_WEIGHT = 350

# The mark's height beside a heading at its designed size, in inches, and its
# width as a share of that (beacon.plot.mark draws it).
MARK_INCHES = 0.192
MARK_ASPECT = 0.6333483705911227

PASSES = 3


@dataclass(frozen=True)
class _Heading:
    """The heading's sizes and where it sits."""

    title: float
    subtitle: float
    title_font: Font
    subtitle_font: Font
    bar_width: float  # points
    bar_gap: float    # points
    top: float        # inches
    subtitle_top: float
    bottom: float
    reach: float      # how far its text reaches past the plot's left edge


def _heading(chart: Description,
             style: Style) -> _Heading:
    heading = style.heading
    scale = heading.scale
    margin = MARGIN_INCHES * style.size.scale
    title_size, subtitle_size = heading.title_size * scale, heading.subtitle_size * scale
    title_font = Font(heading.title_font, TITLE_WEIGHT)
    subtitle_font = Font(heading.subtitle_font, SUBTITLE_WEIGHT, heading.subtitle_italic)
    bar_width = BAR_WIDTH * scale if heading.accent_bar else 0.0
    bar_gap = BAR_GAP * scale if heading.accent_bar else 0.0

    top = chart.size[1] - margin
    subtitle_top = top - (title_size * HEADING_LINE_HEIGHT + LINE_GAP * scale) / 72.0
    # As tall with or without a subtitle, so the accent bar is one size on
    # every chart; a title alone is centred on it.
    bottom = subtitle_top - subtitle_size * HEADING_LINE_HEIGHT / 72.0

    reach = (bar_width + bar_gap) / 72.0 + max(
        metrics.width(chart.title, title_size, title_font),
        metrics.width(chart.subtitle, subtitle_size, subtitle_font))

    return _Heading(title_size, subtitle_size, title_font, subtitle_font, bar_width,
                    bar_gap, top, subtitle_top, bottom, reach)


def wrap(text: str,
         width: float,
         size: float,
         font: Font) -> list[str]:
    """Text broken into lines no wider than *width* inches, at spaces; a word
    longer than that has a line to itself."""
    lines: list[str] = []
    current = ""

    for token in re.findall(r"\S+|\s+", text):
        if token.isspace():
            current += token if current else ""
            continue
        trial = current + token
        if current and metrics.width(trial, size, font) > width:
            lines.append(current.rstrip())
            current = token
        else:
            current = trial

    if current.strip():
        lines.append(current.rstrip())

    return lines or [text]


def lay_out(chart: Description,
            style: Style) -> Layout:
    """Lay a chart's frame out in a style.

    Args:
        chart: What the chart draws.
        style: The settings it is drawn in.

    Returns:
        Layout: Where everything goes.
    """
    zoom = style.size.scale
    width, height = chart.size
    heading = _heading(chart, style)
    margin = MARGIN_INCHES * zoom
    plot_gap, notes_gap = PLOT_GAP_INCHES * zoom, NOTES_GAP_INCHES * zoom
    bottom_room = chart.bottom_room if chart.bottom_room is not None else TICK_ROOM_INCHES * zoom
    zoom_text = zoom * style.text.scale
    notes_size = style.notes.size * zoom_text
    notes_font = Font(style.text.font, panels.CHART_WEIGHT, style.notes.italic)
    notes = chart.notes if style.notes.show else ""
    position = style.legend.position
    items = chart.legend if position != "none" else ()
    legend_size = style.text.label_size * zoom_text
    legend_font = Font(style.text.font, panels.CHART_WEIGHT)
    labels = [item.label for item in items]

    passes = Passes(chart, style)
    lines: list[str] = []
    band = heading.bottom
    plan: LegendPlan | None = None
    beneath = 0.0
    floor = margin
    left = right = 0.0

    for _ in range(PASSES):
        floor = margin + len(lines) * notes_size * NOTES_LINE_HEIGHT / 72.0
        top_share = (band - plot_gap) / height
        bottom_share = (floor + notes_gap + bottom_room + beneath) / height
        passes.place(bottom_share, top_share)
        passes.end_axes()
        passes.fit_sides(margin)

        if position == "right" and plan is not None:
            room = plan.width + LEGEND_GAP_INCHES * zoom
            passes.sides[1] = min(passes.sides[1], (width - margin - room) / width)

        passes.place(bottom_share, top_share)
        main = passes.placed[0][0]
        left, right = main.left, main.right

        if notes:
            lines = wrap(f"Notes: {notes}", right - left, notes_size, notes_font)

        if position == "heading":
            plan = legend_plan(labels, legend_size, legend_font, heading.reach, right - left)
            band = _band_bottom(heading.bottom, heading.top, plan)
        elif position == "bottom":
            plan = legend_plan(labels, legend_size, legend_font, 0.0, right - left, under=True)
            beneath = plan.height + LEGEND_GAP_INCHES * zoom if plan else 0.0
        elif position == "right":
            plan = legend_plan(labels, legend_size, legend_font, 0.0, right - left, columns=1)

    laid = []

    for (plot, bar), drawn, panel in zip(passes.placed, passes.axes(), chart.panels,
                                         strict=True):
        bar_layout = None
        if bar is not None and drawn.bar is not None and panel.colour_bar is not None:
            bar_layout = ColourBarLayout(bar, drawn.bar, tuple(
                _bar_text(label, bar) for label in panel.colour_bar.labels))
        y2 = _titled(drawn.y2, drawn.y2_title) if drawn.y2 is not None else None
        laid.append(PanelLayout(plot, _titled(drawn.x, drawn.x_title),
                                _titled(drawn.y, drawn.y_title), bar_layout, y2, panel.origin))

    legend = None
    if plan is not None:
        anchor, corner = _legend_anchor(position, left, right, band, floor, notes_gap,
                                        beneath, width, margin, passes.placed[0][0])
        box, placed = entries(items, plan.columns, legend_size, legend_font, anchor, corner)
        legend = LegendLayout(position, plan.columns, box, placed, legend_size, anchor, corner)

    return _framed(chart, style, heading, laid, legend, left,
                   Text("\n".join(lines), left, floor, notes_size, notes_font, "text-primary",
                        ha="left", va="top", line_height=NOTES_LINE_HEIGHT,
                        id="beacon-notes") if lines else None)


def _titled(axis: AxisLayout,
            title: Text | None) -> AxisLayout:
    return axis if title is None else dataclasses.replace(axis, title=title)


def _bar_text(label: Label,
              bar: Box) -> Text:
    """A colour bar's label, placed."""
    return Text(label.text, bar.left + label.x * bar.width + label.dx / 72.0,
                bar.bottom + label.y * bar.height + label.dy / 72.0, label.size,
                label.font, "text-muted", ha=label.ha, va=label.va, rotation=label.rotation)


def _band_bottom(heading_bottom: float,
                 top: float,
                 plan: LegendPlan | None) -> float:
    """Where the heading band ends: under the heading, or under the legend
    when it is taller than the heading or sits beneath it."""
    if plan is None:
        return heading_bottom

    if plan.beside:
        return min(heading_bottom, top - plan.height)

    return heading_bottom - LINE_GAP / 72.0 - plan.height


def _legend_anchor(position: str,
                   left: float,
                   right: float,
                   band: float,
                   floor: float,
                   notes_gap: float,
                   beneath: float,
                   width: float,
                   margin: float,
                   main: Box) -> tuple[tuple[float, float], str]:
    """Where the legend is pinned, and by which corner."""
    if position == "bottom":
        return (left, floor + notes_gap + beneath), "upper left"
    if position == "right":
        return (width - margin, main.top), "upper right"

    # Its bottom on the heading band's: level with the subtitle when it fits
    # beside the heading, growing upward when it has more rows.
    return (right, band), "lower right"


def _framed(chart: Description,
            style: Style,
            heading: _Heading,
            laid: list[PanelLayout],
            legend: LegendLayout | None,
            left: float,
            notes: Text | None) -> Layout:
    """The layout, with the heading placed beside the plot's left edge."""
    scale = style.heading.scale
    text_left = left + (heading.bar_width + heading.bar_gap) / 72.0
    title = Text(chart.title, text_left, heading.top, heading.title, heading.title_font,
                 "text-primary", ha="left", va="top", id="beacon-title")
    if not chart.subtitle:
        title = dataclasses.replace(title, y=(heading.top + heading.bottom) / 2.0, va="center")
    subtitle = None
    if chart.subtitle:
        subtitle = Text(chart.subtitle, text_left, heading.subtitle_top, heading.subtitle,
                        heading.subtitle_font, "text-secondary", ha="left", va="top",
                        id="beacon-subtitle")

    bar = None
    if style.heading.accent_bar:
        bar = Box(left, heading.bottom, heading.bar_width / 72.0, heading.top - heading.bottom)

    mark = None
    if style.heading.mark:
        size = MARK_INCHES * scale
        right = left - BAR_GAP * scale / 72.0
        middle = (heading.top + heading.bottom) / 2.0
        mark = Box(right - size * MARK_ASPECT, middle - size / 2.0, size * MARK_ASPECT, size)

    return Layout(width=chart.size[0], height=chart.size[1], panels=tuple(laid), title=title,
                  subtitle=subtitle, notes=notes, accent_bar=bar, mark=mark, legend=legend)

