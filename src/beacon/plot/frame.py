# src/beacon/plot/frame.py
"""
The furniture around a chart: its title and subtitle, its notes, and the mark.

A chart that makes its own figure is framed the way the application frames
one:

* the **title** in Inter Tight and the **subtitle** beneath it in Source
  Serif 4 italic, both left aligned, with a thin square accent bar down their
  left, level with the y axis
* the beta **mark**, the glyph the application shows in its menu bar, just
  left of the bar and centred on it
* one italic **Notes** line at the bottom left: where the data came from, the
  dates it covers, and whatever the chart itself needs said, run together as
  sentences rather than listed

Every axis that measures something ends on its outermost ticks, so its line
stops where its scale does.

A chart drawn on an axes the caller supplied gets a plain title and its note
instead, because the caller's figure may hold several charts, and one figure
cannot carry a frame for each.

The fonts ship in the package under the SIL Open Font License, whose texts sit
beside them, so a chart looks the same on every machine and the image
regression tests can compare it.
"""
import functools
import logging
import textwrap
import weakref
from pathlib import Path
from typing import Any

import pandas as pd

from .._optional import require
from ..tokens import LIGHT

require("matplotlib", "Charting")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from . import design  # noqa: E402
from .legend import LegendPlan, text_width  # noqa: E402
from .legend import place as place_legend  # noqa: E402
from .legend import place_at as place_legend_at  # noqa: E402
from .legend import plan as legend_plan  # noqa: E402
from .mark import MARK_INCHES, draw_mark  # noqa: E402

# MARK_LABEL is re-exported: callers find the mark's axes by it through frame.
from .scales import MARK_LABEL, apply_grid, end_on_ticks, fit_sides  # noqa: E402, F401
from .style import SCALE, apply, font_family  # noqa: E402
from .themes import Theme, for_background, of_figure, remember, theme_named  # noqa: E402

logger = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "fonts"
# Inter Tight regular for the title and light for everything else on the
# chart; Source Serif 4 at 350 for the subtitle, an instance cut from its
# variable font (SIL Open Font License, which permits the cut).
FONT_FILES = ("InterTight-Regular.ttf", "InterTight-Light.ttf",
              "InterTight-LightItalic.ttf", "SourceSerif4-Italic350.ttf")

TITLE_FONT = "Inter Tight"
SUBTITLE_FONT = "Source Serif 4"
SUBTITLE_WEIGHT = 350
NOTES_FONT = "Inter Tight"
CHART_WEIGHT = 300

# Sizes in points: a 9pt title over a 6.3pt subtitle, set tight beside the
# mark.
TITLE_SIZE = 9.0
SUBTITLE_SIZE = 6.3
# The notes, as drawn: a quiet line under the chart, smaller than its tick
# labels.
NOTES_SIZE = 4.5

# Distances in points.
BAR_WIDTH = 2.5
BAR_GAP = 5.0
LINE_GAP = 0.5

# Distances in inches, so a frame keeps its proportions on any figure size.
MARGIN_INCHES = 0.25
PLOT_GAP_INCHES = 0.12
TICK_ROOM_INCHES = 0.3 * SCALE
NOTES_GAP_INCHES = 0.04 * SCALE
# Between the plot and a legend placed under it or beside it.
LEGEND_GAP_INCHES = 0.12

# Line heights as multiples of the font size: tight in the heading, so its
# two lines sit together beside the mark, and easier in the notes. The frame
# is laid out from the sizes alone: measuring the drawn text needs a
# renderer, which not every backend can give before the figure is shown.
HEADING_LINE_HEIGHT = 1.1
NOTES_LINE_HEIGHT = 1.2

# An Inter Tight character's average advance as a share of its size, for
# wrapping the notes to the width they have.
CHARACTER_WIDTH = 0.5

# What a data store records as its source, in words.
SOURCE_NAMES = {
    "synthetic": "Synthetic data",
    "yfinance": "Yahoo Finance",
    "local": "Local data store",
    "imported": "Imported data",
    "database": "Database",
}

# The label on the axes that holds the mark, and the ids on the frame's other
# pieces and on a level chart's reference line, so a caller can find them.
REFERENCE_ID = "beacon-reference"
PARTS = ("title", "subtitle", "notes")

# Figures this module created, which are the ones it frames. Weak, so a closed
# figure is not kept alive by having once been drawn here.
_OWNED: "weakref.WeakSet[Figure]" = weakref.WeakSet()


@functools.cache
def register_fonts() -> None:
    """Make the bundled fonts known to matplotlib. Runs once; cached."""
    for name in FONT_FILES:
        font_manager.fontManager.addfont(str(FONTS / name))

    logger.debug("Registered the chart fonts from %s.", FONTS)


def new_figure(size: tuple[float, float]) -> Figure:
    """A figure this module will frame, in the current theme and style."""
    apply()
    figure = plt.figure(figsize=size)
    _OWNED.add(figure)
    remember(figure, figure.get_facecolor())

    return figure


def owns(figure: Any) -> bool:
    """Whether a figure was made here, and so gets the whole frame."""
    return isinstance(figure, Figure) and figure in _OWNED


def theme_of(ax: Axes) -> Theme:
    """The theme an axes is drawn in.

    The one its figure was made in when beacon made it; otherwise read off
    the figure's background, because a caller may have styled this figure
    alone, or drawn it on matplotlib's own white.
    """
    figure = ax.get_figure()

    if figure is None:
        return theme_named(LIGHT)

    return of_figure(figure) or for_background(figure.get_facecolor())


def mode_of(ax: Axes) -> str:
    """Which set of Beacon colours an axes' theme draws with: LIGHT or DARK."""
    return theme_of(ax).mode


# The token colours a style can replace, and the setting that replaces each.
STYLE_COLOURS = {"success": "up", "danger": "down", "text-primary": "text",
                 "text-secondary": "muted", "period": "period"}


def ink(ax: Axes,
        name: str) -> str:
    """One token colour, in whichever theme this axes is drawn in, or the
    current style's colour in its place."""
    setting = STYLE_COLOURS.get(name)
    chosen = getattr(design.current().colors, setting) if setting else None

    return chosen or theme_of(ax).colour(name)


def provenance(result: Any,
               what: str,
               dates: pd.Index) -> str:
    """Where a result's data came from and the dates it covers, as a sentence.

    Args:
        result: Anything bound to the data it was calculated from.
        what: What the dates are of, such as "backtest" or "index".
        dates: The dates drawn.

    Returns:
        str: Such as "Source: Synthetic data, backtest dated from 02/01/2024
        to 31/12/2024.", or "weights as of 01/10/2024" for a single date. The
        source is left out when the data recorded none, and the whole is
        empty when there are no dates.
    """
    if not len(dates):
        return ""

    window = (f"{what} as of {_day(dates[0])}" if len(dates) == 1
              else f"{what} dated from {_day(dates[0])} to {_day(dates[-1])}")
    fetcher = getattr(result, "_data_fetcher", None)
    source = getattr(fetcher, "source", None)

    if source:
        return f"Source: {SOURCE_NAMES.get(source, source)}, {window}."

    return f"{window[0].upper()}{window[1:]}."


def sentences(*parts: str) -> str:
    """Notes run together as sentences, skipping the empty ones."""
    return " ".join(part.strip() for part in parts if part and part.strip())


def finish(ax: Axes,
           title: str,
           subtitle: str = "",
           ylabel: str = "",
           notes: str = "",
           note_offset: float = -0.14,
           bottom_room: float | None = None) -> Axes:
    """Give a chart its title, subtitle, axis label and notes.

    The whole frame when the chart made its own figure; a plain title and the
    notes under the axes when it was drawn on the caller's. Either way the
    chart's axes are ended on their outermost ticks.

    Args:
        ax: The chart's axes; for a chart of several panels, the top one.
        title: The heading.
        subtitle: What the chart shows, beneath the heading.
        ylabel: The vertical axis label.
        notes: The notes line's text, without the "Notes:" prefix.
        note_offset: Where the notes sit under a supplied axes, in axes
            coordinates. A short panel needs a larger offset.
        bottom_room: Inches kept free under the plot for its tick labels and
            anything else the chart writes beneath it; by default room for
            one row of tick labels at the current scale.

    Returns:
        Axes: The same axes.
    """
    if ylabel:
        ax.set_ylabel(ylabel)

    if bottom_room is None:
        bottom_room = TICK_ROOM_INCHES * design.current().size.scale

    apply_grid(ax)
    figure = ax.get_figure()

    if isinstance(figure, Figure) and owns(figure):
        frame(ax,
              title,
              subtitle,
              notes,
              theme_of(ax),
              bottom_room)

        return ax

    end_on_ticks(ax)
    register_fonts()
    ax.set_title(title, loc="left", pad=12,
                 fontfamily=font_family(design.current().heading.title_font),
                 fontweight="normal")

    if notes and design.current().notes.show:
        ax.text(0.0, note_offset, notes, transform=ax.transAxes,
                fontsize=7 * SCALE * design.current().size.scale,
                color=ink(ax, "text-primary"), va="top",
                gid="beacon-notes")

    return ax


def frame_text(ax: Axes,
               part: str) -> str:
    """What a chart's title, subtitle or notes say.

    Args:
        ax: The axes a chart method returned.
        part: "title", "subtitle" or "notes".

    Returns:
        str: The text on one line, the notes without their "Notes:" prefix,
        and empty when the chart has none.

    Raises:
        ValueError: If *part* is not one of the three.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {', '.join(PARTS)}, not {part!r}")

    figure = ax.get_figure()
    texts = [*(figure.texts if isinstance(figure, Figure) else []), *ax.texts]
    found = next((text.get_text() for text in texts
                  if text.get_gid() == f"beacon-{part}"), None)

    if found is None and part == "title":
        found = ax.get_title(loc="left")

    return " ".join((found or "").split()).removeprefix("Notes: ")


def frame(ax: Axes,
          title: str,
          subtitle: str,
          notes: str,
          theme: Theme,
          bottom_room: float = TICK_ROOM_INCHES) -> None:
    """Draw the heading, the mark, the legend and the notes, and fit the plot
    between.

    The accent bar and the notes start at the y axis line, the mark sits just
    left of the bar, and the legend ends where the x axis does, so the frame
    lines up with the plot it surrounds. Every size, font and colour comes
    from the current style (beacon.plot.design).

    Args:
        ax: The chart's axes; for a chart of several panels, the top one.
        title: The heading.
        subtitle: Beneath the heading; left out when empty.
        notes: The notes line's text, without the prefix; left out when empty.
        theme: The theme the chart is drawn in.
        bottom_room: Inches kept free under the plot.
    """
    register_fonts()

    figure = ax.get_figure()
    assert isinstance(figure, Figure)  # finish() frames only figures it owns

    current = design.current()
    heading, zoom = current.heading, current.size.scale
    hs = heading.scale
    text_zoom = zoom * current.text.scale
    title_size, subtitle_size = heading.title_size * hs, heading.subtitle_size * hs
    title_font, subtitle_font = font_family(heading.title_font), font_family(heading.subtitle_font)
    notes_font, notes_size = font_family(current.text.font), current.notes.size * text_zoom
    bar_width = BAR_WIDTH * hs if heading.accent_bar else 0.0
    bar_gap = BAR_GAP * hs if heading.accent_bar else 0.0
    line_gap = LINE_GAP * hs
    margin = MARGIN_INCHES * zoom
    plot_gap, notes_gap = PLOT_GAP_INCHES * zoom, NOTES_GAP_INCHES * zoom
    text_colour = current.colors.text or theme.colour("text-primary")
    muted_colour = current.colors.muted or theme.colour("text-secondary")
    notes = notes if current.notes.show else ""
    position = current.legend.position

    shown = ax.get_legend()
    if position == "none" and shown is not None:
        shown.remove()

    width, height = (float(size) for size in figure.get_size_inches())
    top = height - margin

    bottom = top - title_size * HEADING_LINE_HEIGHT / 72.0
    subtitle_top = bottom - line_gap / 72.0
    if subtitle:
        bottom = subtitle_top - subtitle_size * HEADING_LINE_HEIGHT / 72.0

    # How far the heading's text reaches past the plot's left edge, which is
    # the room the legend has to keep clear of.
    reach = (bar_width + bar_gap) / 72.0 + max(
        text_width(title, title_size, title_font),
        text_width(subtitle, subtitle_size, subtitle_font,
                   "italic" if heading.subtitle_italic else "normal", SUBTITLE_WEIGHT))

    # Placed in passes. The plot's height depends on how many lines the notes
    # take and how tall the legend is, their widths on where the axes end up,
    # and an axes with a fixed aspect (a heatmap) narrows as it shortens.
    lines: list[str] = []
    band = bottom
    legend_layout = None
    beneath = 0.0
    for _ in range(3):
        floor = margin + len(lines) * notes_size * NOTES_LINE_HEIGHT / 72.0
        figure.subplots_adjust(top=(band - plot_gap) / height,
                               bottom=(floor + notes_gap + bottom_room + beneath) / height)
        # The axes are ended on their ticks at the size the plot has in this
        # pass: matplotlib picks its tick step from the plot's size, so ends
        # chosen for a taller plot can fall between the ticks of a shorter one.
        for panel in figure.get_axes():
            end_on_ticks(panel)

        fit_sides(figure, margin)
        if position == "right" and legend_layout is not None:
            # Room on the right for a legend of one column.
            room = legend_layout.width + LEGEND_GAP_INCHES * zoom
            figure.subplots_adjust(right=min(figure.subplotpars.right,
                                             (width - margin - room) / width))
        left, right = _plot_edges(ax, width)

        if notes:
            lines = _wrap(f"Notes: {notes}", right - left, notes_size)

        if position == "heading":
            legend_layout = legend_plan(ax, reach, right - left)
            band = _band_bottom(bottom, top, legend_layout)
        elif position == "bottom":
            legend_layout = legend_plan(ax, 0.0, right - left, under=True)
            beneath = legend_layout.height + LEGEND_GAP_INCHES * zoom if legend_layout else 0.0
        elif position == "right":
            legend_layout = legend_plan(ax, 0.0, right - left, columns=1)

    text_left = left + (bar_width + bar_gap) / 72.0

    if legend_layout is not None:
        if position == "heading":
            # Its bottom on the heading band's: level with the subtitle when it
            # fits beside the heading, growing upward when it has more rows.
            place_legend(ax, legend_layout, right / width, band / height)
        elif position == "bottom":
            place_legend_at(ax, legend_layout, left / width,
                            (floor + notes_gap + beneath) / height, "upper left")
        elif position == "right":
            box = ax.get_position()
            place_legend_at(ax, legend_layout, (width - margin) / width, box.y1, "upper right")

    figure.text(text_left / width, top / height, title,
                fontfamily=title_font, fontsize=title_size, fontweight="normal",
                color=text_colour, ha="left", va="top",
                gid="beacon-title")

    if subtitle:
        figure.text(text_left / width, subtitle_top / height, subtitle,
                    fontfamily=subtitle_font, fontsize=subtitle_size,
                    fontstyle="italic" if heading.subtitle_italic else "normal",
                    fontweight=SUBTITLE_WEIGHT, color=muted_colour,
                    ha="left", va="top", gid="beacon-subtitle")

    if heading.accent_bar:
        # Square-cornered: a Rectangle has no rounding, and with no edge drawn
        # its corners are exactly the box's.
        figure.add_artist(Rectangle((left / width, bottom / height),
                                    bar_width / 72.0 / width,
                                    (top - bottom) / height,
                                    transform=figure.transFigure,
                                    facecolor=current.colors.accent_bar or theme.colour("accent"),
                                    edgecolor="none",
                                    linewidth=0,
                                    gid="beacon-accent-bar"))

    if heading.mark:
        draw_mark(figure,
                   left - BAR_GAP * hs / 72.0,
                   (top + bottom) / 2.0,
                   text_colour,
                   MARK_INCHES * hs)

    if lines:
        figure.text(left / width, floor / height,
                    "\n".join(lines), fontfamily=notes_font, fontsize=notes_size,
                    fontstyle="italic" if current.notes.italic else "normal",
                    fontweight=CHART_WEIGHT, color=text_colour,
                    ha="left", va="top", linespacing=NOTES_LINE_HEIGHT,
                    gid="beacon-notes")


def _band_bottom(heading_bottom: float,
                 top: float,
                 layout: LegendPlan | None) -> float:
    """Where the heading band ends: under the heading, or under the legend
    when it is taller than the heading or sits beneath it."""
    if layout is None:
        return heading_bottom

    if layout.beside:
        return min(heading_bottom, top - layout.height)

    return heading_bottom - LINE_GAP / 72.0 - layout.height


def _plot_edges(ax: Axes,
                width: float) -> tuple[float, float]:
    """Where the axes' left and right edges fall, in inches from the left.

    Applies the axes' aspect first, so a heatmap reports the box it is drawn
    in rather than the one it was given.
    """
    ax.apply_aspect()
    box = ax.get_position()

    return box.x0 * width, box.x1 * width


def _wrap(text: str,
          inches: float,
          size: float = NOTES_SIZE) -> list[str]:
    """The notes broken into lines that fit the width they have."""
    characters = max(20, int(inches * 72.0 / (size * CHARACTER_WIDTH)))

    return textwrap.wrap(text, characters) or [text]


def _day(when: Any) -> str:
    """A date as dd/mm/yyyy."""
    return str(pd.Timestamp(when).strftime("%d/%m/%Y"))
