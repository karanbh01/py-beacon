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
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402

from .legend import LegendPlan, text_width  # noqa: E402
from .legend import place as place_legend  # noqa: E402
from .legend import plan as legend_plan  # noqa: E402
from .mark import beta_outline  # noqa: E402
from .scales import MARK_LABEL, end_on_ticks, fit_sides  # noqa: E402
from .style import SCALE  # noqa: E402
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
MARK_INCHES = 0.192
PLOT_GAP_INCHES = 0.12
TICK_ROOM_INCHES = 0.3 * SCALE
NOTES_GAP_INCHES = 0.04 * SCALE

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
    """A figure this module will frame, in the current theme."""
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


def ink(ax: Axes,
        name: str) -> str:
    """One token colour, in whichever theme this axes is drawn in."""
    return theme_of(ax).colour(name)


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
           bottom_room: float = TICK_ROOM_INCHES) -> Axes:
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
            anything else the chart writes beneath it.

    Returns:
        Axes: The same axes.
    """
    if ylabel:
        ax.set_ylabel(ylabel)

    figure = ax.get_figure()

    if isinstance(figure, Figure) and owns(figure):
        for panel in figure.get_axes():
            end_on_ticks(panel)

        frame(ax,
              title,
              subtitle,
              notes,
              theme_of(ax),
              bottom_room)

        return ax

    end_on_ticks(ax)
    register_fonts()
    ax.set_title(title, loc="left", pad=12, fontfamily=TITLE_FONT,
                 fontweight="normal")

    if notes:
        ax.text(0.0, note_offset, notes, transform=ax.transAxes,
                fontsize=7 * SCALE, color=ink(ax, "text-primary"), va="top",
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
    lines up with the plot it surrounds.

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

    width, height = (float(size) for size in figure.get_size_inches())
    top = height - MARGIN_INCHES

    bottom = top - TITLE_SIZE * HEADING_LINE_HEIGHT / 72.0
    subtitle_top = bottom - LINE_GAP / 72.0
    if subtitle:
        bottom = subtitle_top - SUBTITLE_SIZE * HEADING_LINE_HEIGHT / 72.0

    # How far the heading's text reaches past the plot's left edge, which is
    # the room the legend has to keep clear of.
    reach = (BAR_WIDTH + BAR_GAP) / 72.0 + max(
        text_width(title, TITLE_SIZE, TITLE_FONT),
        text_width(subtitle, SUBTITLE_SIZE, SUBTITLE_FONT, "italic", SUBTITLE_WEIGHT))

    # Placed in passes. The plot's height depends on how many lines the notes
    # take and how tall the legend is, their widths on where the axes end up,
    # and an axes with a fixed aspect (a heatmap) narrows as it shortens.
    lines: list[str] = []
    band = bottom
    legend_layout = None
    for _ in range(3):
        floor = MARGIN_INCHES + len(lines) * NOTES_SIZE * NOTES_LINE_HEIGHT / 72.0
        figure.subplots_adjust(top=(band - PLOT_GAP_INCHES) / height,
                               bottom=(floor + NOTES_GAP_INCHES + bottom_room) / height)
        fit_sides(figure, MARGIN_INCHES)
        left, right = _plot_edges(ax, width)

        if notes:
            lines = _wrap(f"Notes: {notes}", right - left)

        legend_layout = legend_plan(ax, reach, right - left)
        band = _band_bottom(bottom, top, legend_layout)

    text_left = left + (BAR_WIDTH + BAR_GAP) / 72.0

    if legend_layout is not None:
        # Its bottom on the heading band's: level with the subtitle when it
        # fits beside the heading, growing upward when it has more rows.
        place_legend(ax, legend_layout, right / width, band / height)

    figure.text(text_left / width, top / height, title,
                fontfamily=TITLE_FONT, fontsize=TITLE_SIZE, fontweight="normal",
                color=theme.colour("text-primary"), ha="left", va="top",
                gid="beacon-title")

    if subtitle:
        figure.text(text_left / width, subtitle_top / height, subtitle,
                    fontfamily=SUBTITLE_FONT, fontsize=SUBTITLE_SIZE,
                    fontstyle="italic", fontweight=SUBTITLE_WEIGHT,
                    color=theme.colour("text-secondary"),
                    ha="left", va="top", gid="beacon-subtitle")

    # Square-cornered: a Rectangle has no rounding, and with no edge drawn its
    # corners are exactly the box's.
    figure.add_artist(Rectangle((left / width, bottom / height),
                                BAR_WIDTH / 72.0 / width,
                                (top - bottom) / height,
                                transform=figure.transFigure,
                                facecolor=theme.colour("accent"),
                                edgecolor="none",
                                linewidth=0))

    _draw_mark(figure,
               left - BAR_GAP / 72.0,
               (top + bottom) / 2.0,
               theme.colour("text-primary"))

    if lines:
        figure.text(left / width, floor / height,
                    "\n".join(lines), fontfamily=NOTES_FONT, fontsize=NOTES_SIZE,
                    fontstyle="italic", fontweight=CHART_WEIGHT,
                    color=theme.colour("text-primary"),
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


def _mark_width() -> float:
    """How wide the mark is drawn, in inches."""
    extent = beta_outline().get_extents()

    return MARK_INCHES * extent.width / extent.height


def _draw_mark(figure: Figure,
               right: float,
               middle: float,
               colour_value: str) -> None:
    """The beta mark beside the heading.

    Args:
        figure: The figure to draw on.
        right: Where the mark ends, in inches from the left.
        middle: The heading's middle, in inches from the bottom.
        colour_value: The mark's colour.
    """
    width, height = (float(size) for size in figure.get_size_inches())
    outline = beta_outline()
    extent = outline.get_extents()
    mark_width = _mark_width()

    holder = figure.add_axes(((right - mark_width) / width,
                              (middle - MARK_INCHES / 2.0) / height,
                              mark_width / width,
                              MARK_INCHES / height),
                             label=MARK_LABEL)
    holder.set_axis_off()
    # Unclipped, or the glyph's anti-aliased edges are cut at the holder's
    # boundary.
    holder.add_patch(PathPatch(outline,
                               facecolor=colour_value,
                               edgecolor="none",
                               linewidth=0,
                               clip_on=False))
    holder.set_xlim(extent.x0, extent.x1)
    holder.set_ylim(extent.y0, extent.y1)


def _wrap(text: str,
          inches: float) -> list[str]:
    """The notes broken into lines that fit the width they have."""
    characters = max(20, int(inches * 72.0 / (NOTES_SIZE * CHARACTER_WIDTH)))

    return textwrap.wrap(text, characters) or [text]


def _day(when: Any) -> str:
    """A date as dd/mm/yyyy."""
    return str(pd.Timestamp(when).strftime("%d/%m/%Y"))
