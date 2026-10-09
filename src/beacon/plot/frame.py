# src/beacon/plot/frame.py
"""
The furniture around a chart: its title and subtitle, its notes, and the mark.

A chart that makes its own figure is framed the way the application frames
one:

* the **title** in Inter Tight and the **subtitle** beneath it in Playfair
  Display italic, both left aligned, a size apart rather than a step apart,
  with a thin square accent bar down their left
* one italic **Notes** line at the bottom left: where the data came from, the
  dates it covers, and whatever the chart itself needs said, run together as
  sentences rather than listed
* the beta **mark** at the bottom right, the glyph the application shows in
  its menu bar

A chart drawn on an axes the caller supplied gets a plain title and its note
instead, because the caller's figure may hold several charts, and one figure
cannot carry a frame for each.

The fonts ship in the package under the SIL Open Font License, whose texts sit
beside them, so a chart looks the same on every machine and the image
regression tests can compare it.
"""
import functools
import logging
import re
import textwrap
import weakref
from pathlib import Path
from typing import Any

import pandas as pd

from .._optional import require
from ..tokens import DARK, LIGHT, colour

require("matplotlib", "Charting")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as Outline  # noqa: E402

logger = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "fonts"
FONT_FILES = ("InterTight-Regular.ttf", "InterTight-Italic.ttf",
              "PlayfairDisplay-Italic.ttf")

TITLE_FONT = "Inter Tight"
SUBTITLE_FONT = "Playfair Display"
NOTES_FONT = "Inter Tight"

# Sizes in points. The title leads, but by a size rather than a step: the
# subtitle is part of the heading, not a caption under it.
TITLE_SIZE = 12.0
SUBTITLE_SIZE = 8.0
NOTES_SIZE = 7.5

# Distances in points.
BAR_WIDTH = 2.5
BAR_GAP = 7.0
LINE_GAP = 0.5

# Distances in inches, so a frame keeps its proportions on any figure size.
MARGIN_INCHES = 0.25
MARK_INCHES = 0.24
PLOT_GAP_INCHES = 0.2
TICK_ROOM_INCHES = 0.3
NOTES_GAP_INCHES = 0.04

# Line height as a multiple of the font size. The frame is laid out from the
# sizes alone: measuring the drawn text needs a renderer, which not every
# backend can give before the figure is shown.
LINE_HEIGHT = 1.2

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

# The beta mark: the glyph the application shows at the top left of its menu
# bar and pybeacon.dev in its header, an italic from Playfair Display drawn as
# an outline. Copied from beacon-site's overrides/partials/logo-beta.html,
# itself from beacon-ui's src/renderer/src/icons/svg/logo-beta.svg, in the
# glyph's own coordinates with y up.
BETA_PATH = (
    "M14.4344 33.6673C12.4646 33.1878 10.6078 31.6051 9.60677 "
    "29.5269C8.7349 27.7204 8.20208 26.0418 5.69948 17.4252C4.40781 12.981 "
    "2.95469 7.9932 2.47031 6.31463C1.92135 4.47619 1.27552 2.63776 "
    "0.791146 1.6466L0 0H1.35625H2.72865L3.11615 0.767347C3.64896 1.80646 "
    "4.8599 5.17959 5.39271 7.11395C5.65104 7.9932 5.89323 8.77653 5.97396 "
    "8.85646C6.03854 8.92041 6.4099 8.76054 6.81354 8.48878C9.26771 6.82619 "
    "13.3526 7.36973 15.9844 9.71973C17.0661 10.6789 18.2609 12.5014 "
    "18.7292 13.9242C19.2943 15.6507 19.2781 17.9367 18.6969 "
    "19.1837C18.2448 20.1429 17.2276 21.2139 16.3557 21.6776C16.049 21.8374 "
    "15.8068 22.0133 15.8068 22.0612C15.8068 22.1252 16.2589 22.381 16.8078 "
    "22.6367C18.2286 23.2922 19.5203 24.2833 20.2307 25.2425C21.3609 "
    "26.7612 21.7484 28.8874 21.2156 30.6619C20.4891 33.0759 17.4859 "
    "34.4027 14.4344 33.6673ZM17.5828 32.5643C18.4708 31.9888 18.7937 "
    "31.2854 18.7776 29.9745C18.7453 27.4806 17.1469 24.2034 15.3708 "
    "23.0364C14.6927 22.5888 14.6604 22.5888 13.9984 22.8126C12.4807 "
    "23.3241 11.4474 23.0364 11.6734 22.1412C11.8187 21.5656 12.3677 "
    "21.4058 13.4979 21.5816C14.6766 21.7735 15.0318 21.5497 15.726 "
    "20.2228C16.1781 19.3595 16.2104 19.1517 16.2104 17.681C16.1943 15.7306 "
    "15.9359 14.8194 14.7573 12.3895C14.2083 11.2544 13.6432 10.3752 "
    "13.1911 9.92755C11.738 8.4568 9.8974 8.07313 8.23438 8.88844C7.37865 "
    "9.32007 6.39375 10.3912 6.52292 10.7588C6.5875 10.9507 6.74896 11.4622 "
    "8.94479 19.0238C11.7057 28.4718 12.4646 30.518 13.6594 31.717C14.8542 "
    "32.932 16.5172 33.2837 17.5828 32.5643Z")

# The label on the axes that holds the mark, and the ids on the frame's other
# pieces and on a level chart's reference line, so a caller can find them.
MARK_LABEL = "beacon-mark"
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
    """A figure this module will frame."""
    figure = plt.figure(figsize=size)
    _OWNED.add(figure)

    return figure


def owns(figure: Any) -> bool:
    """Whether a figure was made here, and so gets the whole frame."""
    return isinstance(figure, Figure) and figure in _OWNED


def mode_of(ax: Axes) -> str:
    """Which style an axes is drawn in, read off its own background.

    Charts need a colour the style does not carry, a marker or a sign, and
    hard-coding one would break in the other mode. Asking the axes is more
    reliable than tracking global state, because a caller may have applied a
    style to this figure alone.
    """
    from matplotlib.colors import to_rgb

    figure = ax.get_figure()

    if figure is None:
        return LIGHT

    # Dark only when the background is. Matching the light canvas exactly
    # read matplotlib's own white, before `beacon.plot.use()`, as dark
    # (BN-259).
    red, green, blue = to_rgb(figure.get_facecolor())
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    return DARK if luminance < 0.5 else LIGHT


def ink(ax: Axes,
        name: str) -> str:
    """One token colour, in whichever mode this axes is drawn in."""
    return colour(name, mode_of(ax))


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
           bottom_room: float = TICK_ROOM_INCHES,
           reach: Axes | None = None) -> Axes:
    """Give a chart its title, subtitle, axis label and notes.

    The whole frame when the chart made its own figure; a plain title and the
    notes under the axes when it was drawn on the caller's.

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
        reach: A side panel, such as a colour bar, whose labels the mark
            should end under instead of the x axis.

    Returns:
        Axes: The same axes.
    """
    if ylabel:
        ax.set_ylabel(ylabel)

    figure = ax.get_figure()

    if isinstance(figure, Figure) and owns(figure):
        frame(ax,
              title,
              subtitle,
              notes,
              mode_of(ax),
              bottom_room,
              reach)

        return ax

    register_fonts()
    ax.set_title(title, loc="left", pad=12, fontfamily=TITLE_FONT,
                 fontweight="normal")

    if notes:
        ax.text(0.0, note_offset, notes, transform=ax.transAxes,
                fontsize=7, color=ink(ax, "text-muted"), va="top",
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
          mode: str,
          bottom_room: float = TICK_ROOM_INCHES,
          reach: Axes | None = None) -> None:
    """Draw the title block, the notes and the mark, and fit the plot between.

    The accent bar and the notes start at the y axis line, and the mark ends
    where the x axis does, or under the labels of *reach* when given, so the
    frame lines up with the plot it surrounds.

    Args:
        ax: The chart's axes; for a chart of several panels, the top one.
        title: The heading.
        subtitle: Beneath the heading; left out when empty.
        notes: The notes line's text, without the prefix; left out when empty.
        mode: LIGHT or DARK.
        bottom_room: Inches kept free under the plot.
        reach: A side panel whose labels the mark ends under.
    """
    register_fonts()

    figure = ax.get_figure()
    assert isinstance(figure, Figure)  # finish() frames only figures it owns

    width, height = (float(size) for size in figure.get_size_inches())
    top = height - MARGIN_INCHES

    bottom = top - TITLE_SIZE * LINE_HEIGHT / 72.0
    subtitle_top = bottom - LINE_GAP / 72.0
    if subtitle:
        bottom = subtitle_top - SUBTITLE_SIZE * LINE_HEIGHT / 72.0

    # Placed in two passes. The plot's height depends on how many lines the
    # notes take, their width on where the axes end up, and an axes with a
    # fixed aspect (a heatmap) narrows as it shortens.
    lines: list[str] = []
    for _ in range(2):
        floor = max(MARGIN_INCHES + MARK_INCHES,
                    MARGIN_INCHES + len(lines) * NOTES_SIZE * LINE_HEIGHT / 72.0)
        figure.subplots_adjust(top=(bottom - PLOT_GAP_INCHES) / height,
                               bottom=(floor + NOTES_GAP_INCHES + bottom_room) / height)
        left, right = _plot_edges(ax, width)

        if notes:
            room = right - left - _mark_width() - BAR_GAP / 72.0
            lines = _wrap(f"Notes: {notes}", room)

    if reach is not None:
        right = max(right, _label_edge(reach, figure))

    text_left = left + (BAR_WIDTH + BAR_GAP) / 72.0

    figure.text(text_left / width, top / height, title,
                fontfamily=TITLE_FONT, fontsize=TITLE_SIZE, fontweight="normal",
                color=colour("text-primary", mode), ha="left", va="top",
                gid="beacon-title")

    if subtitle:
        figure.text(text_left / width, subtitle_top / height, subtitle,
                    fontfamily=SUBTITLE_FONT, fontsize=SUBTITLE_SIZE,
                    fontstyle="italic", color=colour("text-secondary", mode),
                    ha="left", va="top", gid="beacon-subtitle")

    # Square-cornered: a Rectangle has no rounding, and with no edge drawn its
    # corners are exactly the box's.
    figure.add_artist(Rectangle((left / width, bottom / height),
                                BAR_WIDTH / 72.0 / width,
                                (top - bottom) / height,
                                transform=figure.transFigure,
                                facecolor=colour("accent", mode),
                                edgecolor="none",
                                linewidth=0))

    # The footer's top: the notes hang from it and the mark sits level with
    # them, so the footer reads as one row under the plot.
    if lines:
        figure.text(left / width, floor / height,
                    "\n".join(lines), fontfamily=NOTES_FONT, fontsize=NOTES_SIZE,
                    fontstyle="italic", color=colour("text-muted", mode),
                    ha="left", va="top", linespacing=LINE_HEIGHT,
                    gid="beacon-notes")

    _draw_mark(figure,
               right,
               floor,
               colour("text-primary", mode))


def _plot_edges(ax: Axes,
                width: float) -> tuple[float, float]:
    """Where the axes' left and right edges fall, in inches from the left.

    Applies the axes' aspect first, so a heatmap reports the box it is drawn
    in rather than the one it was given.
    """
    ax.apply_aspect()
    box = ax.get_position()

    return box.x0 * width, box.x1 * width


def _label_edge(panel: Axes,
                figure: Figure) -> float:
    """Where a panel's labels end on the right, in inches from the left.

    Measured with the figure's renderer, which every raster backend has. A
    canvas that cannot give one gets the panel's own edge, which leaves the
    mark short of the labels rather than past them.
    """
    get_renderer = getattr(figure.canvas, "get_renderer", None)

    if get_renderer is None:
        return float(panel.get_position().x1) * float(figure.get_size_inches()[0])

    box = panel.get_tightbbox(get_renderer())

    if box is None:
        return float(panel.get_position().x1) * float(figure.get_size_inches()[0])

    return float(box.x1) / float(figure.dpi)


def _mark_width() -> float:
    """How wide the mark is drawn, in inches."""
    extent = beta_outline().get_extents()

    return MARK_INCHES * extent.width / extent.height


@functools.cache
def beta_outline() -> Outline:
    """The beta mark as a matplotlib path, parsed once from BETA_PATH."""
    return _parse(BETA_PATH)


def _draw_mark(figure: Figure,
               right: float,
               top: float,
               colour_value: str) -> None:
    """The beta mark in the footer, its right edge where the x axis ends.

    Args:
        figure: The figure to draw on.
        right: Where the x axis ends, in inches from the left.
        top: The footer's top, in inches from the bottom.
        colour_value: The mark's colour.
    """
    width, height = (float(size) for size in figure.get_size_inches())
    outline = beta_outline()
    extent = outline.get_extents()
    mark_width = _mark_width()

    holder = figure.add_axes(((right - mark_width) / width,
                              (top - MARK_INCHES) / height,
                              mark_width / width,
                              MARK_INCHES / height),
                             label=MARK_LABEL)
    holder.set_axis_off()
    holder.add_patch(PathPatch(outline,
                               facecolor=colour_value,
                               edgecolor="none",
                               linewidth=0))
    holder.set_xlim(extent.x0, extent.x1)
    holder.set_ylim(extent.y0, extent.y1)


def _parse(d: str) -> Outline:
    """An SVG path of absolute M, L, H, C and Z commands as a matplotlib path.

    Only those commands, because the mark uses only those. Anything else
    raises rather than drawing a wrong shape.
    """
    tokens = re.findall(r"[A-Za-z]|-?\d*\.?\d+(?:e-?\d+)?", d)
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    start = current = (0.0, 0.0)
    command = ""
    position = 0

    while position < len(tokens):
        if tokens[position].isalpha():
            command = tokens[position]
            position += 1

        arity = {"M": 2, "L": 2, "H": 1, "C": 6, "Z": 0}.get(command)
        if arity is None:
            raise ValueError(f"the mark's path uses {command!r}, which this "
                             f"parser does not read")

        values = [float(token) for token in tokens[position:position + arity]]
        position += arity

        if command == "M":
            current = start = (values[0], values[1])
            vertices.append(current)
            codes.append(int(Outline.MOVETO))
            command = "L"  # further pairs after a move are lines
        elif command == "L":
            current = (values[0], values[1])
            vertices.append(current)
            codes.append(int(Outline.LINETO))
        elif command == "H":
            current = (values[0], current[1])
            vertices.append(current)
            codes.append(int(Outline.LINETO))
        elif command == "C":
            current = (values[4], values[5])
            vertices.extend([(values[0], values[1]), (values[2], values[3]), current])
            codes.extend([int(Outline.CURVE4)] * 3)
        else:
            vertices.append(start)
            codes.append(int(Outline.CLOSEPOLY))
            current = start
            command = ""

    return Outline(vertices, codes)


def _wrap(text: str,
          inches: float) -> list[str]:
    """The notes broken into lines that fit the width they have."""
    characters = max(20, int(inches * 72.0 / (NOTES_SIZE * CHARACTER_WIDTH)))

    return textwrap.wrap(text, characters) or [text]


def _day(when: Any) -> str:
    """A date as dd/mm/yyyy."""
    return str(pd.Timestamp(when).strftime("%d/%m/%Y"))
