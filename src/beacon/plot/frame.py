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

from . import design  # noqa: E402
from .arrange import TICK_ROOM_INCHES, lay_out  # noqa: E402
from .fitting import lay_out_panel  # noqa: E402

# MARK_LABEL is re-exported: callers find the mark's axes by it through frame.
from .mark import MARK_LABEL  # noqa: E402
from .render import apply_grid, describe, describe_panel, draw, draw_axes  # noqa: E402
from .style import SCALE, apply, font_family  # noqa: E402
from .themes import Theme, for_background, of_figure, remember, theme_named  # noqa: E402

logger = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "fonts"
# Inter Tight regular for the title and light for everything else on the
# chart; Source Serif 4 at 350 for the subtitle, an instance cut from its
# variable font (SIL Open Font License, which permits the cut).
FONT_FILES = ("InterTight-Regular.ttf", "InterTight-Light.ttf",
              "InterTight-LightItalic.ttf", "SourceSerif4-Italic350.ttf")

# The notes' size as designed, in points (the style's notes.size).
NOTES_SIZE = 4.5

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

    end_axes(ax)
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

    The chart is described for the layout (beacon.plot.arrange), laid out in
    the current style, and drawn back from the layout, so every renderer
    places the frame the same way.

    Args:
        ax: The chart's axes; for a chart of several panels, the top one.
        title: The heading.
        subtitle: Beneath the heading; left out when empty.
        notes: The notes line's text, without the prefix; left out when empty.
        theme: The theme the chart is drawn in (its colours are resolved from
            the axes' own).
        bottom_room: Inches kept free under the plot.
    """
    register_fonts()

    figure = ax.get_figure()
    assert isinstance(figure, Figure)  # finish() frames only figures it owns

    chart, panels = describe(ax, title, subtitle, notes, bottom_room)
    draw(lay_out(chart, design.current()), panels, figure)


def end_axes(ax: Axes) -> None:
    """End an axes' measuring axes on their outermost ticks, at the size it
    has on a figure the caller laid out, and blank the lowest y label where
    both measure."""
    if ax.get_label() == MARK_LABEL or hasattr(ax, "_colorbar"):
        return

    panel, box, size = describe_panel(ax)
    draw_axes(ax, lay_out_panel(panel, box, design.current(), size))


def _day(when: Any) -> str:
    """A date as dd/mm/yyyy."""
    return str(pd.Timestamp(when).strftime("%d/%m/%Y"))
