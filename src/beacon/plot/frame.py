# src/beacon/plot/frame.py
"""
The furniture around a chart, as matplotlib draws it: the bundled fonts, the
theme a figure is drawn in, and reading a drawn chart's heading and notes
back.

A chart's frame is laid out by beacon.plot.arrange and drawn by
beacon.plot.paint:

* the **title** in Inter Tight and the **subtitle** beneath it in Source
  Serif 4 italic, both left aligned, with a thin square accent bar down their
  left, level with the y axis
* the beta **mark**, the glyph the application shows in its menu bar, just
  left of the bar and centred on it
* one italic **Notes** line at the bottom left: where the data came from, the
  dates it covers, and whatever the chart itself needs said, run together as
  sentences rather than listed

The fonts ship in the package under the SIL Open Font License, whose texts sit
beside them, so a chart looks the same on every machine and the image
regression tests can compare it.
"""
import functools
import logging
from pathlib import Path
from typing import Any

from .._optional import require
from ..tokens import LIGHT

require("matplotlib", "Charting")

from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import design  # noqa: E402
from .charts.lines import REFERENCE_ID  # noqa: E402, F401
from .colours import resolve  # noqa: E402

# MARK_LABEL is re-exported: callers find the mark's axes by it through frame.
from .mark import MARK_LABEL  # noqa: E402, F401
from .notes import SOURCE_NAMES, provenance, sentences  # noqa: E402, F401
from .themes import Theme, for_background, of_figure, theme_named  # noqa: E402

logger = logging.getLogger(__name__)

FONTS = Path(__file__).parent / "fonts"
# Inter Tight regular for the title and light for everything else on the
# chart; Source Serif 4 at 350 for the subtitle, an instance cut from its
# variable font (SIL Open Font License, which permits the cut).
FONT_FILES = ("InterTight-Regular.ttf", "InterTight-Light.ttf",
              "InterTight-LightItalic.ttf", "SourceSerif4-Italic350.ttf")

# The notes' size as designed, in points (the style's notes.size).
NOTES_SIZE = 4.5

# The frame's pieces a caller can read back, by the id each is drawn with.
PARTS = ("title", "subtitle", "notes")


@functools.cache
def register_fonts() -> None:
    """Make the bundled fonts known to matplotlib. Runs once; cached."""
    for name in FONT_FILES:
        font_manager.fontManager.addfont(str(FONTS / name))

    logger.debug("Registered the chart fonts from %s.", FONTS)


def theme_of(ax: Axes) -> Theme:
    """The theme an axes is drawn in.

    The one its figure was drawn in when beacon drew it; otherwise read off
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
    """One colour by name, in whichever theme this axes is drawn in and the
    current style."""
    return resolve(name, theme_of(ax), design.current())


def frame_text(chart: Any,
               part: str) -> str:
    """What a chart's title, subtitle or notes say.

    Args:
        chart: A chart, or the axes of a drawn one.
        part: "title", "subtitle" or "notes".

    Returns:
        str: The text on one line, the notes without their "Notes:" prefix,
        and empty when the chart has none.

    Raises:
        ValueError: If *part* is not one of the three.
    """
    if part not in PARTS:
        raise ValueError(f"part must be one of {', '.join(PARTS)}, not {part!r}")

    ax = chart if isinstance(chart, Axes) else chart.ax
    figure = ax.get_figure()
    texts = [*(figure.texts if isinstance(figure, Figure) else []), *ax.texts]
    found = next((text.get_text() for text in texts
                  if text.get_gid() == f"beacon-{part}"), None)

    if found is None and part == "title":
        found = ax.get_title(loc="left")

    return " ".join((found or "").split()).removeprefix("Notes: ")
