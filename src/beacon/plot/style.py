# src/beacon/plot/style.py
"""
The "beacon" matplotlib styles, generated from the design tokens.

Not hand-written: every colour comes from `beacon.tokens`, which is a vendored
copy of the file the desktop client generates its CSS from. That is the point:
a chart embedded in the application should be the same colours as the panel
around it, and the only way to guarantee that is for both to read one source.
A palette retyped here would be right on the day it was written.

Two styles, `beacon` and `beacon-dark`, because the client has two modes and a
chart has to follow. Both are registered with matplotlib by name, so once
:func:`register` has run (every chart method, :func:`use` and
`beacon.plot.compare` run it) `plt.style.use("beacon")` restyles any plot,
including one this library never drew.

## What the grammar is

* an **accent** line at 1.5pt for the primary series, **text-secondary** for a
  benchmark: the comparison should read as subordinate without being hidden
* **divider** gridlines on the y axis only, and behind the data
* **muted** axis labels and tick text, with the top and right spines removed:
  a chart is mostly data, and the frame is not the data

## The correlation colormap

`beacon_corr` is registered here too, from the `raw.heatmap-*` tokens.
Mode-independent by design: a correlation of 0.8 must look the same whichever
theme the surrounding application is wearing, or two screenshots of the same
matrix disagree.
"""
import logging
from typing import Any

from .._optional import require
from ..tokens import DARK, LIGHT, raw_colours
from .themes import THEMES, set_current, theme_named

logger = logging.getLogger(__name__)

# Style names registered with matplotlib: one per theme (beacon.plot.themes),
# of which these two are the Beacon light and dark canvases.
BEACON = "beacon"
BEACON_DARK = "beacon-dark"
STYLE_FOR_MODE = {LIGHT: BEACON, DARK: BEACON_DARK}

# The diverging-to-hot colormap for correlations.
CORRELATION_COLORMAP = "beacon_corr"

# Correlation is shown from 0.2 upward: below that the distinction between
# 0.05 and 0.1 is noise on any real estimate, and giving it a fifth of the
# colour range implies a precision the number does not have.
CORRELATION_DOMAIN = (0.2, 1.0)

# Every chart but its heading is drawn at this share of its designed size:
# the figure, its text, its lines, its marks and its footer. The heading
# (title, subtitle, accent bar and mark) keeps its own size, so it reads the
# same on every chart.
SCALE = 0.8


def scaled(value: float) -> float:
    """A designed size at the scale charts are drawn at."""
    return value * SCALE


# The chart's small text, in points as drawn rather than scaled, so it stays
# readable whatever SCALE is: the axis titles, and the tick labels and legend
# entries, which share a size.
AXIS_TITLE_SIZE = 5.5
TICK_LABEL_SIZE = 5.3

# Line weights, in points. The primary series is deliberately heavier than the
# grid and the benchmark so the eye lands on it first.
SERIES_WIDTH = scaled(1.5)
BENCHMARK_WIDTH = scaled(1.1)
GRID_WIDTH = scaled(0.6)
SPINE_WIDTH = scaled(0.6)

# The lines a chart is read against: a level chart's line at 100, a bar
# chart's zero. Lighter than the axes, so they read as guides, not frame.
REFERENCE_WIDTH = scaled(0.5)

# Designed figure sizes, in inches, before SCALE. Every chart 6in wide; the
# time series, annual returns and the bar charts of names at 6 by 4.5. The
# performance chart's upper panel matches the level chart's plot, and its
# extra height carries the drawdown panel. The frontier and the correlation
# heatmap are taller, for their square-ish shapes.
FIGSIZE = {
    "level": (6.0, 4.5),
    "constituents": (6.0, 4.5),
    "performance": (6.0, 5.75),
    "annual_returns": (6.0, 4.5),
    "weights": (6.0, 4.5),
    "contributions": (6.0, 4.5),
    "compare": (6.0, 4.5),
    "frontier": (6.0, 5.5),
    "exposures": (6.0, 4.5),
    "correlation": (6.0, 5.5),
}


def figure_size(kind: str) -> tuple[float, float]:
    """The size a chart of *kind* is drawn at, in inches."""
    width, height = FIGSIZE.get(kind, (6.0, 4.5))

    return scaled(width), scaled(height)


def palette(theme: str = LIGHT) -> dict[str, str]:
    """The colours a chart draws with, in one theme.

    Args:
        theme: A theme's name, or LIGHT or DARK for the Beacon canvases.

    Returns:
        dict: The token names this module uses, so a caller composing a custom
        chart can reach the same values rather than sampling them off a figure.
        A transparent theme's canvas is "none".
    """
    chosen = theme_named(theme)

    return {name: chosen.colour(name)
            for name in ("canvas", "surface", "border", "divider", "text-primary",
                         "text-secondary", "text-muted", "accent", "success",
                         "danger", "series-2", "series-3")}


def style_dict(theme: str = LIGHT) -> dict[str, Any]:
    """The rcParams for one theme.

    Built as a mapping rather than written to an `.mplstyle` file so the values
    stay derived from the tokens. A generated file would be a second copy to
    keep in step, which is the thing this module exists to avoid.

    Args:
        theme: A theme's name, or LIGHT or DARK for the Beacon canvases.

    Returns:
        dict: matplotlib rcParams names to values.
    """
    ink = palette(theme)

    return {
        "figure.facecolor": ink["canvas"],
        "figure.edgecolor": ink["canvas"],
        "figure.dpi": 100,
        "savefig.facecolor": "auto",
        "savefig.edgecolor": "auto",
        # Deliberately NOT "tight". A tight bounding box crops to the actual
        # text extents, which depend on the platform's font rasteriser, so the
        # saved image changes size between operating systems — and an image
        # comparison that cannot agree on dimensions cannot compare anything.
        # Room for the notes is reserved in the layout instead, below.
        "savefig.bbox": "standard",

        # Space for the title above and the notes below, so both sit inside
        # a canvas whose size is fixed by figsize alone.
        "figure.subplot.top": 0.90,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.left": 0.10,
        # Room on the right for the last value marked beside a line's end.
        "figure.subplot.right": 0.92,

        "axes.facecolor": ink["canvas"],
        # The axis lines in the muted text colour rather than the border, so
        # the frame of the plot reads clearly against the canvas.
        "axes.edgecolor": ink["text-primary"],
        # The chart's own words (axis titles, tick labels, legends) in the
        # title's colour and a light weight; the tick marks and axis lines
        # stay muted.
        "axes.labelcolor": ink["text-primary"],
        "axes.labelweight": "light",
        "axes.titlecolor": ink["text-primary"],
        "axes.linewidth": SPINE_WIDTH,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": AXIS_TITLE_SIZE,
        "axes.prop_cycle": _cycler(ink),

        "grid.color": ink["divider"],
        "grid.linewidth": GRID_WIDTH,
        "grid.alpha": 1.0,

        "lines.linewidth": SERIES_WIDTH,
        "lines.solid_capstyle": "round",

        "text.color": ink["text-primary"],
        "xtick.color": ink["text-primary"],
        "ytick.color": ink["text-primary"],
        "xtick.labelcolor": ink["text-primary"],
        "ytick.labelcolor": ink["text-primary"],
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "xtick.major.size": scaled(3.5),
        "ytick.major.size": scaled(3.5),
        "xtick.major.pad": scaled(3.5),
        "ytick.major.pad": scaled(3.5),
        "axes.labelpad": scaled(4.0),
        "lines.markersize": scaled(6.0),
        # Inward, and on the bottom and left only. Pinned rather than left to
        # matplotlib's defaults, which another style (pytest-mpl applies
        # "classic") can turn on for all four sides.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": SPINE_WIDTH,
        "ytick.major.width": SPINE_WIDTH,

        # Pinned for the same reason as the ticks: "classic" outlines every
        # bar in black, doubles a legend's markers and blurs a heatmap.
        "patch.force_edgecolor": False,
        "patch.linewidth": 0.0,
        "image.interpolation": "nearest",
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,

        "legend.frameon": False,
        "legend.fontsize": TICK_LABEL_SIZE,
        "legend.labelcolor": ink["text-primary"],

        # No padding beyond the data: each measuring axis is then ended on its
        # outermost ticks (beacon.plot.frame.end_on_ticks).
        "axes.xmargin": 0.0,
        "axes.ymargin": 0.0,

        "font.size": scaled(9),
        "font.family": "sans-serif",
        "font.weight": "light",
        # Inter Tight ships with py-beacon (beacon/plot/fonts), so a chart
        # looks the same on every machine and image regression can compare
        # it; DejaVu, which ships with matplotlib, is the fallback for a glyph
        # Inter Tight lacks.
        "font.sans-serif": ["Inter Tight", "DejaVu Sans"],
    }


def _cycler(ink: dict[str, str]) -> Any:
    """The series colour order.

    Accent first, then the two colours added for compare lines. Success and
    danger are deliberately absent: green and red mean up and down elsewhere in
    the application, and a third series that happened to be green would read as
    a gain rather than as a series.
    """
    from cycler import cycler

    return cycler(color=[ink["accent"], ink["series-2"], ink["series-3"],
                         ink["text-secondary"]])


def correlation_colormap() -> Any:
    """The `beacon_corr` colormap, green through amber to red.

    Built from the mode-independent `raw.heatmap-*` tokens: a correlation of
    0.8 must look the same whichever theme the application is wearing, or two
    screenshots of one matrix disagree.

    Returns:
        Colormap: Registered under CORRELATION_COLORMAP.
    """
    require("matplotlib", "Charting")

    from matplotlib.colors import LinearSegmentedColormap

    raw = raw_colours()

    return LinearSegmentedColormap.from_list(
        CORRELATION_COLORMAP,
        [raw["heatmap-low"], raw["heatmap-mid"], raw["heatmap-high"]])


def register() -> None:
    """Register both styles and the colormap with matplotlib.

    Idempotent, and called on first use of any accessor, so
    `plt.style.use("beacon")` works as soon as anything in this package has
    been touched. Registering at import of `beacon.plot` instead would mean
    importing matplotlib to do it, which is exactly what the lazy accessor
    exists to avoid.
    """
    require("matplotlib", "Charting")

    from .frame import register_fonts

    register_fonts()

    # The submodule is imported explicitly, which also binds `matplotlib`
    # itself. Importing the package alone does not bind `matplotlib.style`; it
    # is merely present once pyplot has pulled it in, which made an earlier
    # version of this work locally and fail on every CI runner.
    import matplotlib.style

    # matplotlib types the library's values as RcParams, but it accepts and
    # stores a plain mapping — `plt.style.use` takes either. The cast keeps the
    # declaration honest without pretending to build an RcParams here.
    library: dict[str, Any] = matplotlib.style.library

    for theme in THEMES.values():
        library[theme.style] = style_dict(theme.name)

    # `available` is a cached list rather than a view over the library, so it
    # needs rebuilding or `plt.style.available` reports the styles missing
    # while `use()` finds them.
    matplotlib.style.available[:] = sorted(matplotlib.style.library)

    if CORRELATION_COLORMAP not in matplotlib.colormaps:
        matplotlib.colormaps.register(correlation_colormap(),
                                      name=CORRELATION_COLORMAP)

    logger.debug("Registered the beacon styles and the correlation colormap.")


def use(theme: str = LIGHT) -> None:
    """Apply a beacon theme to every chart drawn afterwards.

    Args:
        theme: "beacon-light" (or LIGHT), "beacon-dark" (or DARK),
            "github-dark", "white", "transparent-dark-axes" or
            "transparent-light-axes".

    Raises:
        ValueError: If there is no such theme.
    """
    chosen = theme_named(theme)
    register()

    import matplotlib.pyplot as plt

    plt.style.use(chosen.style)
    set_current(chosen.name)
