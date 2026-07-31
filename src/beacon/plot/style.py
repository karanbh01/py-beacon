# src/beacon/plot/style.py
"""
The "beacon" matplotlib styles, generated from the design tokens.

Not hand-written: every colour comes from `beacon.tokens`, which is a vendored
copy of the file the desktop client generates its CSS from. That is the point —
a chart embedded in the application should be the same colours as the panel
around it, and the only way to guarantee that is for both to read one source.
A palette retyped here would be right on the day it was written.

Two styles, `beacon` and `beacon-dark`, because the client has two modes and a
chart has to follow. Registering both by name means
`plt.style.use("beacon")` restyles any plot, including one this library never
drew — which is the acceptance criterion, and also the honest test of whether
the style is a style rather than a set of hard-coded arguments.

## What the grammar is

* an **accent** line at 1.5pt for the primary series, **text-secondary** for a
  benchmark: the comparison should read as subordinate without being hidden
* **divider** gridlines on the y axis only, and behind the data
* **muted** axis labels and tick text, with the top and right spines removed —
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
from ..tokens import DARK, LIGHT, MODES, colour, raw_colours

logger = logging.getLogger(__name__)

# Style names registered with matplotlib.
BEACON = "beacon"
BEACON_DARK = "beacon-dark"
STYLE_FOR_MODE = {LIGHT: BEACON, DARK: BEACON_DARK}

# The diverging-to-hot colormap for correlations.
CORRELATION_COLORMAP = "beacon_corr"

# Correlation is shown from 0.2 upward: below that the distinction between
# 0.05 and 0.1 is noise on any real estimate, and giving it a fifth of the
# colour range implies a precision the number does not have.
CORRELATION_DOMAIN = (0.2, 1.0)

# Line weights, in points. The primary series is deliberately heavier than the
# grid and the benchmark so the eye lands on it first.
SERIES_WIDTH = 1.5
BENCHMARK_WIDTH = 1.1
GRID_WIDTH = 0.6
SPINE_WIDTH = 0.8

# Per-kind figure sizes, in inches. A level chart is wide because time is the
# long axis; a weights chart is tall because names stack.
FIGSIZE = {
    "level": (9.0, 4.5),
    "performance": (9.0, 6.0),
    "annual_returns": (9.0, 4.0),
    "weights": (7.5, 5.5),
    "contributions": (8.0, 5.5),
    "compare": (9.0, 5.0),
    "frontier": (7.5, 5.5),
    "exposures": (7.5, 4.5),
    "correlation": (6.5, 5.5),
}


def palette(mode: str = LIGHT) -> dict[str, str]:
    """The colours a chart draws with, in one mode.

    Returns:
        dict: The token names this module uses, so a caller composing a custom
        chart can reach the same values rather than sampling them off a figure.
    """
    return {name: colour(name, mode)
            for name in ("canvas", "surface", "border", "divider", "text-primary",
                         "text-secondary", "text-muted", "accent", "success",
                         "danger", "series-2", "series-3")}


def style_dict(mode: str = LIGHT) -> dict[str, Any]:
    """The rcParams for one mode.

    Built as a mapping rather than written to an `.mplstyle` file so the values
    stay derived from the tokens at import time. A generated file would be a
    second copy to keep in step, which is the thing this module exists to
    avoid.
    """
    ink = palette(mode)

    return {
        "figure.facecolor": ink["canvas"],
        "figure.edgecolor": ink["canvas"],
        "figure.dpi": 100,
        "savefig.facecolor": ink["canvas"],
        "savefig.edgecolor": ink["canvas"],
        # Deliberately NOT "tight". A tight bounding box crops to the actual
        # text extents, which depend on the platform's font rasteriser, so the
        # saved image changes size between operating systems — and an image
        # comparison that cannot agree on dimensions cannot compare anything.
        # Room for the footnotes is reserved in the layout instead, below.
        "savefig.bbox": "standard",

        # Space for the title above and the footnote below, so both sit inside
        # a canvas whose size is fixed by figsize alone.
        "figure.subplot.top": 0.90,
        "figure.subplot.bottom": 0.18,
        "figure.subplot.left": 0.10,
        "figure.subplot.right": 0.96,

        "axes.facecolor": ink["canvas"],
        "axes.edgecolor": ink["border"],
        "axes.labelcolor": ink["text-muted"],
        "axes.titlecolor": ink["text-primary"],
        "axes.linewidth": SPINE_WIDTH,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.prop_cycle": _cycler(ink),

        "grid.color": ink["divider"],
        "grid.linewidth": GRID_WIDTH,
        "grid.alpha": 1.0,

        "lines.linewidth": SERIES_WIDTH,
        "lines.solid_capstyle": "round",

        "text.color": ink["text-primary"],
        "xtick.color": ink["text-muted"],
        "ytick.color": ink["text-muted"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": ink["text-secondary"],

        "font.size": 9,
        "font.family": "sans-serif",
        # DejaVu ships with matplotlib, so a chart looks the same on a machine
        # with no fonts installed as on a designer's. Falling through to
        # whatever the system has would make image regression meaningless.
        "font.sans-serif": ["DejaVu Sans"],
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

    # The submodule is imported explicitly, which also binds `matplotlib`
    # itself. Importing the package alone does not bind `matplotlib.style`; it
    # is merely present once pyplot has pulled it in, which made an earlier
    # version of this work locally and fail on every CI runner.
    import matplotlib.style

    # matplotlib types the library's values as RcParams, but it accepts and
    # stores a plain mapping — `plt.style.use` takes either. The cast keeps the
    # declaration honest without pretending to build an RcParams here.
    library: dict[str, Any] = matplotlib.style.library

    for mode in MODES:
        library[STYLE_FOR_MODE[mode]] = style_dict(mode)

    # `available` is a cached list rather than a view over the library, so it
    # needs rebuilding or `plt.style.available` reports the styles missing
    # while `use()` finds them.
    matplotlib.style.available[:] = sorted(matplotlib.style.library)

    if CORRELATION_COLORMAP not in matplotlib.colormaps:
        matplotlib.colormaps.register(correlation_colormap(),
                                      name=CORRELATION_COLORMAP)

    logger.debug("Registered the beacon styles and the correlation colormap.")


def use(mode: str = LIGHT) -> None:
    """Apply a beacon style globally.

    Args:
        mode: LIGHT or DARK.
    """
    register()

    import matplotlib.pyplot as plt

    plt.style.use(STYLE_FOR_MODE[mode])
