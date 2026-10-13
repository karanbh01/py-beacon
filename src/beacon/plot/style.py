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
import warnings
from typing import Any

from .._optional import require
from ..tokens import DARK, LIGHT, raw_colours
from . import design
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
# same on every chart. A style's size.scale zooms on top of it (BN-303).
SCALE = 0.8


def scaled(value: float) -> float:
    """A designed size at the scale charts are drawn at, in the current
    style."""
    return value * SCALE * design.current().size.scale


def _text_zoom() -> float:
    """How much the chart's small text is scaled: the chart's zoom and the
    text's own scale."""
    current = design.current()

    return current.size.scale * current.text.scale


def series_width() -> float:
    """The leading series' line weight, as drawn."""
    return scaled(design.current().lines.series_width)


def secondary_width() -> float:
    """Every other series' line weight, as drawn."""
    return scaled(design.current().lines.secondary_width)


def reference_width() -> float:
    """A reference line's weight, as drawn."""
    return scaled(design.current().lines.reference_width)


def label_size() -> float:
    """Legend entries and labels on the chart, in points as drawn."""
    return design.current().text.label_size * _text_zoom()


def tick_label_size() -> float:
    """Tick labels, in points as drawn."""
    return design.current().axes.tick_label_size * _text_zoom()


def axis_title_size() -> float:
    """Axis titles, in points as drawn."""
    return design.current().axes.title_size * _text_zoom()


def reference_lines() -> bool:
    """Whether charts draw the lines they are read against."""
    return design.current().axes.reference


def font_family(name: str) -> str:
    """A font's family name: *name* itself, or the family of the font file it
    names (.ttf or .otf), which is registered with matplotlib first."""
    if not name.lower().endswith((".ttf", ".otf")):
        return name

    from matplotlib import font_manager

    font_manager.fontManager.addfont(name)

    return str(font_manager.FontProperties(fname=name).get_name())


# The defaults of the settings below, kept for code that reads them; charts
# read the current style through the functions above (BN-303).
#
# The chart's small text, in points as drawn rather than scaled, so it stays
# readable whatever SCALE is: the axis titles, and the tick labels and legend
# entries, which share a size.
AXIS_TITLE_SIZE = 8.0
TICK_LABEL_SIZE = 7.8

# Labels drawn on the chart itself (the value at a line's end, the names at
# the constituents' ends, the cap, the Sharpe ratio) and the legend's, smaller
# than the axes' so they annotate rather than compete.
LINE_LABEL_SIZE = 5.5

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
    """The size a chart of *kind* is drawn at, in inches, in the current
    style: its designed size, or the style's width (at the same proportions)
    and aspect, zoomed by the style's scale."""
    return sized(*FIGSIZE.get(kind, (6.0, 4.5)))


def sized(width: float,
          height: float) -> tuple[float, float]:
    """A designed size as drawn, in inches, in the current style: the
    style's width (at the same proportions) and aspect where it sets them,
    zoomed by the style's scale."""
    size = design.current().size

    if size.width is not None:
        width, height = size.width, height * size.width / width

    ratio = size.ratio()
    if ratio is not None:
        height = width / ratio

    return scaled(width), scaled(height)


def palette(theme: str | None = None,
            mode: str | None = None) -> dict[str, str]:
    """The colours a chart draws with, in one theme.

    Args:
        theme: A theme's name, or LIGHT or DARK for the Beacon canvases.
            Defaults to LIGHT.
        mode: Deprecated: the parameter's name before themes, which still
            works and warns. Use `theme`.

    Returns:
        dict: The token names this module uses, so a caller composing a custom
        chart can reach the same values rather than sampling them off a figure.
        A transparent theme's canvas is "none".
    """
    chosen = theme_named(_theme_argument(theme, mode, "palette"))

    return {name: chosen.colour(name)
            for name in ("canvas", "surface", "border", "divider", "text-primary",
                         "text-secondary", "text-muted", "accent", "success",
                         "danger", "series-2", "series-3", "period")}


def style_dict(theme: str | None = None,
               mode: str | None = None,
               style: "design.Style | None" = None) -> dict[str, Any]:
    """The rcParams for one theme, in one style.

    Built as a mapping rather than written to an `.mplstyle` file so the values
    stay derived from the tokens. A generated file would be a second copy to
    keep in step, which is the thing this module exists to avoid.

    Args:
        theme: A theme's name, or LIGHT or DARK for the Beacon canvases.
            Defaults to the style's theme.
        mode: Deprecated: the parameter's name before themes, which still
            works and warns. Use `theme`.
        style: The settings; Beacon's defaults when None.

    Returns:
        dict: matplotlib rcParams names to values.
    """
    chosen = style if style is not None else design.Style()
    if theme is None and mode is None:
        theme = chosen.theme
    name = _theme_argument(theme, mode, "style_dict")
    ink = palette(name)
    # A style's colours replace the theme's where it sets them.
    colours = chosen.colors
    for token, value in (("text-primary", colours.text), ("text-secondary", colours.muted),
                         ("canvas", colours.background), ("success", colours.up),
                         ("danger", colours.down)):
        if value is not None:
            ink[token] = value
    series = colours.series or theme_named(name).series()
    k = SCALE * chosen.size.scale
    zoom = chosen.size.scale * chosen.text.scale
    axes = chosen.axes

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
        "axes.linewidth": chosen.lines.axis_width * k,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": axes.y_line,
        "axes.spines.bottom": axes.x_line,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": axes.title_size * zoom,
        "axes.prop_cycle": _cycler(series),

        "grid.color": ink["divider"],
        "grid.linewidth": chosen.lines.grid_width * k,
        "grid.alpha": 1.0,

        "lines.linewidth": chosen.lines.series_width * k,
        "lines.solid_capstyle": "round",

        "text.color": ink["text-primary"],
        "xtick.color": ink["text-primary"],
        "ytick.color": ink["text-primary"],
        "xtick.labelcolor": ink["text-primary"],
        "ytick.labelcolor": ink["text-primary"],
        "xtick.labelsize": axes.tick_label_size * zoom,
        "ytick.labelsize": axes.tick_label_size * zoom,
        "xtick.major.size": axes.tick_length * k if axes.x_ticks else 0.0,
        "ytick.major.size": axes.tick_length * k if axes.y_ticks else 0.0,
        "xtick.major.pad": 3.5 * k,
        "ytick.major.pad": 3.5 * k,
        "axes.labelpad": 4.0 * k,
        "lines.markersize": 6.0 * k,
        # Inward, and on the bottom and left only. Pinned rather than left to
        # matplotlib's defaults, which another style (pytest-mpl applies
        # "classic") can turn on for all four sides.
        "xtick.direction": axes.tick_direction,
        "ytick.direction": axes.tick_direction,
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.width": chosen.lines.axis_width * k,
        "ytick.major.width": chosen.lines.axis_width * k,

        # Pinned for the same reason as the ticks: "classic" outlines every
        # bar in black, doubles a legend's markers and blurs a heatmap.
        "patch.force_edgecolor": False,
        "patch.linewidth": 0.0,
        "image.interpolation": "nearest",
        "legend.numpoints": 1,
        "legend.scatterpoints": 1,

        "legend.frameon": False,
        "legend.fontsize": chosen.text.label_size * zoom,
        "legend.labelcolor": ink["text-primary"],

        # No padding beyond the data: each measuring axis is then ended on its
        # outermost ticks (beacon.plot.frame.end_on_ticks).
        # Dates as the year at each January and the month between, so a
        # two-year axis reads "2023, Apr, Jul, Oct, 2024" rather than
        # crowding nine full dates.
        "date.converter": "concise",

        "axes.xmargin": 0.0,
        "axes.ymargin": 0.0,

        "font.size": 9 * k,
        "font.family": "sans-serif",
        "font.weight": "light",
        # Inter Tight ships with py-beacon (beacon/plot/fonts), so a chart
        # looks the same on every machine and image regression can compare
        # it; DejaVu, which ships with matplotlib, is the fallback for a glyph
        # Inter Tight lacks.
        "font.sans-serif": list(dict.fromkeys([font_family(chosen.text.font), "Inter Tight",
                                                "DejaVu Sans"])),
    }


def _cycler(series: tuple[str, ...]) -> Any:
    """The series colour order: the style's series colours, or the theme's
    palette (beacon.plot.themes.SERIES)."""
    from cycler import cycler

    return cycler(color=list(series))


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


# The design version last applied to matplotlib's rcParams (BN-303).
_APPLIED = {"version": design.version()}


def apply(force: bool = False) -> None:
    """Put the current style into matplotlib's rcParams, when it has changed
    since it was last applied.

    Charts draw in whatever rcParams are active, so nothing is applied until a
    style is chosen: `configure()` takes effect at the next chart, and a user
    who never calls it keeps the rcParams they set themselves.
    """
    if not force and design.version() == _APPLIED["version"]:
        return

    register()

    import matplotlib.pyplot as plt

    current = design.current()
    plt.style.use(style_dict(current.theme, style=current))
    set_current(current.theme)
    _APPLIED["version"] = design.version()


def use(theme: str | None = None,
        mode: str | None = None) -> None:
    """Apply a beacon theme to every chart drawn afterwards: a shortcut for
    `configure(theme=...)`, which changes nothing else in the style.

    Args:
        theme: "beacon-light" (or LIGHT, the default), "beacon-dark" (or
            DARK), "github-dark", "white", "transparent-dark-axes" or
            "transparent-light-axes".
        mode: Deprecated: the parameter's name before themes, which still
            works and warns. Use `theme`.

    Raises:
        ValueError: If there is no such theme.
        TypeError: If both `theme` and `mode` are given.
    """
    chosen = theme_named(_theme_argument(theme, mode, "use"))
    design.configure(theme=chosen.name)
    apply(force=True)


def _theme_argument(theme: str | None,
                    mode: str | None,
                    function: str) -> str:
    """The theme asked for, by its own name or by the deprecated `mode`."""
    # BN-296: `mode` was the parameter's name until themes (BN-289) replaced
    # the light and dark modes; it keeps working, with a warning, so code
    # that named it is not broken by the rename.
    if mode is None:
        return LIGHT if theme is None else theme

    if theme is not None:
        raise TypeError(f"{function}() takes theme or mode, not both; mode is "
                        f"the deprecated name of theme")

    warnings.warn(f"{function}(mode=...) is deprecated; use "
                  f"{function}(theme=...), which also takes the new themes.",
                  DeprecationWarning, stacklevel=3)

    return mode
