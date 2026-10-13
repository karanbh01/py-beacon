# src/beacon/plot/render.py
"""
The matplotlib renderer: a chart's figure described for the layout, and the
layout drawn back onto it.

The chart methods draw their data on matplotlib axes as they always have.
`describe()` reads those axes into the `Chart` the layout needs (each axis's
range and how it writes its ticks, the labels on the plot, the legend's
entries), `beacon.plot.arrange.lay_out()` lays it out, and `draw()` puts the
axes where the layout says, with its ranges, ticks and labels, and draws the
heading, the legend and the notes.
"""
# BN-298.
from typing import Any

from .._optional import require
from . import design
from .layout import (
    Axis,
    AxisKind,
    AxisLayout,
    Box,
    Chart,
    ColourBar,
    Key,
    Label,
    Layout,
    LegendItem,
    Panel,
    PanelLayout,
    Text,
)
from .metrics import Font, weight_of
from .ticks import STEPS, Spacing

require("matplotlib", "Charting")

import matplotlib  # noqa: E402
from matplotlib import dates, font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.category import StrCategoryConverter  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FixedLocator,
    Formatter,
    FuncFormatter,
    MaxNLocator,
    PercentFormatter,
    ScalarFormatter,
)

from .legend import COLUMN_GAP, HANDLE_GAP, HANDLE_LENGTH, ROW_HEIGHT  # noqa: E402
from .mark import MARK_LABEL, draw_mark  # noqa: E402
from .style import font_family  # noqa: E402
from .themes import Theme  # noqa: E402


def panels_of(figure: Figure) -> list[Axes]:
    """The axes a chart drew its data on, top to bottom: not the mark's, and
    not a colour bar's."""
    found = [axes for axes in figure.get_axes()
             if axes.get_label() != MARK_LABEL and not hasattr(axes, "_colorbar")]

    return sorted(found, key=lambda axes: -axes.get_position(original=True).y1)


def describe(ax: Axes,
             title: str,
             subtitle: str,
             notes: str,
             bottom_room: float | None) -> tuple[Chart, list[Axes]]:
    """The chart a figure holds, as the layout needs it.

    Returns:
        tuple: The chart, and its panels' axes in the layout's order.
    """
    figure = ax.get_figure()
    assert isinstance(figure, Figure)
    panels = panels_of(figure)
    described = []

    for index, axes in enumerate(panels):
        shared = next((other for other in range(index)
                       if axes.get_shared_x_axes().joined(axes, panels[other])), None)
        described.append(Panel(x=_axis(axes.xaxis, shared),
                               y=_axis(axes.yaxis, None),
                               weight=_weight(axes),
                               equal=axes.get_aspect() == 1.0,
                               labels=tuple(_labels(axes)),
                               colour_bar=_colour_bar(figure, axes)))

    width, height = (float(size) for size in figure.get_size_inches())

    return Chart(size=(width, height), title=title, subtitle=subtitle, notes=notes,
                 panels=tuple(described), gap=_gap(panels[0]),
                 legend=tuple(_legend(ax)), bottom_room=bottom_room), panels


def describe_panel(ax: Axes) -> tuple[Panel, Box, float]:
    """One axes on a figure the caller laid out, the box it fills, and its
    tick labels' size."""
    figure = ax.get_figure()
    assert isinstance(figure, Figure)
    width, height = (float(size) for size in figure.get_size_inches())
    box = ax.get_position()

    return (Panel(x=_axis(ax.xaxis, None), y=_axis(ax.yaxis, None)),
            Box(box.x0 * width, box.y0 * height, box.width * width, box.height * height),
            _tick_label_size(ax.yaxis))


def _tick_label_size(axis: Any) -> float:
    """An axis's tick labels' size, in points."""
    size = axis.get_tick_params(which="major").get(
        "labelsize", matplotlib.rcParams[f"{axis.axis_name}tick.labelsize"])

    if isinstance(size, str):
        return float(font_manager.FontProperties(size=size).get_size_in_points())

    return float(size)


def _cell(axes: Axes) -> Any:
    """The figure's own grid cell an axes sits in, beyond any grid a colour
    bar divided it into."""
    spec = axes.get_subplotspec()

    return spec.get_topmost_subplotspec() if spec is not None else None


def _weight(axes: Axes) -> float:
    spec = _cell(axes)
    if spec is None:
        return 1.0

    ratios = spec.get_gridspec().get_height_ratios()

    return float(ratios[spec.rowspan.start]) if ratios is not None else 1.0


def _gap(axes: Axes) -> float:
    spec = _cell(axes)
    figure = axes.get_figure()
    assert isinstance(figure, Figure)
    space = spec.get_gridspec().hspace if spec is not None else None

    return float(space if space is not None else figure.subplotpars.hspace)


def _axis(axis: Any,
          shared: int | None) -> Axis:
    """An axis as the layout sees it."""
    locator, formatter = axis.get_major_locator(), axis.get_major_formatter()
    if isinstance(formatter, LaidOutFormatter):
        # Laid out before: read it as the chart drew it.
        locator, formatter = formatter.locator, formatter.inner
    low, high = (float(value) for value in axis.get_view_interval())
    inverted = bool(axis.get_inverted())
    low, high = sorted((low, high))
    first = next(iter(axis.get_majorticklabels()), None)
    keywords = axis._major_tick_kw  # the tick_params the chart set
    common: dict[str, Any] = {
        "title": axis.get_label().get_text(),
        "show_labels": bool(keywords.get("label1On", True)),
        "rotation": float(first.get_rotation()) if first is not None else 0.0,
        "align": first.get_ha() if first is not None else "center",
        "inverted": inverted,
        "tick_length": keywords.get("size"),
        "shared": shared,
    }

    if isinstance(axis.get_converter(), StrCategoryConverter) or isinstance(locator, FixedLocator):
        values = [float(value) for value in locator()]
        kind: AxisKind = ("names" if isinstance(axis.get_converter(), StrCategoryConverter)
                          else "fixed")
        return Axis(kind, low, high, ticks=tuple(values),
                    names=tuple(formatter.format_ticks(values)), **common)

    if isinstance(locator, dates.DateLocator):
        return Axis("date", low, high, **common)

    return Axis("number", low, high, labels=_label_style(formatter),
                spacing=_spacing(locator), **common)


def _spacing(locator: Any) -> Spacing:
    if isinstance(locator, MaxNLocator):
        # matplotlib keeps a locator's settings private.
        params = vars(locator)
        bins = None if params["_nbins"] == "auto" else int(params["_nbins"])
        return Spacing(bins=bins, steps=tuple(float(step) for step in params["_steps"]),
                       least=int(params["_min_n_ticks"]))

    return Spacing(steps=STEPS)


def _label_style(formatter: Formatter) -> Any:
    if isinstance(formatter, PercentFormatter) and formatter.xmax == 1.0 \
            and formatter.decimals is None and formatter.symbol == "%":
        return "percent"

    if isinstance(formatter, ScalarFormatter):
        return "number"

    if isinstance(formatter, FuncFormatter):
        function = formatter.func
        return lambda value: str(function(value, None))

    return lambda value: str(formatter(value, None))


def _font(text: Any) -> Font:
    """The face a piece of matplotlib text is set in, as the layout names it:
    the style's font file where the text is in that font."""
    family = text.get_fontfamily()[0]
    if family in ("sans-serif", "serif", "monospace"):
        family = matplotlib.rcParams[f"font.{family}"][0]

    chosen = design.current().text.font
    if chosen.lower().endswith((".ttf", ".otf")) and family == font_family(chosen):
        family = chosen

    return Font(family, weight_of(text.get_fontweight()), text.get_fontstyle() != "normal")


def _labels(axes: Axes) -> list[Label]:
    """The annotations a chart wrote on a panel."""
    found = []

    for text in axes.texts:
        if not isinstance(text, Annotation) or text.anncoords != "offset points":
            continue
        fraction = text.xycoords == "axes fraction"
        x, y = text.xy
        if not fraction:
            # convert_units is untyped in matplotlib's stubs.
            x_axis: Any = axes.xaxis
            y_axis: Any = axes.yaxis
            x, y = x_axis.convert_units(x), y_axis.convert_units(y)
        dx, dy = text.xyann
        clip = text.get_annotation_clip()
        found.append(Label(text.get_text(), float(x), float(y), float(dx), float(dy),
                           float(text.get_fontsize()), _font(text),
                           text.get_horizontalalignment(), text.get_verticalalignment(),
                           float(text.get_rotation()), fraction,
                           clip=bool(clip or (clip is None and text.xycoords == "data"))))

    return found


def _colour_bar(figure: Figure,
                axes: Axes) -> ColourBar | None:
    """The colour bar a panel's image was given, if any."""
    other = _colour_bar_of(figure, axes)
    if other is None:
        return None

    # A colour bar's settings are private to matplotlib.
    bar: Any = vars(other)["_colorbar"]
    info = vars(other).get("_colorbar_info", {})

    return ColourBar(low=float(bar.vmin), high=float(bar.vmax),
                     fraction=float(info.get("fraction", 0.15)),
                     pad=float(info.get("pad", 0.05)),
                     aspect=float(info.get("aspect", 20.0)),
                     labels=tuple(_labels(other)))


def _legend(ax: Axes) -> list[LegendItem]:
    """The legend's entries, with their colours by name."""
    legend = ax.get_legend()
    if legend is None:
        return []

    theme = _theme(ax)
    items = []

    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
        key: Key
        if isinstance(handle, Line2D):
            key = ("marker" if handle.get_linestyle() in ("None", "none", "")
                   else "dashed" if handle.get_linestyle() in ("--", "dashed") else "line")
            colour = handle.get_color()
        else:
            key, colour = "patch", handle.get_facecolor() if isinstance(handle, Patch) else "none"
        items.append(LegendItem(text.get_text(), _name_of(colour, theme), key))

    return items


def _theme(ax: Axes) -> Theme:
    from .frame import theme_of

    return theme_of(ax)


# The tokens a colour is looked for among, so a legend keeps its colours by
# name: the frame's own, then the series palette.
TOKENS = ("accent", "text-primary", "text-secondary", "text-muted", "success", "danger",
          "border", "divider", "series-2", "series-3")


def _name_of(colour: Any,
             theme: Theme) -> str:
    """A colour's name in a theme, or the colour itself."""
    value = to_hex(colour)

    for token in TOKENS:
        if to_hex(theme.colour(token)) == value:
            return token

    for index, series in enumerate(theme.series(), start=1):
        if to_hex(series) == value:
            return f"palette-{index}"

    return value


def colour(name: str,
           ax: Axes) -> str:
    """A colour by name, in the theme an axes is drawn in and the current
    style."""
    from .frame import ink

    current = design.current()

    if name.startswith("#"):
        return name
    if name == "accent-bar":
        return current.colors.accent_bar or _theme(ax).colour("accent")
    if name.startswith("palette-"):
        series = current.colors.series or _theme(ax).series()
        return series[(int(name.removeprefix("palette-")) - 1) % len(series)]

    return ink(ax, name)


class LaidOutFormatter(Formatter):
    """Tick labels as the layout wrote them, and the chart's own formatter
    for any tick the layout did not place (after a caller moves the axis)."""

    def __init__(self,
                 axis: AxisLayout,
                 inner: Formatter,
                 locator: Any):
        self.labels = {tick.value: tick.label for tick in axis.ticks}
        self.inner = inner
        self.locator = locator
        self.offset = axis.offset

    def __call__(self,
                 x: float,
                 pos: int | None = None) -> str:
        for value, label in self.labels.items():
            if abs(value - x) <= 1e-9 * max(1.0, abs(value)):
                return label

        return str(self.inner(x, pos))

    def get_offset(self) -> str:
        return self.offset


def draw_axes(ax: Axes,
              panel: PanelLayout) -> None:
    """Put an axes' ranges, ticks and labels where the layout says."""
    for axis, laid, limits in ((ax.xaxis, panel.x, ax.set_xlim), (ax.yaxis, panel.y, ax.set_ylim)):
        low, high = (laid.high, laid.low) if laid.inverted else (laid.low, laid.high)
        limits(low, high)
        if laid.kind in ("number", "date"):
            inner, locator = axis.get_major_formatter(), axis.get_major_locator()
            if isinstance(inner, LaidOutFormatter):
                earlier = inner
                inner, locator = earlier.inner, earlier.locator
            axis.set_major_locator(FixedLocator([tick.value for tick in laid.ticks]))
            axis.set_major_formatter(LaidOutFormatter(laid, inner, locator))


def draw(layout: Layout,
         axes: list[Axes],
         figure: Figure) -> None:
    """Draw a layout onto the figure whose chart it was laid out from.

    Args:
        layout: From lay_out().
        axes: The panels' axes, in the layout's order.
        figure: Their figure.
    """
    width, height = layout.width, layout.height
    main = axes[0]

    for ax, panel in zip(axes, layout.panels, strict=True):
        ax.set_position(_fractions(panel.box, width, height))
        draw_axes(ax, panel)
        _place_titles(ax, panel, figure)
        scale = _colour_bar_of(figure, ax)
        if panel.colour_bar is not None and scale is not None:
            scale.set_position(_fractions(panel.colour_bar.box, width, height))

    _draw_legend(main, layout)

    for text in (layout.title, layout.subtitle, layout.notes):
        if text is not None:
            _text(figure, text, main, width, height)

    if layout.accent_bar is not None:
        bar = layout.accent_bar
        # Square-cornered: a Rectangle has no rounding, and with no edge drawn
        # its corners are exactly the box's.
        figure.add_artist(Rectangle((bar.left / width, bar.bottom / height),
                                    bar.width / width, bar.height / height,
                                    transform=figure.transFigure,
                                    facecolor=colour("accent-bar", main), edgecolor="none",
                                    linewidth=0, gid="beacon-accent-bar"))

    if layout.mark is not None:
        mark = layout.mark
        draw_mark(figure, mark.right, mark.bottom + mark.height / 2.0,
                  colour("text-primary", main), mark.height)


def _colour_bar_of(figure: Figure,
                   ax: Axes) -> Axes | None:
    """The axes of the colour bar an axes' image was given, if any."""
    for other in figure.get_axes():
        bar = getattr(other, "_colorbar", None)
        if bar is not None and bar.mappable.axes is ax:
            return other

    return None


def _fractions(box: Box,
               width: float,
               height: float) -> tuple[float, float, float, float]:
    return (box.left / width, box.bottom / height, box.width / width, box.height / height)


def _place_titles(ax: Axes,
                  panel: PanelLayout,
                  figure: Figure) -> None:
    """Pin the axis titles where the layout put them."""
    if panel.y.title is not None:
        ax.yaxis.set_label_coords(panel.y.title.x, panel.y.title.y,
                                  transform=figure.dpi_scale_trans)
    if panel.x.title is not None:
        ax.xaxis.set_label_coords(panel.x.title.x, panel.x.title.y,
                                  transform=figure.dpi_scale_trans)


def _text(figure: Figure,
          text: Text,
          ax: Axes,
          width: float,
          height: float) -> None:
    figure.text(text.x / width, text.y / height, text.text,
                fontfamily=font_family(text.font.family), fontsize=text.size,
                fontweight=text.font.weight,
                fontstyle="italic" if text.font.italic else "normal",
                color=colour(text.colour, ax), ha=text.ha, va=text.va,
                rotation=text.rotation, linespacing=text.line_height, gid=text.id)


def _draw_legend(ax: Axes,
                 layout: Layout) -> None:
    """Redraw the axes' legend where the layout pins it, or remove it."""
    shown = ax.get_legend()
    legend = layout.legend

    if shown is None:
        return

    if legend is None:
        shown.remove()
        return

    figure = ax.get_figure()
    assert figure is not None
    pairs = [(handle, text.get_text())
             for handle, text in zip(shown.legend_handles, shown.get_texts(), strict=False)
             if handle is not None]
    handles = [handle for handle, _ in pairs]
    labels = [label for _, label in pairs]
    x, y = legend.anchor

    ax.legend(handles, labels, ncol=legend.columns, loc=legend.corner,
              bbox_to_anchor=(x / layout.width, y / layout.height),
              bbox_transform=figure.transFigure, frameon=False,
              borderaxespad=0.0, borderpad=0.0,
              handlelength=HANDLE_LENGTH, handletextpad=HANDLE_GAP,
              columnspacing=COLUMN_GAP, labelspacing=ROW_HEIGHT - 1.0)


def apply_grid(ax: Axes) -> None:
    """The style's gridlines, when it sets them; otherwise the chart's own."""
    grid = design.current().axes.grid

    if grid is None:
        return

    figure = ax.get_figure()

    for panel in (figure.get_axes() if isinstance(figure, Figure) else [ax]):
        if panel.get_label() == MARK_LABEL or hasattr(panel, "_colorbar"):
            continue
        panel.grid(visible=False)
        if grid in ("x", "both"):
            panel.grid(visible=True, axis="x")
        if grid in ("y", "both"):
            panel.grid(visible=True, axis="y")

