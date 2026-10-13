# src/beacon/plot/paint.py
"""
The matplotlib renderer for the chart classes: a chart's plan and layout
drawn onto a figure.

Each panel's axes are put where the layout says, its marks drawn on them,
and its ranges, ticks and titles set from the layout; then the frame is
drawn around them: the mark, the accent bar, the heading, the legend and the
notes. Colours are resolved from their names in the chart's theme and style.
"""
# BN-299.
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .._optional import require
from .colormaps import colour_at
from .colours import resolve
from .design import Style
from .layout import AxisLayout, Box, Layout, LegendLayout, PanelLayout, Text
from .marks import (
    AreaMark,
    BarMark,
    CellsMark,
    LineMark,
    Mark,
    PointMark,
    RuleMark,
    SpanMark,
    TextMark,
)
from .themes import Theme, remember, theme_named

require("matplotlib", "Charting")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.collections import PatchCollection  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from . import style as sizes  # noqa: E402
from .charts.base import Plan  # noqa: E402
from .legend import COLUMN_GAP, HANDLE_GAP, HANDLE_LENGTH, ROW_HEIGHT  # noqa: E402
from .mark import draw_mark  # noqa: E402
from .panels import CHART_WEIGHT  # noqa: E402
from .style import register, style_dict  # noqa: E402

# A dashed line's dash and gap, in multiples of its width.
DASH = (2.0, 1.6)

# How far a cell overlaps its neighbours when there is no gap between them,
# as a share of a cell.
CELL_OVERLAP = 0.01

# The outline round text drawn over the lines, in points.
HALO_WIDTH = 2.0

# A marked period's legend key: a thin rectangle drawn as a thick,
# square-ended line, in points.
PERIOD_KEY_WIDTH = 3.2


@dataclass(frozen=True)
class Painted:
    """A chart drawn: its figure, each panel's axes, each panel's second axes
    on the right where it has one, and the rcParams it is drawn in."""

    figure: Figure
    axes: tuple[Axes, ...]
    right: tuple[Axes | None, ...]
    params: dict[str, Any]

    def save(self,
             target: Any,
             **options: Any) -> None:
        """Save the figure in the rcParams it was drawn in."""
        with matplotlib.rc_context(self.params):
            self.figure.savefig(target, **options)


def paint(plan: Plan,
          layout: Layout,
          style: Style,
          make: Callable[..., Figure] | None = None) -> Painted:
    """Draw a chart.

    Args:
        plan: Its marks and description.
        layout: Where everything goes.
        style: The settings it is drawn in.
        make: Makes the figure, given its size; a figure of its own, outside
            pyplot, by default.
    """
    register()
    theme = theme_named(style.theme)
    params = style_dict(theme.name, style=style)

    with matplotlib.rc_context(params):
        figure = (make or _figure)(figsize=(layout.width, layout.height))
        remember(figure, figure.get_facecolor())
        painter = _Painter(figure, layout, theme, style)
        axes, right = painter.panels(plan.front)
        painter.marks(plan.marks, axes, right)
        for index, (ax, twin, panel) in enumerate(zip(axes, right, layout.panels, strict=True)):
            painter.dress(ax, twin, panel, plan.grid)
            stops = next((mark.stops for mark in plan.marks
                          if isinstance(mark, CellsMark) and mark.panel == index), None)
            if panel.colour_bar is not None and stops is not None:
                painter.colour_bar(panel, [painter.ink(stop) for stop in stops])
        painter.frame(axes[0])
        # Drawn once here, so every tick and label is made in the style's
        # rcParams rather than whichever are in force when it is shown.
        figure.canvas.draw()

    return Painted(figure, tuple(axes), tuple(right), params)


def show(plan: Plan,
         layout: Layout,
         style: Style) -> None:
    """Draw a chart on a pyplot figure and show it."""
    import matplotlib.pyplot as plt

    paint(plan, layout, style, make=plt.figure)
    plt.show()


def _figure(figsize: tuple[float, float]) -> Figure:
    figure = Figure(figsize=figsize)
    FigureCanvasAgg(figure)

    return figure


class _Painter:
    """Draws one chart's panels, marks and frame."""

    def __init__(self,
                 figure: Figure,
                 layout: Layout,
                 theme: Theme,
                 style: Style):
        self.figure = figure
        self.layout = layout
        self.theme = theme
        self.style = style
        self.families = _families(style.text.font)
        self.tick_size = sizes.tick_label_size()
        self.title_size = sizes.axis_title_size()

    def ink(self, name: str) -> str:
        """A colour by name."""
        return resolve(name, self.theme, self.style)

    def fractions(self, box: Box) -> tuple[float, float, float, float]:
        width, height = self.layout.width, self.layout.height

        return (box.left / width, box.bottom / height, box.width / width, box.height / height)

    def panels(self,
               front: str) -> tuple[list[Axes], list[Axes | None]]:
        """Each panel's axes, and its second axes where it has one, with the
        *front* side's drawn over the other's."""
        axes: list[Axes] = []
        right: list[Axes | None] = []

        for panel in self.layout.panels:
            ax = self.figure.add_axes(self.fractions(panel.box))
            twin = ax.twinx() if panel.y2 is not None else None
            if twin is not None and front == "left":
                # The main axes over the second, so the leading series and
                # its labels are drawn over those measured on the right.
                ax.set_zorder(twin.get_zorder() + 1)
                ax.patch.set_visible(False)
            axes.append(ax)
            right.append(twin)

        return axes, right

    def marks(self,
              marks: tuple[Mark, ...],
              axes: list[Axes],
              right: list[Axes | None]) -> None:
        """Draw each mark on its panel, on the side it is measured on."""
        for mark in marks:
            ax = axes[mark.panel]
            twin = right[mark.panel]
            if getattr(mark, "side", "left") == "right" and twin is not None:
                ax = twin
            self.mark(ax, mark)

    def mark(self,
             ax: Axes,
             mark: Mark) -> None:
        """Draw one mark."""
        if isinstance(mark, LineMark):
            (line,) = ax.plot(mark.x, mark.y, color=self.ink(mark.colour),
                              linewidth=mark.width, alpha=mark.alpha, zorder=mark.layer)
            if mark.dashed:
                line.set_dashes(DASH)
        elif isinstance(mark, AreaMark):
            ax.fill_between(mark.x, mark.lower, mark.upper, color=self.ink(mark.colour),
                            alpha=mark.alpha, linewidth=0, zorder=mark.layer)
        elif isinstance(mark, BarMark):
            self.bars(ax, mark)
        elif isinstance(mark, PointMark):
            ax.plot(mark.x, mark.y, linestyle="none", marker="o", markersize=mark.size,
                    markerfacecolor=self.ink(mark.colour), markeredgewidth=0,
                    alpha=mark.alpha, zorder=mark.layer, clip_on=mark.clip)
        elif isinstance(mark, RuleMark):
            rule = ax.axvline if mark.vertical else ax.axhline
            drawn = rule(mark.value, color=self.ink(mark.colour), linewidth=mark.width,
                         zorder=mark.layer, gid=mark.id or None)
            if mark.dashed:
                drawn.set_dashes(DASH)
        elif isinstance(mark, SpanMark):
            # Opaque, borderless and behind every series.
            ax.axvspan(mark.start, mark.end, facecolor=self.ink(mark.colour),
                       edgecolor="none", linewidth=0, zorder=0.5)
        elif isinstance(mark, TextMark):
            self.text_mark(ax, mark)
        elif isinstance(mark, CellsMark):
            self.cells(ax, mark)

    def cells(self,
              ax: Axes,
              mark: CellsMark) -> None:
        """A grid of coloured cells, row 0 at the top: each its own square,
        or blended into its neighbours."""
        stops = [self.ink(stop) for stop in mark.stops]
        rows, columns = len(mark.values), len(mark.values[0]) if mark.values else 0
        for side in ax.spines.values():
            side.set_visible(False)

        if mark.blended:
            table = np.array([[np.nan if value is None else value for value in row]
                              for row in mark.values])
            ax.imshow(table, cmap=_colour_map(stops), vmin=mark.low, vmax=mark.high,
                      interpolation="bilinear", aspect="auto",
                      extent=(-0.5, columns - 0.5, rows - 0.5, -0.5), zorder=2)
            return

        span = (mark.high - mark.low) or 1.0
        gap = mark.gap
        squares, fills = [], []

        for i, row in enumerate(mark.values):
            for j, value in enumerate(row):
                if value is None:
                    continue
                # With no gap, each square overlaps its neighbours a little,
                # or an anti-aliased seam shows the background between them.
                wide = 1.0 - gap + (CELL_OVERLAP if gap == 0 and j < columns - 1 else 0.0)
                tall = 1.0 - gap + (CELL_OVERLAP if gap == 0 and i < rows - 1 else 0.0)
                squares.append(Rectangle((j - 0.5 + gap / 2.0, i - 0.5 + gap / 2.0), wide, tall))
                fills.append(colour_at(stops, (value - mark.low) / span))

        ax.add_collection(PatchCollection(squares, facecolors=fills, edgecolors="none",
                                          linewidths=0, zorder=2))

    def colour_bar(self,
                   panel: PanelLayout,
                   stops: list[str]) -> None:
        """A colour scale beside a panel, labelled at its ends or along it."""
        bar = panel.colour_bar
        assert bar is not None
        cax = self.figure.add_axes(self.fractions(bar.box))
        low, high = bar.axis.low, bar.axis.high
        cax.imshow(np.linspace(0.0, 1.0, 256)[:, None], aspect="auto", origin="lower",
                   cmap=_colour_map(stops), extent=(0.0, 1.0, low, high))
        cax.set_xlim(0.0, 1.0)
        cax.set_ylim(low, high)
        cax.set_xticks([])
        for side in cax.spines.values():
            side.set_visible(False)
        cax.yaxis.tick_right()
        cax.yaxis.set_ticks([tick.value for tick in bar.axis.ticks],
                            [tick.label for tick in bar.axis.ticks])
        cax.tick_params(axis="y", length=0, labelsize=self.tick_size,
                        labelfontfamily=self.families, labelcolor=self.ink("text-primary"))

        if bar.axis.kind == "fixed":
            # Labels at the ends sit within the bar's height.
            labels = cax.get_yticklabels()
            labels[0].set_verticalalignment("bottom")
            labels[-1].set_verticalalignment("top")

    def bars(self,
             ax: Axes,
             mark: BarMark) -> None:
        colours = [self.ink(name) for name in mark.colours]
        base = list(mark.base) or 0.0
        extra: dict[str, Any] = {"align": "edge" if mark.edge else "center"}
        if mark.seamless:
            # Bars that meet edge to edge: an edge in each bar's own colour
            # hides the hairline seam anti-aliasing leaves between them.
            extra.update(edgecolor=colours, linewidth=0.25)

        if mark.horizontal:
            ax.barh(mark.positions, mark.values, height=mark.thickness, left=base,
                    color=colours, zorder=mark.layer, **extra)
        else:
            ax.bar(mark.positions, mark.values, width=mark.thickness, bottom=base,
                   color=colours, zorder=mark.layer, **extra)

    def text_mark(self,
                  ax: Axes,
                  mark: TextMark) -> None:
        coords: Any = {"data": "data", "axes": "axes fraction",
                       "xdata": ("data", "axes fraction"),
                       "ydata": ("axes fraction", "data")}[mark.coords]
        text = ax.annotate(mark.text, xy=(mark.x, mark.y), xycoords=coords,
                           xytext=(mark.dx, mark.dy), textcoords="offset points",
                           ha=mark.ha, va=mark.va, fontsize=mark.size,
                           color=self.ink(mark.colour), fontfamily=self.families,
                           fontweight=mark.weight, annotation_clip=mark.clip, zorder=6)
        canvas = self.ink("canvas")
        if mark.halo and canvas != "none":
            text.set_path_effects([patheffects.withStroke(linewidth=HALO_WIDTH,
                                                          foreground=canvas)])

    def dress(self,
              ax: Axes,
              twin: Axes | None,
              panel: PanelLayout,
              grid: str) -> None:
        """Set a panel's ranges, ticks, titles, axis lines and gridlines from
        the layout."""
        self.axis(ax, ax.xaxis, panel.x, ax.set_xlim)
        self.axis(ax, ax.yaxis, panel.y, ax.set_ylim)
        _gridlines(ax, self.style.axes.grid or grid)

        if panel.origin:
            ax.spines["left"].set_position(("data", 0.0))
            ax.spines["bottom"].set_position(("data", 0.0))

        if twin is not None and panel.y2 is not None:
            self.axis(twin, twin.yaxis, panel.y2, twin.set_ylim)
            twin.spines["right"].set_visible(self.style.axes.y_line)
            twin.tick_params(axis="y", right=self.style.axes.y_ticks, labelright=True)
            twin.grid(visible=False)
            twin.yaxis.set_label_position("right")

    def axis(self,
             ax: Axes,
             axis: Any,
             laid: AxisLayout,
             limits: Callable[..., Any]) -> None:
        """One axis: its range, ticks, labels and title, as laid out."""
        low, high = (laid.high, laid.low) if laid.inverted else (laid.low, laid.high)
        limits(low, high)
        axis.set_ticks([tick.value for tick in laid.ticks],
                       [tick.label for tick in laid.ticks])
        name: Literal["x", "y"] = "x" if axis is ax.xaxis else "y"
        ax.tick_params(axis=name, labelsize=self.tick_size, labelfontfamily=self.families,
                       labelcolor=self.ink("text-primary"), color=self.ink("text-primary"))

        if laid.rotation:
            ax.tick_params(axis=name, labelrotation=laid.rotation)
            for label in axis.get_majorticklabels():
                label.set_horizontalalignment(laid.align)
        if not laid.show_labels:
            ax.tick_params(axis=name, labelbottom=False, labelleft=False)
        if laid.kind in ("names", "fixed"):
            ax.tick_params(axis=name, length=0)

        if laid.title is not None:
            axis.set_label_text(laid.title.text, fontfamily=self.families,
                                fontsize=self.title_size, fontweight=CHART_WEIGHT,
                                color=self.ink("text-primary"))
            axis.set_label_coords(laid.title.x, laid.title.y,
                                  transform=self.figure.dpi_scale_trans)

    def frame(self, main: Axes) -> None:
        """The heading, the legend, the notes, the accent bar and the mark."""
        layout = self.layout

        if layout.legend is not None:
            self.legend(main, layout.legend)

        for text in (layout.title, layout.subtitle, layout.notes):
            if text is not None:
                self.text(text)

        if layout.accent_bar is not None:
            bar = layout.accent_bar
            # Square-cornered: a Rectangle has no rounding, and with no edge
            # drawn its corners are exactly the box's.
            self.figure.add_artist(Rectangle(self.fractions(bar)[:2],
                                             bar.width / layout.width,
                                             bar.height / layout.height,
                                             transform=self.figure.transFigure,
                                             facecolor=self.ink("accent-bar"),
                                             edgecolor="none", linewidth=0,
                                             gid="beacon-accent-bar"))

        if layout.mark is not None:
            mark = layout.mark
            draw_mark(self.figure, mark.right, mark.bottom + mark.height / 2.0,
                      self.ink("text-primary"), mark.height)

    def text(self, text: Text) -> None:
        from .style import font_family

        self.figure.text(text.x / self.layout.width, text.y / self.layout.height, text.text,
                         fontfamily=font_family(text.font.family), fontsize=text.size,
                         fontweight=text.font.weight,
                         fontstyle="italic" if text.font.italic else "normal",
                         color=self.ink(text.colour), ha=text.ha, va=text.va,
                         rotation=text.rotation, linespacing=text.line_height, gid=text.id)

    def legend(self,
               main: Axes,
               legend: LegendLayout) -> None:
        """The legend, pinned where the layout puts it."""
        handles = [self.key(entry.key, entry.colour) for entry in legend.entries]
        labels = [entry.label for entry in legend.entries]
        x, y = legend.anchor

        main.legend(handles, labels, ncol=legend.columns, loc=legend.corner,
                    prop=FontProperties(family=self.families, weight=CHART_WEIGHT,
                                        size=legend.size),
                    labelcolor=self.ink("text-primary"),
                    bbox_to_anchor=(x / self.layout.width, y / self.layout.height),
                    bbox_transform=self.figure.transFigure, frameon=False,
                    borderaxespad=0.0, borderpad=0.0,
                    handlelength=HANDLE_LENGTH, handletextpad=HANDLE_GAP,
                    columnspacing=COLUMN_GAP, labelspacing=ROW_HEIGHT - 1.0)

    def key(self,
            kind: str,
            colour: str) -> Any:
        """A legend key."""
        ink = self.ink(colour)
        width = float(matplotlib.rcParams["lines.linewidth"])

        if kind == "patch":
            return Patch(facecolor=ink, edgecolor="none")
        if kind == "marker":
            return Line2D([], [], linestyle="none", marker="o", markerfacecolor=ink,
                          markeredgewidth=0)
        if kind == "period":
            return Line2D([], [], color=ink, linewidth=PERIOD_KEY_WIDTH, solid_capstyle="butt")
        if kind == "dashed":
            return Line2D([], [], color=ink, linewidth=width, dashes=DASH)

        return Line2D([], [], color=ink, linewidth=width)


def _colour_map(stops: list[str]) -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("beacon-cells", stops)


def _families(font: str) -> list[str]:
    """The font a chart's text is set in, then the fonts that stand in for a
    glyph it lacks."""
    from .style import font_family

    return list(dict.fromkeys([font_family(font), "Inter Tight", "DejaVu Sans"]))


def _gridlines(ax: Axes,
               grid: str) -> None:
    ax.grid(visible=False)
    if grid in ("x", "both"):
        ax.grid(visible=True, axis="x")
    if grid in ("y", "both"):
        ax.grid(visible=True, axis="y")
