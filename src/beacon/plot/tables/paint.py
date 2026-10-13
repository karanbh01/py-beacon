# src/beacon/plot/tables/paint.py
"""
The matplotlib renderer for a table: its geometry drawn onto a figure, in the
frame every chart has.
"""
# BN-302.
from collections.abc import Callable
from typing import Any

from ..._optional import require
from ...tokens import DARK
from ..colormaps import ink_on
from ..layout import Box
from ..themes import Theme, remember, theme_named
from .draw import HeatText
from .geometry import measure
from .model import over
from .sizes import Blend, Geometry

require("matplotlib", "Charting")

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..paint import Painted, _figure, _Painter  # noqa: E402
from ..style import register, style_dict  # noqa: E402

# What a faded heatmap cell is laid over where the theme draws no background.
GROUND = {"light": "#ffffff", "dark": "#232323"}

# A faded heatmap cell paler than this keeps the table's own text colour.
FAINT = 0.45

# How many pixels a blended heatmap's colours are worked out at, across each
# cell.
BLEND_STEPS = 24


def paint_table(table: Any,
                make: Callable[..., Figure] | None = None) -> Painted:
    """Draw a table."""
    register()
    style = table.style
    theme = theme_named(style.theme)
    params = style_dict(theme.name, style=style)
    geometry = measure(table)
    layout = geometry.layout

    with matplotlib.rc_context(params):
        figure = (make or _figure)(figsize=(layout.width, layout.height))
        remember(figure, figure.get_facecolor())
        ax = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.set_axis_off()
        ax.set_xlim(0.0, layout.width)
        ax.set_ylim(0.0, layout.height)
        _TablePainter(figure, ax, geometry, theme, style).paint()
        figure.canvas.draw()

    return Painted(figure, (ax,), (None,), params)


def show_table(table: Any) -> None:
    """Draw a table on a pyplot figure and show it."""
    import matplotlib.pyplot as plt

    paint_table(table, make=plt.figure)
    plt.show()


class _TablePainter:
    """Draws one table's fills, rules, text and frame, in inches on an axes
    the size of the figure."""

    def __init__(self,
                 figure: Figure,
                 ax: Any,
                 geometry: Geometry,
                 theme: Theme,
                 style: Any):
        self.ax = ax
        self.geometry = geometry
        self.frame = _Painter(figure, geometry.layout, theme, style)
        self.ground = (theme.canvas if theme.canvas is not None
                       else GROUND["dark" if theme.mode == DARK else "light"])

    def ink(self, name: str) -> str:
        return self.frame.ink(name)

    def paint(self) -> None:
        geometry = self.geometry

        for patch in geometry.patches:
            colour = (over(patch.fill, self.ground) if patch.fill is not None
                      else self.ink(patch.colour))
            # Stripes lie under a blended heatmap; its cells and blanks over it.
            self._rectangle(patch.box, colour, 1.0 if patch.fill is not None else 0.3)

        for blend in geometry.blends:
            self._blend(blend)

        for rule in geometry.rules:
            self.ax.add_line(Line2D((rule.x0, rule.x1), (rule.y0, rule.y1),
                                    linewidth=rule.width, color=self.ink(rule.colour),
                                    solid_capstyle="butt"))

        for text in geometry.texts:
            self.frame.text(text)
        for heat in geometry.heat_texts:
            self._heat_text(heat)

        self.frame.frame(self.ax)

    def _rectangle(self,
                   box: Box,
                   colour: str,
                   zorder: float) -> None:
        self.ax.add_patch(Rectangle((box.left, box.bottom), box.width, box.height,
                                    facecolor=colour, edgecolor="none", linewidth=0,
                                    zorder=zorder))

    def _blend(self,
               blend: Blend) -> None:
        """Colours blended between cells' middles, across, down or both."""
        grid = np.array([[_rgb(over(fill, self.ground)) for fill in row]
                         for row in blend.fills])
        rows, columns = grid.shape[0], grid.shape[1]
        across = _stretched(grid, columns, axis=1) if columns > 1 else grid
        image = _stretched(across, rows, axis=0) if rows > 1 else across
        box = blend.box
        self.ax.imshow(image, extent=(box.left, box.right, box.bottom, box.top),
                       aspect="auto", interpolation="bilinear", origin="upper", zorder=0.5)

    def _heat_text(self,
                   heat: HeatText) -> None:
        from dataclasses import replace

        ink = (self.ink("text-primary") if heat.fill.alpha < FAINT
               else ink_on(over(heat.fill, self.ground)))
        self.frame.text(replace(heat.text, colour=ink))


def _stretched(grid: Any,
               cells: int,
               axis: int) -> Any:
    """Colours worked out between each cell's middle along an axis, holding
    the first and last cells' colours out to their outer edges."""
    steps = cells * BLEND_STEPS
    places = (np.arange(steps) + 0.5) / BLEND_STEPS - 0.5
    places = np.clip(places, 0.0, cells - 1.0)
    low = np.floor(places).astype(int)
    high = np.minimum(low + 1, cells - 1)
    part = (places - low).reshape((-1, 1, 1) if axis == 0 else (1, -1, 1))
    first = np.take(grid, low, axis=axis)
    second = np.take(grid, high, axis=axis)

    return first + (second - first) * part


def _rgb(colour: str) -> tuple[float, float, float]:
    value = colour.lstrip("#")

    return (int(value[0:2], 16) / 255.0, int(value[2:4], 16) / 255.0,
            int(value[4:6], 16) / 255.0)


__all__ = ["paint_table", "show_table"]
