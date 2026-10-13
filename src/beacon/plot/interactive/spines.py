# src/beacon/plot/interactive/spines.py
"""
A panel's axis lines and tick marks, drawn where the image draws them.

Each panel's lines are one path, joined at the corner where they meet and
ending at the plot's edges, and each axis's tick marks one path of the same
weight, all placed in the figure's own coordinates from the layout, as the
styles preview draws them. Plotly's own lines are snapped to the pixel and
drawn outside the plot, and its tick marks leave out an axis's two ends, so
neither meets the other as the image's do.
"""
# BN-301.
from typing import Any

from ..layout import AxisLayout, Box, PanelLayout
from .units import PT, PX, Paint


class Spines:
    """One panel's axis lines and tick marks, as path shapes."""

    def __init__(self,
                 panel: PanelLayout,
                 width: float,
                 height: float,
                 paint: Paint,
                 lines: bool = True):
        self.panel = panel
        self.width = width
        self.height = height
        self.paint = paint
        self.lines = lines
        axes = paint.style.axes
        self.ink = paint.ink("text-primary")
        self.weight = paint.style.lines.axis_width * paint.k * PT
        self.reach = axes.tick_length * paint.k / 72.0 * (1.0 if axes.tick_direction == "in"
                                                          else -1.0)

    def shapes(self) -> list[dict[str, Any]]:
        """The panel's lines, then its tick marks."""
        box = self.panel.box
        axes = self.paint.style.axes
        across = self.at(self.panel.y, box.bottom, box.height, 0.0) if self.panel.origin \
            else box.bottom
        up = self.at(self.panel.x, box.left, box.width, 0.0) if self.panel.origin \
            else box.left
        drawn = []

        if self.lines:
            drawn += self.spine(box, across, up, axes.x_line, axes.y_line)
            if self.panel.y2 is not None and axes.y_line:
                drawn.append(self.path(f"M{self.x(box.right)},{self.y(box.top)}"
                                       f"V{self.y(box.bottom)}"))

        drawn += self.ticks(self.panel.x, across, True, 1.0, axes.x_ticks)
        drawn += self.ticks(self.panel.y, up, False, 1.0, axes.y_ticks)
        if self.panel.y2 is not None:
            drawn += self.ticks(self.panel.y2, box.right, False, -1.0, axes.y_ticks)

        return drawn

    def spine(self,
              box: Box,
              across: float,
              up: float,
              x_line: bool,
              y_line: bool) -> list[dict[str, Any]]:
        """The two axis lines: joined at the corner where they meet at the
        plot's edges, or each on its own where they cross at zero."""
        left, right = self.x(box.left), self.x(box.right)
        bottom, top = self.y(box.bottom), self.y(box.top)

        if self.panel.origin:
            parts = ([f"M{left},{self.y(across)}H{right}"] if x_line else []) + (
                [f"M{self.x(up)},{top}V{bottom}"] if y_line else [])
            return [self.path("".join(parts))] if parts else []
        if x_line and y_line:
            return [self.path(f"M{left},{top}V{bottom}H{right}")]
        if x_line:
            return [self.path(f"M{left},{bottom}H{right}")]
        if y_line:
            return [self.path(f"M{left},{top}V{bottom}")]

        return []

    def ticks(self,
              laid: AxisLayout,
              base: float,
              horizontal: bool,
              side: float,
              shown: bool) -> list[dict[str, Any]]:
        """An axis's tick marks, every one the layout has, ends included, as
        one path: up from a horizontal axis, and in from a vertical one on
        whichever side it is."""
        if not shown or laid.kind in ("names", "fixed") or not laid.ticks:
            return []

        box = self.panel.box
        reach = self.reach * side
        # From the axis line's far edge rather than its middle, so a tick at
        # the line's square end closes the corner instead of leaving a notch
        # half the line's width across.
        start = base - (self.weight / PX / 2.0) * (1.0 if reach > 0 else -1.0)
        marks = []

        for tick in laid.ticks:
            if horizontal:
                at = self.at(laid, box.left, box.width, tick.value)
                marks.append(f"M{self.x(at)},{self.y(start)}V{self.y(base + reach)}")
            else:
                at = self.at(laid, box.bottom, box.height, tick.value)
                marks.append(f"M{self.x(start)},{self.y(at)}H{self.x(base + reach)}")

        return [self.path("".join(marks), "beacon-ticks")]

    @staticmethod
    def at(laid: AxisLayout,
           start: float,
           length: float,
           value: float) -> float:
        """Where a value falls along an axis, in inches."""
        share = (value - laid.low) / ((laid.high - laid.low) or 1.0)

        return start + (1.0 - share if laid.inverted else share) * length

    def x(self, inches: float) -> str:
        return f"{inches / self.width:.6f}"

    def y(self, inches: float) -> str:
        return f"{inches / self.height:.6f}"

    def path(self,
             d: str,
             name: str = "beacon-axes") -> dict[str, Any]:
        return {"type": "path", "path": d, "xref": "paper", "yref": "paper", "name": name,
                "layer": "above", "line": {"color": self.ink, "width": self.weight},
                "fillcolor": "rgba(0,0,0,0)"}
