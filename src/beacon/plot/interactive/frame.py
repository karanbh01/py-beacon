# src/beacon/plot/interactive/frame.py
"""
The frame around an interactive chart, drawn where the layout puts it: the
beta mark, the accent bar, the heading, the legend, the axis and panel
titles, the notes and a heatmap's colour scale, in the figure's own
coordinates (fractions of its width and height).
"""
# BN-301.
import re
from typing import Any

from .. import style as sizes
from ..layout import Box, Layout, LegendLayout, Text
from ..mark import BETA_PATH
from ..webfonts import css_family
from .baseline import shift
from .traces import DASH
from .units import PT, Paint

# The beta glyph's extent in its own units, x0, y0, x1, y1, as matplotlib
# measures its outline (beacon.plot.mark.beta_outline).
BETA_EXTENT = (0.0, 0.0, 21.451444174920656, 33.86989715454607)

# A marked period's legend key: a thin bar, in points.
PERIOD_KEY_WIDTH = 3.2

# A legend's dot, across, in points as designed (matplotlib's markersize).
MARKER_SIZE = 6.0


class Frame:
    """The frame's shapes and annotations."""

    def __init__(self,
                 layout: Layout,
                 paint: Paint):
        self.layout = layout
        self.paint = paint
        self.shapes: list[dict[str, Any]] = []
        self.annotations: list[dict[str, Any]] = []

    def draw(self) -> "Frame":
        layout = self.layout
        for text in (layout.title, layout.subtitle, layout.notes):
            if text is not None:
                self.text(text)

        for panel in layout.panels:
            for text in (panel.title, panel.x.title, panel.y.title,
                         panel.y2.title if panel.y2 is not None else None):
                if text is not None:
                    self.text(text)
            if panel.colour_bar is not None:
                for text in panel.colour_bar.labels:
                    self.text(text)

        if layout.accent_bar is not None:
            self.rect(layout.accent_bar, self.paint.ink("accent-bar"))
        if layout.mark is not None:
            self.mark(layout.mark)
        if layout.legend is not None:
            self.legend(layout.legend)

        return self

    def x(self, inches: float) -> float:
        return inches / self.layout.width

    def y(self, inches: float) -> float:
        return inches / self.layout.height

    def text(self, text: Text) -> None:
        """Text placed in inches; each line of several placed on its own, at
        the layout's line height."""
        lines = text.text.split("\n")
        step = text.size * text.line_height / 72.0
        top = text.y + (step * (len(lines) - 1) if text.va == "bottom" else
                        step * (len(lines) - 1) / 2.0 if text.va == "center" else 0.0)
        font = {"family": css_family(text.font.family), "size": text.size * PT,
                "color": self.paint.ink(text.colour), "weight": text.font.weight,
                "style": "italic" if text.font.italic else "normal"}

        across, up = anchors(text)
        turned = abs(text.rotation) > 1e-9
        for index, line in enumerate(lines):
            self.annotations.append({
                "text": line, "showarrow": False, "font": font, "borderpad": 0,
                "x": self.x(text.x), "y": self.y(top - index * step),
                "yshift": 0.0 if turned else shift(lines[0], text.size, text.font, text.va),
                "xref": "paper", "yref": "paper", "xanchor": across, "yanchor": up,
                "textangle": -text.rotation, "name": text.id})

    def rect(self,
             box: Box,
             colour: str,
             name: str = "") -> None:
        self.shapes.append({"type": "rect", "xref": "paper", "yref": "paper",
                            "x0": self.x(box.left), "x1": self.x(box.right),
                            "y0": self.y(box.bottom), "y1": self.y(box.top),
                            "fillcolor": colour, "line": {"width": 0}, "layer": "above",
                            "name": name})

    def mark(self, box: Box) -> None:
        """The beta mark, its outline fitted to the layout's box."""
        x0, y0, x1, y1 = BETA_EXTENT
        across = box.width / (x1 - x0)
        up = box.height / (y1 - y0)

        def place(command: str, numbers: list[float]) -> str:
            if command == "H":
                return f"H{self.x(box.left + (numbers[0] - x0) * across):.6f}"
            if command == "V":
                return f"V{self.y(box.bottom + (numbers[0] - y0) * up):.6f}"
            pairs = zip(numbers[::2], numbers[1::2], strict=True)
            return command + " ".join(f"{self.x(box.left + (a - x0) * across):.6f},"
                                      f"{self.y(box.bottom + (b - y0) * up):.6f}"
                                      for a, b in pairs)

        path = "".join(place(command, [float(value) for value in re.findall(
            r"-?[\d.]+(?:e-?\d+)?", body)]) if command != "Z" else "Z"
            for command, body in re.findall(r"([MCLHVZ])([^MCLHVZ]*)", BETA_PATH))
        self.shapes.append({"type": "path", "path": path, "xref": "paper", "yref": "paper",
                            "fillcolor": self.paint.ink("text-primary"),
                            "line": {"width": 0}, "layer": "above", "name": "beacon-mark"})

    def legend(self, legend: LegendLayout) -> None:
        """Each entry's key and label where the layout puts them."""
        for entry in legend.entries:
            self.key(entry.key, entry.key_box, self.paint.ink(entry.colour))
            self.text(entry.text)

    def key(self,
            kind: str,
            box: Box,
            ink: str) -> None:
        middle = self.y(box.bottom + box.height / 2.0)
        if kind == "patch":
            self.rect(box, ink)
            return
        if kind == "marker":
            # matplotlib's legend marker: its default marker size, across.
            size = MARKER_SIZE * self.paint.k / 72.0
            centre = box.left + box.width / 2.0
            self.shapes.append({
                "type": "circle", "xref": "paper", "yref": "paper", "fillcolor": ink,
                "line": {"width": 0}, "layer": "above",
                "x0": self.x(centre - size / 2.0), "x1": self.x(centre + size / 2.0),
                "y0": middle - self.y(size / 2.0), "y1": middle + self.y(size / 2.0)})
            return

        width = (PERIOD_KEY_WIDTH if kind == "period" else sizes.series_width()) * PT
        line: dict[str, Any] = {"color": ink, "width": width}
        if kind == "dashed":
            line["dash"] = f"{DASH[0] * width:.2f}px,{DASH[1] * width:.2f}px"
        self.shapes.append({"type": "line", "xref": "paper", "yref": "paper", "line": line,
                            "x0": self.x(box.left), "x1": self.x(box.right),
                            "y0": middle, "y1": middle, "layer": "above"})


def anchors(text: Text) -> tuple[str, str]:
    """Where plotly anchors text: the layout gives its alignment in the
    text's own frame, and plotly aligns the box it fills once turned, so a
    title turned a quarter to read upward has its bottom at the right."""
    if abs(text.rotation - 90.0) < 1e-9:
        across = {"bottom": "right", "baseline": "right", "top": "left"}.get(text.va, "center")
        up = {"left": "bottom", "right": "top"}.get(text.ha, "middle")
        return across, up

    # Level text hangs from its baseline (beacon.plot.interactive.baseline).
    return text.ha, "bottom"
