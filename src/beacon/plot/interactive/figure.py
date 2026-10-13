# src/beacon/plot/interactive/figure.py
"""
A chart as a plotly figure, drawn from the same layout as its image: the
figure is the layout's size, with no margins, each panel's axes in the box
the layout puts them, their ranges and ticks the laid-out ones, and the
frame drawn where the layout puts it.
"""
# BN-301.
from typing import Any

from ..._optional import require
from .. import design
from ..layout import ColourBarLayout, Description, Layout, PanelLayout
from ..marks import BarMark, CellsMark, Mark, PointMark
from .axes import Names, axis, domain, origin, tick_labels
from .frame import Frame
from .popup import glass
from .spines import Spines
from .traces import Marks
from .units import PT, PX, Paint

require("plotly", "Interactive charts")

import plotly.graph_objects as go  # noqa: E402

from .html import figure_html, write_figure_html  # noqa: E402

# The colour bar's own colours, sampled along it.
BAR_STEPS = 256


class BeaconFigure(go.Figure):  # type: ignore[misc]
    """A Beacon chart as a plotly figure. Its HTML carries the fonts the
    chart is set in and the script that draws the popup over it, so it looks
    and reads the same offline as on the page."""

    def to_html(self, *args: Any, **kwargs: Any) -> str:
        """The figure as HTML, with its fonts and popup; plotly's own
        options otherwise.

        Takes two options of its own: fonts="embed" (the default) carries
        the fonts in the page, a stylesheet's address links one that has
        them, and None leaves them out; fit=True makes the figure fill the
        frame it is shown in rather than keep its own size.
        """
        return figure_html(self, super().to_html, *args, **kwargs)

    def write_html(self, file: Any, *args: Any, **kwargs: Any) -> None:
        """Write the figure's HTML, with its fonts and popup, to a file.

        Takes to_html's options. With include_plotlyjs="directory", plotly.js
        is written once beside it, as plotly's own write_html does, for
        several figures to share.
        """
        write_figure_html(self.to_html(*args, **kwargs), file,
                          kwargs.get("include_plotlyjs") == "directory")


def aligned(annotation: dict[str, Any]) -> dict[str, Any]:
    """Text set from its anchored edge. plotly otherwise centres text in a
    box it measured, and a box measured before the chart's fonts arrive puts
    each line a little off once they do."""
    return {**annotation, "align": annotation.get("xanchor", "center")}


def interactive(chart: Any) -> BeaconFigure:
    """A chart as a plotly figure, laid out as its image is."""
    with design.scoped(chart.style):
        plan = chart.plan
        return _Builder(plan.description, plan.marks, plan.grid, chart.layout,
                        Paint(chart.style)).build()


class _Builder:
    def __init__(self,
                 description: Description,
                 marks: tuple[Mark, ...],
                 grid: str,
                 layout: Layout,
                 paint: Paint):
        self.panels = description.panels
        self.legend = description.legend
        self.marks = marks
        self.grid = paint.style.axes.grid or grid
        self.layout = layout
        self.paint = paint
        self.axes: dict[str, Any] = {}
        self.ticks: list[dict[str, Any]] = []
        self.labels: list[dict[str, Any]] = []
        self.extra: list[dict[str, Any]] = []
        self.count = 0

    def build(self) -> BeaconFigure:
        names = [self.panel(index, panel) for index, panel in enumerate(self.layout.panels)]
        legend = [item.label for item in self.legend]
        drawn = Marks(self.panels, names, self.paint, self.layout, legend).draw(self.marks)
        frame = Frame(self.layout, self.paint).draw()
        canvas = self.paint.canvas
        figure = BeaconFigure()
        figure.add_traces([*drawn.traces, *self.extra])
        figure.update_layout(
            width=self.layout.width * PX, height=self.layout.height * PX,
            autosize=False, margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
            paper_bgcolor=canvas or "rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": self.paint.family, "color": self.paint.ink("text-primary"),
                  "size": self.paint.tick_size * PT},
            showlegend=False, dragmode=False, barmode="overlay", hovermode=self.hovermode(),
            hoverlabel={"font": {"family": self.paint.family}},
            shapes=[*drawn.shapes, *self.ticks, *frame.shapes],
            annotations=[aligned(annotation) for annotation in
                         (*drawn.annotations, *self.labels, *frame.annotations)],
            meta={"beacon": glass(self.paint, self.crosshair())},
            **self.axes)

        return figure

    def panel(self,
              index: int,
              laid: PanelLayout) -> Names:
        """A panel's axes, and a second vertical axis where it has one."""
        self.count += 1
        number = "" if self.count == 1 else str(self.count)
        x, y = f"x{number}", f"y{number}"
        across, up = domain(laid.box, self.layout.width, self.layout.height)
        grid = self.grid
        x_axis = axis(laid.x, across, y, self.paint, True, grid in ("x", "both"))
        y_axis = axis(laid.y, up, x, self.paint, False, grid in ("y", "both"))

        x_base = y_base = 0.0
        if laid.origin:
            x_axis.update(anchor="free", position=origin(0.0, up, laid.y.low, laid.y.high))
            y_axis.update(anchor="free", position=origin(0.0, across, laid.x.low, laid.x.high))
            x_base = origin(0.0, [0.0, 1.0], laid.y.low, laid.y.high)
            y_base = origin(0.0, [0.0, 1.0], laid.x.low, laid.x.high)

        # A grid of cells has no axis lines, as the image draws it.
        cells = any(isinstance(mark, CellsMark) and mark.panel == index for mark in self.marks)
        self.ticks += Spines(laid, self.layout.width, self.layout.height, self.paint,
                             lines=not cells).shapes()
        self.labels += tick_labels(laid.x, (x, y), "bottom", x_base, self.paint)
        self.labels += tick_labels(laid.y, (y, x), "left", y_base, self.paint)

        self.axes[Names.key(x)] = x_axis
        self.axes[Names.key(y)] = y_axis
        second = None

        if laid.y2 is not None:
            self.count += 1
            second = f"y{self.count}"
            right = axis(laid.y2, up, x, self.paint, False, False)
            right.update(overlaying=y, side="right")
            self.axes[Names.key(second)] = right
            self.labels += tick_labels(laid.y2, (second, x), "right", 1.0, self.paint)

        if laid.colour_bar is not None:
            self.colour_bar(index, laid.colour_bar)

        return Names(x, y, second)

    def colour_bar(self,
                   index: int,
                   bar: ColourBarLayout) -> None:
        """A heatmap's colour scale, on axes of its own beside the panel."""
        self.count += 1
        x, y = f"x{self.count}", f"y{self.count}"
        across, up = domain(bar.box, self.layout.width, self.layout.height)
        scale = axis(bar.axis, up, x, self.paint, False, False)
        scale.update(side="right")
        self.axes[Names.key(x)] = {"domain": across, "anchor": y, "visible": False,
                                   "range": [0, 1], "fixedrange": True}
        self.axes[Names.key(y)] = scale
        self.labels += tick_labels(bar.axis, (y, x), "right", 1.0, self.paint)
        cells = next(mark for mark in self.marks
                     if isinstance(mark, CellsMark) and mark.panel == index)
        stops = [self.paint.ink(stop) for stop in cells.stops]
        last = max(len(stops) - 1, 1)
        low, high = bar.axis.low, bar.axis.high
        steps = [low + (high - low) * step / (BAR_STEPS - 1) for step in range(BAR_STEPS)]
        self.extra.append({
            "type": "heatmap", "xaxis": x, "yaxis": y, "showscale": False, "hoverinfo": "skip",
            "z": [[value] for value in steps], "x": [0.5], "y": steps, "zsmooth": "best",
            "zmin": low, "zmax": high,
            "colorscale": [[step / last, stop] for step, stop in enumerate(stops)]})

    def hovermode(self) -> str:
        """Every series at a date or category, or the one point or cell under
        the pointer."""
        if any(isinstance(mark, CellsMark) or (isinstance(mark, PointMark) and mark.name)
               for mark in self.marks):
            return "closest"
        if any(isinstance(mark, BarMark) and mark.horizontal for mark in self.marks):
            return "y unified"

        return "x unified"

    def crosshair(self) -> str:
        """The popup's crosshair: down through a date, through a point both
        ways, or none."""
        if any(isinstance(mark, CellsMark) for mark in self.marks):
            return "none"
        if any(isinstance(mark, PointMark) and mark.name for mark in self.marks):
            return "both"
        if any(panel.x.kind == "date" for panel in self.panels):
            return "x"

        return "none"
