# src/beacon/plot/charts/base.py
"""
What every chart has: a heading, notes and a style, and a drawing made only
when it is first used.

Building a chart describes it and nothing more. It is drawn the first time
it is shown, saved, or its figure or axes are asked for, and a notebook shows
it when it is a cell's last value. So a chart that is never drawn costs
nothing, and the layout and marks it is drawn from need no renderer.
"""
# BN-299.
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from .. import design
from ..arrange import lay_out
from ..design import Style, StyleArgument
from ..layout import Description, Layout
from ..marks import Mark, Side
from ..style import figure_size

# The size a chart is drawn at in a notebook: its pixels per inch.
NOTEBOOK_DPI = 200
NOTEBOOK_SCALE = 100


@dataclass(frozen=True)
class Plan:
    """A chart as its renderers draw it: what the layout needs, and the
    marks inside the plot.

    Attributes:
        description: For the layout.
        marks: Drawn in order, by layer.
        grid: The chart's own gridlines, "x", "y", "both" or "none"; the
            style's grid replaces them where it sets one.
        front: Where a panel has a second axis, which side's marks are drawn
            over the other's.
    """

    description: Description
    marks: tuple[Mark, ...]
    grid: str = "none"
    front: Side = "left"


class Chart:
    """A chart in the Beacon frame: the mark, the heading, the plot and the
    notes line.

    Attributes:
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        style: The settings it is drawn in, fixed when it is built.
    """

    # The key of the chart's designed size (beacon.plot.style.FIGSIZE).
    kind: ClassVar[str] = "chart"

    def __init__(self,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 style: StyleArgument = None):
        self.title = title
        self.subtitle = subtitle
        self.notes = notes
        self.source = source
        self.style: Style = design.resolve(style)
        self._plan: Plan | None = None
        self._drawn: Any = None

    def _build(self) -> Plan:
        """The chart's plan, in its own style. Each chart class makes its
        own."""
        raise NotImplementedError

    @property
    def plan(self) -> Plan:
        """What the chart's renderers draw: its layout's description and its
        marks."""
        if self._plan is None:
            with design.scoped(self.style):
                self._plan = self._build()

        return self._plan

    @property
    def layout(self) -> Layout:
        """Where everything in the chart's frame goes."""
        return lay_out(self.plan.description, self.style)

    def _size(self) -> tuple[float, float]:
        """The figure's size, in inches, in the chart's style."""
        with design.scoped(self.style):
            return figure_size(self.kind)

    def _notes(self) -> str:
        """The notes line: the source, then the notes."""
        parts = [f"Source: {self.source.rstrip('.')}." if self.source else "", self.notes]

        return " ".join(part.strip() for part in parts if part and part.strip())

    def _drawing(self) -> Any:
        """The chart drawn with matplotlib, drawing it the first time."""
        if self._drawn is None:
            from ..paint import Painted, paint

            self._drawn = paint(self.plan, self.layout, self.style)
            assert isinstance(self._drawn, Painted)

        return self._drawn

    @property
    def figure(self) -> Any:
        """The matplotlib figure, drawn on first use."""
        return self._drawing().figure

    @property
    def ax(self) -> Any:
        """The matplotlib axes of the chart's top panel, drawn on first
        use."""
        return self._drawing().axes[0]

    @property
    def axes(self) -> list[Any]:
        """Every panel's matplotlib axes, top to bottom."""
        return list(self._drawing().axes)

    def show(self) -> None:
        """Show the chart: in a notebook, beneath the cell; otherwise in a
        window."""
        from ..paint import show

        show(self.plan, self.layout, self.style)

    def interactive(self) -> Any:
        """The chart as an interactive plotly figure, laid out as its image
        is, with a popup and crosshair under the pointer.

        Its `.write_html()` and `.to_html()` carry the chart's fonts,
        plotly.js and the popup's script, so the file looks the same offline.
        Needs the plot-interactive extra.

        Returns:
            BeaconFigure: A plotly figure.
        """
        from ..interactive.figure import interactive

        return interactive(self)

    def save(self,
             path: str | Path,
             dpi: int | None = None) -> Path:
        """Write the chart to a file at its designed size.

        Args:
            path: Where; its suffix picks the format: .png, .svg or .pdf.
            dpi: Pixels per inch for a PNG; the style's by default.

        Returns:
            Path: Where it was written.
        """
        target = Path(path)
        self._drawing().save(target, dpi=dpi or self.style.size.dpi)

        return target

    def _repr_png_(self) -> tuple[bytes, dict[str, int]]:
        """The chart as a PNG, for a notebook to show."""
        buffer = io.BytesIO()
        self._drawing().save(buffer, format="png", dpi=NOTEBOOK_DPI)
        width = round(float(self.figure.get_size_inches()[0]) * NOTEBOOK_SCALE)

        return buffer.getvalue(), {"width": width}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(title={self.title!r})"
