# src/beacon/plot/charts/heatmap.py
"""
A table of values drawn as colour: `Heatmap`.
"""
# BN-299.
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from .. import colormaps
from .. import style as sizes
from ..design import StyleArgument
from ..layout import Axis, ColourBar, Description, Panel
from ..marks import CellsMark, Mark, TextMark
from .base import Chart, Plan
from .data import percent, plain

# The space between cells, as a share of a cell.
GAP = 0.04

# Where the slanted column names sit under the cells, in inches before
# scaling, and the angle they slant at.
SLANT = 45.0
SLANTED_ROOM = 0.5


class Heatmap(Chart):
    """A table of values as colour: one cell per row and column.

    The frame is cropped to what the chart draws, at the same scale, so a
    square table does not leave the frame's width empty.

    Args:
        data: A DataFrame: its index names the rows, top to bottom, and its
            columns the columns, left to right.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        colormap: A colour map's name (beacon.plot.colormaps), "_r" on the
            end to reverse it, or a list of hex colours from low to high. By
            default "blue-red" around zero for values either side of it, and
            "blues" otherwise.
        vmin: The value at the map's low end; by default the data's, or
            minus the largest size either side of zero.
        vmax: The value at its high end.
        values: Whether each cell's value is written in it.
        decimals: Decimals in the written values and the colour bar's ends.
        percent: Whether values are shares, written as percentages.
        lower: Whether only the lower triangle is drawn, for a symmetric
            table such as correlations.
        shading: "squares" draws each cell as its own square; "blended"
            blends each colour into the next.
        gap: The space between cells, as a share of a cell; 0 for none.
        colorbar: Whether the colour scale is drawn beside, labelled at its
            ends.
        style: Settings for this chart alone: a Style, or settings by name.
    """

    kind = "heatmap"

    def __init__(self,
                 data: pd.DataFrame,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 colormap: str | Sequence[str] | None = None,
                 vmin: float | None = None,
                 vmax: float | None = None,
                 values: bool = True,
                 decimals: int = 2,
                 percent: bool = False,
                 lower: bool = False,
                 shading: Literal["squares", "blended"] = "squares",
                 gap: float = GAP,
                 colorbar: bool = True,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("a heatmap takes a DataFrame with at least one value")
        if shading not in ("squares", "blended"):
            raise ValueError(f"shading must be 'squares' or 'blended', not {shading!r}")

        self.data = data
        self.values = values
        self.decimals = decimals
        self.percent = percent
        self.lower = lower
        self.shading = shading
        self.gap = gap
        self.colorbar = colorbar
        self.low, self.high, default = _scale(data.to_numpy(dtype=float), vmin, vmax)
        self.stops = colormaps.stops(colormap if colormap is not None else default)

    def _value(self, value: float) -> str:
        return (percent(value, self.decimals) if self.percent
                else plain(value, self.decimals))

    def _build(self) -> Plan:
        table = self.data.to_numpy(dtype=float)
        rows, columns = table.shape
        cells = tuple(tuple(None if (self.lower and j > i) or not math.isfinite(table[i, j])
                            else float(table[i, j]) for j in range(columns))
                      for i in range(rows))
        written = tuple(tuple("" if value is None else self._value(value) for value in row)
                        for row in cells)
        marks: list[Mark] = [CellsMark(cells, self.stops, self.low, self.high,
                                       blended=self.shading == "blended", gap=self.gap,
                                       written=written)]

        if self.values:
            marks.extend(self._written(cells))

        bar = None
        if self.colorbar:
            bar = ColourBar(self.low, self.high,
                            ends=(self._value(self.low), self._value(self.high)))

        panel = Panel(x=Axis("fixed", -0.5, columns - 0.5,
                             ticks=tuple(float(j) for j in range(columns)),
                             names=tuple(str(name) for name in self.data.columns),
                             rotation=SLANT, align="right", tick_length=0.0),
                      y=Axis("fixed", -0.5, rows - 0.5,
                             ticks=tuple(float(i) for i in range(rows)),
                             names=tuple(str(name) for name in self.data.index),
                             inverted=True, tick_length=0.0),
                      equal=True, colour_bar=bar)
        description = Description(size=self._size(), title=self.title, subtitle=self.subtitle,
                                  notes=self._notes(), panels=(panel,),
                                  bottom_room=sizes.scaled(SLANTED_ROOM), crop=True)

        return Plan(description, tuple(marks))

    def _written(self,
                 cells: tuple[tuple[float | None, ...], ...]) -> list[Mark]:
        """Each cell's value, in a colour that reads on the cell."""
        span = (self.high - self.low) or 1.0
        written: list[Mark] = []

        for i, row in enumerate(cells):
            for j, value in enumerate(row):
                if value is None:
                    continue
                fill = colormaps.colour_at(self.stops, (value - self.low) / span)
                written.append(TextMark(self._value(value), float(j), float(i),
                                        colormaps.ink_on(fill), sizes.label_size(),
                                        ha="center", va="center"))

        return written


def _scale(table: np.ndarray,
           vmin: float | None,
           vmax: float | None) -> tuple[float, float, str]:
    """The map's ends, and the map a table takes by default: diverging about
    zero where values fall either side of it, one way otherwise."""
    present = table[np.isfinite(table)]
    low, high = (float(present.min()), float(present.max())) if present.size else (0.0, 1.0)

    if low < 0.0 < high:
        far = max(-low, high)
        return (vmin if vmin is not None else -far, vmax if vmax is not None else far,
                "blue-red")

    return (vmin if vmin is not None else low, vmax if vmax is not None else high, "blues")
