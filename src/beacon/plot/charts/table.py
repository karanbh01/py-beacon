# src/beacon/plot/charts/table.py
"""
A DataFrame under the Beacon frame: `Table`, as an image or as HTML.
"""
# BN-302.
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .. import colormaps
from ..design import StyleArgument
from ..layout import Layout
from ..tables.model import (
    Cell,
    Column,
    Format,
    Heat,
    Row,
    RowGroup,
    TableModel,
    column_groups,
    formatter,
    frame_of,
    heat_fill,
    is_number,
)
from .base import Chart

# A heatmap's options, and their defaults.
HEATMAP = {"colormap": "blue-red", "reverse": False, "scale": "column", "center": None,
           "fade": False, "values": True, "shading": "squares", "blend": "columns"}


class Table(Chart):
    """A DataFrame under the Beacon frame: the mark, the heading, the table
    and the notes line. Its look (rules, boxes, spacing, text and colours) is
    the style's `table` group.

    Args:
        data: The table: its index names the rows, its columns the columns.
        title: The heading.
        subtitle: Beneath it.
        notes: The notes line, after "Notes:".
        source: Where the data came from, written first in the notes.
        formats: Each column's format, or a group's for all its columns: a
            format string for a value's size such as "{:,.1f}", a mapping of
            decimals, prefix, suffix, thousands and percent, or a function.
        negatives: "minus" writes a negative with a minus sign; "brackets"
            writes -1.2 as (1.2).
        signed: Columns, or groups, whose values are coloured by sign.
        groups: Column groups: {heading: [columns]}, or {heading: how many
            columns} in order. Each heading sits centred over its columns.
        row_groups: Row groups: {heading: [rows]}. Rows in no group follow.
        row_group_headings: Whether each row group's heading is shown: in a
            column at the table's left, merged and centred beside its rows.
        highlight: Rows set in bold.
        heatmap: Colours a column, or a column group, by value: {"column":
            name} or {"group": heading}, with "colormap" (a map's name or
            hex colours; "blue-red" by default), "reverse", "scale" ("column"
            or "group"), "center" (such as 0), "fade" (near-neutral cells
            take the table's background), "values" (False hides the
            numbers), "shading" ("squares" or "blended") and "blend"
            ("columns", "rows" or "both").
        preset: The look's rules and boxes: "lines", "plain" or "boxed"; the
            style's by default.
        style: Settings for this table alone: a Style, or settings by name.

    Raises:
        ValueError: If an option names a column, group or row the table does
            not have.
    """

    kind = "table"

    def __init__(self,
                 data: pd.DataFrame,
                 title: str = "",
                 subtitle: str = "",
                 notes: str = "",
                 source: str | None = None,
                 formats: Mapping[str, Format] | None = None,
                 negatives: str = "minus",
                 signed: Sequence[str] = (),
                 groups: Mapping[str, int | Sequence[str]] | None = None,
                 row_groups: Mapping[str, Sequence[str]] | None = None,
                 row_group_headings: bool = False,
                 highlight: Sequence[str] = (),
                 heatmap: Mapping[str, Any] | None = None,
                 preset: str | None = None,
                 style: StyleArgument = None):
        super().__init__(title, subtitle, notes, source, style)
        if preset is not None:
            self.style = self.style.replace(table={"preset": preset})

        if negatives not in ("minus", "brackets"):
            raise ValueError(f"negatives must be 'minus' or 'brackets', not {negatives!r}")

        self.data = frame_of(data)
        self.formats = dict(formats or {})
        self.negatives = negatives
        self.signed = [str(name) for name in signed]
        self.groups = dict(groups or {})
        self.row_groups = {str(label): [str(name) for name in names]
                           for label, names in (row_groups or {}).items()}
        self.row_group_headings = row_group_headings
        self.highlight = {str(name) for name in highlight}
        self.heatmap = {**HEATMAP, **heatmap} if heatmap else None
        self.model = self._model()

    def _model(self) -> TableModel:
        names = [str(column) for column in self.data.columns]
        headings, membership = column_groups(names, self.groups)
        columns = tuple(self._column(name, index, names, membership)
                        for index, name in enumerate(names))
        heat_columns = self._heat_columns(names, headings, membership)
        ranges = self._heat_ranges(names, heat_columns, membership)
        rows = {str(name): self._row(str(name), values, names, membership, heat_columns, ranges)
                for name, values in zip(self.data.index, self.data.to_numpy(dtype=object),
                                        strict=True)}

        heat = self.heatmap or {}
        return TableModel(columns=columns, groups=headings,
                          row_groups=self._grouped(rows),
                          group_headings=self.row_group_headings,
                          heat=Heat(tuple(heat_columns), heat.get("shading") == "blended",
                                    str(heat.get("blend", "columns"))))

    def _column(self,
                name: str,
                index: int,
                names: list[str],
                membership: dict[str, int]) -> Column:
        group = membership.get(name)
        before = membership.get(names[index - 1]) if index > 0 else None
        after = membership.get(names[index + 1]) if index + 1 < len(names) else None

        return Column(name, group, group is not None and before != group,
                      group is not None and after != group)

    def _format(self,
                name: str,
                membership: dict[str, int]) -> Any:
        group = membership.get(name)
        heading = list(self.groups)[group] if group is not None else None
        chosen = self.formats.get(name, self.formats.get(heading) if heading else None)
        column = self.data[name]
        whole = all(is_number(value) and float(value).is_integer() for value in column.dropna())

        return formatter(chosen, self.negatives, whole)

    def _is_signed(self,
                   name: str,
                   membership: dict[str, int]) -> bool:
        group = membership.get(name)
        heading = list(self.groups)[group] if group is not None else None

        return name in self.signed or (heading is not None and heading in self.signed)

    def _row(self,
             name: str,
             values: Any,
             names: list[str],
             membership: dict[str, int],
             heat_columns: list[int],
             ranges: dict[int, tuple[float, float]]) -> Row:
        bold = name in self.highlight
        cells = []

        for index, (column, value) in enumerate(zip(names, values, strict=True)):
            number = float(value) if is_number(value) and value == value else None
            text = self._format(column, membership)(value)
            colour: str | None = "text-primary"
            if number is not None and self._is_signed(column, membership):
                colour = "success" if number >= 0 else "danger"
            fill = None
            hidden = False
            if index in heat_columns and number is not None:
                low, high = ranges[index]
                heat = self.heatmap or {}
                fill = heat_fill(number, low, high, self._stops(), bool(heat["fade"]),
                                 heat["center"] is not None)
                colour = None
                hidden = not heat["values"]
            cells.append(Cell(text, number, colour, fill, bold, hidden))

        return Row(name, tuple(cells), bold)

    def _stops(self) -> tuple[str, ...]:
        heat = self.heatmap or {}
        found = colormaps.stops(heat["colormap"])

        return tuple(reversed(found)) if heat.get("reverse") else found

    def _heat_columns(self,
                      names: list[str],
                      headings: tuple[str, ...],
                      membership: dict[str, int]) -> list[int]:
        if not self.heatmap:
            return []

        if "column" in self.heatmap:
            chosen = str(self.heatmap["column"])
            if chosen not in names:
                raise ValueError(f"heatmap names column {chosen!r}, which the table does not have")
            return [names.index(chosen)]

        heading = str(self.heatmap.get("group"))
        if heading not in headings:
            raise ValueError(f"heatmap names group {heading!r}, which the table does not have")

        return [index for index, name in enumerate(names)
                if membership.get(name) == headings.index(heading)]

    def _heat_ranges(self,
                     names: list[str],
                     heat_columns: list[int],
                     membership: dict[str, int]) -> dict[int, tuple[float, float]]:
        """Each heatmap column's scale: its own values, or its whole group's,
        made even about the centre where one is set."""
        if not heat_columns:
            return {}

        heat = self.heatmap or {}
        together = heat.get("scale") == "group" and "group" in heat
        ranges = {}

        for index in heat_columns:
            chosen = heat_columns if together else [index]
            values = pd.to_numeric(self.data.iloc[:, chosen].stack(), errors="coerce").dropna()
            low, high = (float(values.min()), float(values.max())) if len(values) else (0.0, 1.0)
            if heat.get("center") is not None:
                centre = float(heat["center"])
                far = max(abs(low - centre), abs(high - centre))
                low, high = centre - far, centre + far
            ranges[index] = (low, high if high > low else low + 1.0)

        return ranges

    def _grouped(self,
                 rows: dict[str, Row]) -> tuple[RowGroup, ...]:
        if not self.row_groups:
            return (RowGroup("", tuple(rows.values())),)

        placed: set[str] = set()
        groups = []
        for label, names in self.row_groups.items():
            missing = [name for name in names if name not in rows]
            if missing:
                raise ValueError(f"row group {label!r} names {missing[0]!r}, which is not a row")
            groups.append(RowGroup(label, tuple(rows[name] for name in names)))
            placed.update(names)

        rest = tuple(row for name, row in rows.items() if name not in placed)
        if rest:
            groups.append(RowGroup("", rest))

        return tuple(groups)

    @property
    def layout(self) -> Layout:
        """Where the table's frame goes: its heading, notes, accent bar and
        mark."""
        from ..tables.geometry import measure

        return measure(self).layout

    def _drawing(self) -> Any:
        """The table drawn with matplotlib, drawing it the first time."""
        if self._drawn is None:
            from ..tables.paint import paint_table

            self._drawn = paint_table(self)

        return self._drawn

    def show(self) -> None:
        """Show the table: in a notebook, beneath the cell; otherwise in a
        window."""
        from ..tables.paint import show_table

        show_table(self)

    def interactive(self) -> Any:
        """A table has no plotly figure: its HTML (`to_html()`) is its
        interactive form, with columns that sort.

        Raises:
            TypeError: Always.
        """
        raise TypeError("a table's interactive form is its HTML: use .to_html()")

    def to_html(self) -> str:
        """The table as HTML: the heading, the table and the notes, in the
        table's theme, with columns that sort when their headings are
        clicked."""
        from ..tables.html import table_html

        return table_html(self)

    def _repr_html_(self) -> str:
        return self.to_html()
