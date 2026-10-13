# src/beacon/plot/tables/draw.py
"""
A table's pieces placed in inches: its headings and values, its rules and
boxes, its stripes and its heatmap's fills.
"""
# BN-302.
from dataclasses import dataclass
from typing import Any

from ..layout import Box, Text
from ..metrics import Font
from .model import Fill, TableModel
from .sizes import (
    BOLD_WEIGHT,
    GROUP_WEIGHT,
    LINE,
    VALUE_WEIGHT,
    Blend,
    Patch,
    Rule,
    blocks,
    gap_before,
    rule_gap,
)

# A column group's rule starts this many cell paddings in from its left, so
# neighbouring groups' rules stay apart.
GROUP_RULE_INSET = 3.0


@dataclass(frozen=True)
class HeatText:
    """A heatmap cell's value, in an ink worked out on its fill when drawn."""

    text: Text
    fill: Fill


@dataclass(frozen=True)
class Span:
    """Where a row is, top and bottom, and what it is."""

    kind: str
    top: float
    bottom: float


class Placer:
    """Puts a table's pieces where they go."""

    def __init__(self,
                 model: TableModel,
                 style: Any,
                 sizes: Any,
                 columns: list[float],
                 rows: list[tuple[str, float]],
                 left: float,
                 top: float,
                 geometry: Any):
        self.model = model
        self.table = style.table
        self.rules = style.table.rules()
        self.grouped = blocks(model, self.rules)
        self.sizes = sizes
        self.geometry = geometry
        self.left = left
        # The row groups' headings, in a column of their own at the left.
        self.side = model.group_headings and columns[0] > 0
        self.labels = (left, left + columns[0] - (rule_gap(sizes) if self.side else 0.0))
        self.edges = self._edges(columns[1:], left + columns[0])
        geometry.widths = self._widths()
        self.right = self.edges[-1][1]
        self.spans = self._spans(rows, top)
        self.header = self.table.header_color or "text-secondary"
        self.rule_colour = self.table.rule_color or self.header
        self.row_colour = self.table.row_rule_color or "divider"

    def _edges(self,
               columns: list[float],
               left: float) -> list[tuple[float, float]]:
        """Each column's left and right edges: the row names', then each
        value column's, with a gap before each boxed group after the first."""
        gap = 2.0 * self.sizes.cell_pad / 72.0
        edges = [(left, left + columns[0])]
        x = left + columns[0]

        for index, width in enumerate(columns[1:]):
            if gap_before(self.model, index, self.rules):
                x += gap
            edges.append((x, x + width))
            x += width

        return edges

    def _widths(self) -> list[float]:
        """Every column's width left to right, in inches, the spaces between
        them included: what the HTML sets its columns to, so it lays out as
        the image does."""
        widths = []
        if self.side:
            widths += [self.labels[1] - self.labels[0], rule_gap(self.sizes)]
        widths.append(self.edges[0][1] - self.edges[0][0])

        for index, (left, right) in enumerate(self.edges[1:]):
            if gap_before(self.model, index, self.rules):
                widths.append(left - self.edges[index][1])
            widths.append(right - left)

        return widths

    def _spans(self,
               rows: list[tuple[str, float]],
               top: float) -> list[Span]:
        spans, y = [], top
        for kind, height in rows:
            spans.append(Span(kind, y, y - height))
            y -= height

        return spans

    def text(self,
             words: str,
             x: float,
             y: float,
             size: float,
             weight: int,
             colour: str,
             ha: str) -> Text:
        return Text(words, x, y, size, Font(self.sizes.font, weight), colour,
                    ha=ha, va="center")  # type: ignore[arg-type]

    def aligned(self,
                edges: tuple[float, float],
                align: str) -> tuple[float, str]:
        """Where text aligned in a cell sits across, and how it is anchored."""
        pad = self.sizes.cell_pad / 72.0
        if align == "left":
            return edges[0] + pad, "left"
        if align == "right":
            return edges[1] - pad, "right"

        return (edges[0] + edges[1]) / 2.0, "center"

    def place(self) -> None:
        body = [span for span in self.spans if span.kind == "row"]

        for span in self.spans:
            if span.kind == "groups":
                self._groups(span)
            elif span.kind == "header":
                self._header(span)

        index = 0
        parts: list[list[tuple[Span, Any]]] = []
        for group in self.model.row_groups:
            block = []
            for position, row in enumerate(group.rows):
                span = body[index]
                self._row(row, span, index, position)
                block.append((span, row))
                index += 1
            parts.append(block)
            if self.side and group.label:
                self._row_label(group.label, block[0][0].top, block[-1][0].bottom)
            if self.side and len(parts) > 1 and self.rules["group_rules"]:
                self._title_border(block[0][0].top)

        self._heat(parts)
        self._lines(body, parts)

    def _groups(self,
                span: Span) -> None:
        sizes = self.sizes
        y = span.top - (sizes.row_pad + sizes.group * LINE / 2.0) / 72.0

        for group, heading in enumerate(self.model.groups):
            members = [index for index, column in enumerate(self.model.columns)
                       if column.group == group]
            left, right = self.edges[members[0] + 1][0], self.edges[members[-1] + 1][1]
            self.geometry.texts.append(self.text(heading, (left + right) / 2.0, y, sizes.group,
                                                 GROUP_WEIGHT, "text-primary", "center"))
            if self.rules["group_rules"]:
                inset = GROUP_RULE_INSET * sizes.cell_pad / 72.0
                self._rule(left + inset, span.bottom, right, span.bottom, sizes.row_rule,
                           self.row_colour)

    def _header(self,
                span: Span) -> None:
        y = (span.top + span.bottom) / 2.0
        align = (self.table.number_align if self.table.header_align == "numbers"
                 else "center")

        for index, column in enumerate(self.model.columns):
            x, ha = self.aligned(self.edges[index + 1], align)
            self.geometry.texts.append(self.text(column.name, x, y, self.sizes.head,
                                                 self.table.header_weight, self.header, ha))

        if self.rules["header_rule"]:
            self._rule(self.edges[0][0], span.bottom, self.right, span.bottom, self.sizes.rule,
                       self.rule_colour)

    def _row_label(self,
                   label: str,
                   top: float,
                   bottom: float) -> None:
        """A row group's heading, centred beside its rows."""
        x = (self.labels[0] + self.labels[1]) / 2.0
        self.geometry.texts.append(self.text(label, x, (top + bottom) / 2.0, self.sizes.group,
                                             GROUP_WEIGHT, "text-primary", "center"))

    def _title_border(self,
                      y: float) -> None:
        """The rule between two row groups' headings, across their column and
        stopping short of the rule between the rows, as a column group's rule
        stops short of its neighbour's."""
        self._rule(self.labels[0], y, self.labels[1], y, self.sizes.row_rule, self.row_colour)

    def _row(self,
             row: Any,
             span: Span,
             index: int,
             position: int) -> None:
        sizes, y = self.sizes, (span.top + span.bottom) / 2.0
        striped = (position if self.grouped else index) % 2 == 1
        if self.table.stripes and striped:
            left = self.edges[0][0]
            self.geometry.patches.append(Patch(Box(left, span.bottom, self.right - left,
                                                   span.top - span.bottom),
                                               self.table.stripe_color or "divider"))

        x, ha = self.aligned(self.edges[0], self.table.name_align)
        if ha == "left":
            x = self.edges[0][0] + sizes.name_space / 72.0
        # Under row group headings the row names take the column headings'
        # colour, so the headings stand out from them.
        colour = self.header if self.side else "text-primary"
        self.geometry.texts.append(self.text(row.name, x, y, sizes.body,
                                             BOLD_WEIGHT if row.bold else self.table.name_weight,
                                             colour, ha))

        for column, cell in enumerate(row.cells):
            if cell.hidden:
                continue
            x, ha = self.aligned(self.edges[column + 1], self.table.number_align)
            text = self.text(cell.text, x, y, sizes.body,
                             BOLD_WEIGHT if cell.bold else VALUE_WEIGHT,
                             cell.colour or "text-primary", ha)
            if cell.fill is not None:
                self.geometry.heat_texts.append(HeatText(text, cell.fill))
            else:
                self.geometry.texts.append(text)

    def _heat(self,
              parts: list[list[tuple[Span, Any]]]) -> None:
        """The heatmap's fills: each cell its own square, or blended across,
        down or both, in blocks of rows."""
        heat = self.model.heat
        if not heat.columns:
            return

        rows = [pair for block in parts for pair in block]
        if not heat.blended:
            for span, row in rows:
                for column in heat.columns:
                    fill = row.cells[column].fill
                    if fill is not None:
                        self.geometry.patches.append(Patch(self._cell(span, column), "", fill))
            return

        groups = parts if self.grouped else [rows]
        left = self.edges[heat.columns[0] + 1][0]
        right = self.edges[heat.columns[-1] + 1][1]

        for block in groups:
            top, bottom = block[0][0].top, block[-1][0].bottom
            raw = [[row.cells[column].fill for column in heat.columns] for _, row in block]
            grid = _filled(raw)
            for (span, _), given in zip(block, raw, strict=True):
                self._blank(span, [column for column, fill in zip(heat.columns, given, strict=True)
                                   if fill is None])
            if heat.blend == "both":
                self.geometry.blends.append(Blend(Box(left, bottom, right - left, top - bottom),
                                                  grid, "both"))
            elif heat.blend == "rows":
                for offset, column in enumerate(heat.columns):
                    edge = self.edges[column + 1]
                    self.geometry.blends.append(Blend(
                        Box(edge[0], bottom, edge[1] - edge[0], top - bottom),
                        tuple((fills[offset],) for fills in grid), "rows"))
            else:
                for (span, _), fills in zip(block, grid, strict=True):
                    self.geometry.blends.append(Blend(
                        Box(left, span.bottom, right - left, span.top - span.bottom),
                        (fills,), "columns"))

    def _blank(self,
               span: Span,
               columns: list[int]) -> None:
        """Cells with no value, drawn over the blend in the background."""
        for column in columns:
            self.geometry.patches.append(Patch(self._cell(span, column), "",
                                               Fill("#000000", 0.0)))

    def _cell(self,
              span: Span,
              column: int) -> Box:
        left, right = self.edges[column + 1]

        return Box(left, span.bottom, right - left, span.top - span.bottom)

    def _lines(self,
               body: list[Span],
               parts: list[list[tuple[Span, Any]]]) -> None:
        """The rules between rows and under the table, the rules between
        column groups, and the boxes."""
        sizes, rules = self.sizes, self.rules
        firsts = {id(block[0][0]) for block in parts if block}

        if rules["row_rules"]:
            for span in body[1:]:
                self._row_rule(span, id(span) in firsts)

        if rules["bottom_rule"]:
            self._rule(self.edges[0][0], body[-1].bottom, self.right, body[-1].bottom, sizes.rule,
                       self.rule_colour)

        if rules["column_rules"]:
            top = self.spans[0].top
            for index, column in enumerate(self.model.columns):
                if column.first and index:
                    x = self.edges[index + 1][0]
                    self._rule(x, top, x, body[-1].bottom, sizes.row_rule, self.row_colour)

        values = (self.edges[1][0], self.edges[-1][1])
        if rules["box"]:
            self._box(values, body[0].top, body[-1].bottom)
        self._group_boxes(body, parts, values)

    def _row_rule(self,
                  span: Span,
                  first: bool) -> None:
        """The rule above a row, from the row names: none between row groups
        kept apart by boxes, but one between groups headed at the left."""
        if first and self.grouped and not self.side:
            return

        self._rule(self.edges[0][0], span.top, self.right, span.top, self.sizes.row_rule,
                   self.row_colour)

    def _group_boxes(self,
                     body: list[Span],
                     parts: list[list[tuple[Span, Any]]],
                     values: tuple[float, float]) -> None:
        rules = self.rules
        if not (rules["group_boxes"] or rules["row_boxes"]):
            return

        header = next(span for span in self.spans if span.kind == "header")
        heights = ([(block[0][0].top, block[-1][0].bottom) for block in parts if block]
                   if rules["row_boxes"] else [(body[0].top, body[-1].bottom)])
        widths = [values]
        if rules["group_boxes"]:
            widths = []
            for group in range(len(self.model.groups)):
                members = [index for index, column in enumerate(self.model.columns)
                           if column.group == group]
                widths.append((self.edges[members[0] + 1][0], self.edges[members[-1] + 1][1]))

        from_header = (rules["group_boxes"] and not rules["row_boxes"]
                       and self.table.group_box_scope == "headings")
        for top, bottom in heights:
            for across in widths:
                self._box(across, header.top if from_header else top, bottom)

    def _box(self,
             across: tuple[float, float],
             top: float,
             bottom: float) -> None:
        left, right = across
        width, colour = self.sizes.box, self.rule_colour
        for x0, y0, x1, y1 in ((left, top, right, top), (left, bottom, right, bottom),
                               (left, top, left, bottom), (right, top, right, bottom)):
            self._rule(x0, y0, x1, y1, width, colour)

    def _rule(self,
              x0: float,
              y0: float,
              x1: float,
              y1: float,
              width: float,
              colour: str) -> None:
        self.geometry.rules.append(Rule(x0, y0, x1, y1, width, colour))


def _filled(grid: list[list[Fill | None]]) -> tuple[tuple[Fill, ...], ...]:
    """A heatmap block's fills, each missing cell taking its nearest
    neighbour's, along its row first, so its neighbours blend into nothing
    that is not there; the missing cells themselves are drawn blank."""
    columns = [list(column) for column in zip(*grid, strict=True)]
    rows = [nearest(row) for row in grid]
    down = [nearest(column) for column in columns]
    stand_in = Fill("#000000", 0.0)

    return tuple(tuple(fill or rows[y][x] or down[x][y] or stand_in
                       for x, fill in enumerate(row)) for y, row in enumerate(grid))


def nearest(line: list[Fill | None]) -> list[Fill | None]:
    """Each place's fill, or its nearest filled neighbour's."""
    found = [(index, fill) for index, fill in enumerate(line) if fill is not None]
    if not found:
        return list(line)

    return [fill if fill is not None else min(found, key=lambda pair: abs(pair[0] - index))[1]
            for index, fill in enumerate(line)]
