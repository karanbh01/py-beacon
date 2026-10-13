# src/beacon/plot/tables/html.py
"""
A table as HTML: the heading, the table and the notes in the table's theme,
with columns that sort when their headings are clicked.

Each table's look is a style block scoped to its own id, so rules, boxes and
stripes follow the rows when they are sorted. Rows sort within their row
groups. A script sorts the rows and blends a heatmap's colours; without it
the table still reads, its heatmap cells each their own colour.
"""
# BN-302.
import html
import uuid
from typing import Any

from ...tokens import DARK
from ..arrange import (
    BAR_GAP,
    BAR_WIDTH,
    HEADING_LINE_HEIGHT,
    MARGIN_INCHES,
    MARK_ASPECT,
    MARK_INCHES,
    NOTES_GAP_INCHES,
    NOTES_LINE_HEIGHT,
    PLOT_GAP_INCHES,
    SUBTITLE_WEIGHT,
    TITLE_WEIGHT,
)
from ..colormaps import ink_on
from ..colours import resolve
from ..mark import BETA_PATH
from ..themes import theme_named
from .draw import nearest
from .fonts import css_family, font_faces
from .geometry import measure
from .model import Fill, Row, TableModel, over
from .script import SCRIPT
from .sizes import LINE, VALUE_WEIGHT, Sizes, blocks, gap_before

# A faded heatmap cell paler than this keeps the table's own text colour.
FAINT = 0.45

# What a faded heatmap cell is laid over where the theme draws no background.
GROUND = {"light": "#ffffff", "dark": "#232323"}

# The beta mark's outline, in its own units, y up.
BETA_BOX = (21.8, 34.5)

def table_html(table: Any) -> str:
    """A table as HTML, with its frame."""
    return _Writer(table).write()


class _Writer:
    """Writes one table's HTML."""

    def __init__(self, table: Any):
        self.table = table
        self.model: TableModel = table.model
        self.style = table.style
        self.settings = table.style.table
        self.rules = self.settings.rules()
        self.theme = theme_named(self.style.theme)
        self.sizes = Sizes.of(self.style)
        self.id = f"beacon-table-{uuid.uuid4().hex[:8]}"
        # The row groups' headings, in a column of their own at the left.
        self.side = self.model.group_headings and any(group.label
                                                      for group in self.model.row_groups)
        self.grouped = blocks(self.model, self.rules)
        self.ground = (self.theme.canvas if self.theme.canvas is not None
                       else GROUND["dark" if self.theme.mode == DARK else "light"])

    def ink(self, name: str) -> str:
        return resolve(name, self.theme, self.style)

    def write(self) -> str:
        geometry = measure(self.table)
        width = geometry.layout.width * 72.0
        self.widths = geometry.widths
        margin = MARGIN_INCHES * self.style.size.scale * 72.0
        canvas = self.theme.canvas
        background = f"background:{canvas};" if canvas is not None else ""
        font = css_family(self.style.text.font)
        # At least its designed width, wider where a stand-in font is wider.
        frame = (f'<div style="display:inline-block;min-width:min({width:.1f}pt,100%);'
                 f'max-width:100%;box-sizing:border-box;'
                 f'padding:{margin:.1f}pt;{background}color:{self.ink("text-primary")};'
                 f'font-family:{font};font-weight:{VALUE_WEIGHT};line-height:{LINE};'
                 # Only the embedded faces, never a bold or italic made up
                 # from them, as the image is set.
                 f'font-synthesis:none">')

        return "".join([
            f"<style>{font_faces(self.table)}{self._css()}</style>", frame, self._heading(),
            f'<div style="padding-left:{self._indent():.1f}pt;overflow:auto hidden">',
            self._table(), self._notes(), "</div></div>",
            f"<script>{SCRIPT.replace('TABLE_ID', self.id)}</script>",
        ])

    def _indent(self) -> float:
        heading = self.style.heading
        if not heading.mark:
            return 0.0

        return float(MARK_INCHES * heading.scale * MARK_ASPECT * 72.0 + BAR_GAP * heading.scale)

    def _heading(self) -> str:
        heading, table = self.style.heading, self.table
        scale = heading.scale
        parts = []

        if heading.mark:
            tall = MARK_INCHES * scale * 72.0
            parts.append(f'<svg viewBox="0 0 {BETA_BOX[0]} {BETA_BOX[1]}" aria-hidden="true" '
                         f'style="width:{tall * MARK_ASPECT:.2f}pt;height:{tall:.2f}pt;flex:none">'
                         f'<path d="{BETA_PATH}" fill="{self.ink("text-primary")}" '
                         f'transform="translate(0 {BETA_BOX[1]}) scale(1 -1)"/></svg>')
        if heading.accent_bar:
            parts.append(f'<span style="align-self:stretch;width:{BAR_WIDTH * scale:.2f}pt;'
                         f'background:{self.ink("accent-bar")};flex:none"></span>')

        title = (f'<div style="font-family:{css_family(heading.title_font)};'
                 f'font-weight:{TITLE_WEIGHT};font-size:{heading.title_size * scale:.2f}pt;'
                 f'line-height:{HEADING_LINE_HEIGHT}">{html.escape(table.title)}</div>')
        subtitle = ""
        if table.subtitle:
            italic = "font-style:italic;" if heading.subtitle_italic else ""
            subtitle = (f'<div style="font-family:{css_family(heading.subtitle_font)};'
                        f'font-weight:{SUBTITLE_WEIGHT};{italic}'
                        f'font-size:{heading.subtitle_size * scale:.2f}pt;'
                        f'line-height:{HEADING_LINE_HEIGHT};color:{self.ink("text-secondary")}">'
                        f'{html.escape(table.subtitle)}</div>')
        parts.append(f'<div style="display:grid;gap:{0.5 * scale:.2f}pt">{title}{subtitle}</div>')
        below = PLOT_GAP_INCHES * self.style.size.scale * 72.0

        return (f'<div style="display:flex;align-items:center;gap:{BAR_GAP * scale:.2f}pt;'
                f'margin-bottom:{below:.1f}pt">{"".join(parts)}</div>')

    def _notes(self) -> str:
        notes = self.table._notes()
        if not (notes and self.style.notes.show):
            return ""

        size = self.style.notes.size * self.style.size.scale * self.style.text.scale
        italic = "font-style:italic;" if self.style.notes.italic else ""
        gap = NOTES_GAP_INCHES * self.style.size.scale * 72.0

        return (f'<div style="margin-top:{gap:.1f}pt;font-size:{size:.2f}pt;{italic}'
                f'line-height:{NOTES_LINE_HEIGHT}">Notes: {html.escape(notes)}</div>')

    def _css(self) -> str:
        from .css import table_css

        return table_css(self)

    def _table(self) -> str:
        model = self.model
        heat = model.heat
        blend = f' data-blend="{heat.blend}"' if heat.blended and heat.columns else ""
        grouped = "1" if self.grouped else "0"
        stripes = "1" if self.settings.stripes else "0"
        head = self._group_row() + self._header_row()

        return (f'<table id="{self.id}"{blend} data-blocks="{grouped}" data-stripes="{stripes}">'
                f"{self._columns()}<thead>{head}</thead>{self._bodies()}</table>")

    def _columns(self) -> str:
        """Each column at the image's width, so the two lay out alike rather
        than the browser sharing out the spare width its own way."""
        cols = "".join(f'<col style="width:{width * 72.0:.2f}pt">' for width in self.widths)

        return f"<colgroup>{cols}</colgroup>"

    def _gap(self,
             tag: str,
             index: int) -> str:
        if not gap_before(self.model, index, self.rules):
            return ""

        return f'<{tag} class="gap" aria-hidden="true"></{tag}>'

    def _group_row(self) -> str:
        model = self.model
        if not model.groups:
            return ""

        cells, index = [self._corner(), '<th class="n"></th>'], 0
        while index < len(model.columns):
            group = model.columns[index].group
            span = 1
            while (index + span < len(model.columns)
                   and group is not None and model.columns[index + span].group == group
                   and not model.columns[index + span].first):
                span += 1
            heading = html.escape(model.groups[group]) if group is not None else ""
            classes = "grp" + (" gs" if index and model.columns[index].first else "")
            cells.append(f'{self._gap("th", index)}<th class="{classes}" colspan="{span}">'
                         f"{heading}</th>")
            index += span

        return f'<tr class="groups">{"".join(cells)}</tr>'

    def _corner(self) -> str:
        return '<th class="rgh"></th><th class="gap rgap"></th>' if self.side else ""

    def _header_row(self) -> str:
        cells = [self._corner(), '<th class="n"></th>']
        for index, column in enumerate(self.model.columns):
            cells.append(f'{self._gap("th", index)}<th class="c{self._edges(index)}" '
                         f'data-col="{index}" title="Sort">{html.escape(column.name)}</th>')

        return f'<tr class="header">{"".join(cells)}</tr>'

    def _edges(self,
               index: int) -> str:
        """A value column's classes: the first or last of the values, the
        first or last of its group, and the start of a group after the
        first."""
        columns = self.model.columns
        column = columns[index]
        found = []
        if index == 0:
            found.append("first")
        if index == len(columns) - 1:
            found.append("last")
        if column.group is not None and column.first:
            found.append("gf")
        if column.group is not None and column.last:
            found.append("gl")
        if index and column.first and column.group is not None:
            found.append("gs")

        return "".join(f" {name}" for name in found)

    def _bodies(self) -> str:
        model = self.model
        width = (1 + 2 * int(self.side) + len(model.columns)
                 + sum(gap_before(model, index, self.rules) for index in range(len(model.columns))))
        bodies = []

        for index, group in enumerate(model.row_groups):
            if self.rules["row_boxes"] and index:
                bodies.append(f'<tbody class="space" aria-hidden="true"><tr>'
                              f'<td colspan="{width}"></td></tr></tbody>')
            label = ""
            if self.side:
                label = (f'<td class="rg" rowspan="{len(group.rows)}">'
                         f"{html.escape(group.label)}</td>")
            rows = "".join(self._row(row, label if place == 0 else "", self.side)
                           for place, row in enumerate(group.rows))
            bodies.append(f"<tbody>{rows}</tbody>")

        return "".join(bodies)

    def _row(self,
             row: Row,
             label: str,
             side: bool) -> str:
        bold = " b" if row.bold else ""
        space = '<td class="gap rgap"></td>' if side else ""
        cells = [label, space, f'<td class="n">{html.escape(row.name)}</td>']

        stand_ins = self._stand_ins(row)
        for index, cell in enumerate(row.cells):
            value = "" if cell.value is None else repr(cell.value)
            colour, fill = self._look(cell.colour, cell.fill, cell.hidden)
            heat = " h" if index in self.model.heat.columns else ""
            painted = f' data-fill="{fill}"' if fill else ""
            if index in stand_ins:
                painted += f' data-mix="{stand_ins[index]}"'
            background = f"background:{fill};" if fill else ""
            cells.append(f'{self._gap("td", index)}<td class="v{heat}{self._edges(index)}" '
                         f'data-v="{value}"{painted} style="{background}color:{colour}">'
                         f"{html.escape(cell.text)}</td>")

        return f'<tr class="r{bold}">{"".join(cells)}</tr>'

    def _stand_ins(self,
                   row: Row) -> dict[int, str]:
        """Each heatmap cell's colour as a blend across and down takes it: a
        missing cell's is its nearest neighbour's along the row."""
        columns = self.model.heat.columns
        if not (self.model.heat.blended and self.model.heat.blend == "both"):
            return {}

        fills = nearest([row.cells[column].fill for column in columns])
        return {column: _css(fill) if fill else "rgba(0, 0, 0, 0)"
                for column, fill in zip(columns, fills, strict=True)}

    def _look(self,
              colour: str | None,
              fill: Fill | None,
              hidden: bool) -> tuple[str, str]:
        """A value's text colour, and its heatmap fill as CSS, or ""."""
        if fill is None:
            return self.ink(colour or "text-primary"), ""

        painted = over(fill, self.ground)
        ink = self.ink("text-primary") if fill.alpha < FAINT else ink_on(painted)
        return "transparent" if hidden else ink, _css(fill)


def _css(fill: Fill) -> str:
    red, green, blue = (int(fill.colour[i:i + 2], 16) for i in (1, 3, 5))

    return f"rgba({red}, {green}, {blue}, {fill.alpha:.3f})"
