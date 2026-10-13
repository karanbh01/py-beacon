# src/beacon/plot/tables/geometry.py
"""
Where a table's text, rules, boxes and fills go as an image, and the frame
around it, worked out from the fonts without a renderer.

Each column is as wide as its widest text and the cell padding either side,
and the columns share any room the frame has to spare in proportion; a table
wider than the frame widens the figure. Each row is a line and the row
padding above and below it. The table starts at the accent bar, so it lines
up with the heading as a chart's plot does, and the notes follow it.
"""
# BN-302.
from typing import Any

from .. import metrics
from ..arrange import (
    BAR_GAP,
    MARGIN_INCHES,
    MARK_ASPECT,
    MARK_INCHES,
    NOTES_GAP_INCHES,
    NOTES_LINE_HEIGHT,
    PLOT_GAP_INCHES,
    framed,
    heading_of,
    wrap,
)
from ..layout import Description, Text
from ..metrics import Font
from ..style import sized
from .draw import Placer
from .model import TableModel
from .sizes import (
    BOLD_WEIGHT,
    GROUP_WEIGHT,
    LINE,
    VALUE_WEIGHT,
    Geometry,
    Sizes,
    gap_before,
    rule_gap,
)


def _wide(text: str,
          size: float,
          font: str,
          weight: int) -> float:
    return metrics.width(text, size, Font(font, weight))


def measure(table: Any) -> Geometry:
    """A table's geometry as an image, and its frame."""
    style, model = table.style, table.model
    sizes = Sizes.of(style)
    columns = _column_widths(model, style, sizes)
    label = _label_width(model, sizes)
    width, _ = _designed(style)
    margin = MARGIN_INCHES * style.size.scale
    heading = style.heading
    indent = (MARK_INCHES * heading.scale * MARK_ASPECT + BAR_GAP * heading.scale / 72.0
              if heading.mark else 0.0)
    left = margin + indent
    room = width - left - margin
    rules = style.table.rules()
    gaps = 2.0 * sizes.cell_pad / 72.0 * sum(gap_before(model, index, rules)
                                             for index in range(len(model.columns)))
    natural = label + sum(columns) + gaps

    # The row groups' headings and the row names keep their widths; the value
    # columns share the room to spare.
    fixed = label + columns[0] + gaps
    if natural < room:
        columns = [columns[0], *(column * (room - fixed) / (natural - fixed)
                                 for column in columns[1:])]
    else:
        width = left + natural + margin

    rows = _row_heights(model, sizes, rules)
    table_height = sum(height for _, height in rows)
    notes_size = style.notes.size * style.size.scale * style.text.scale
    notes_font = Font(style.text.font, 300, style.notes.italic)
    notes = table._notes() if style.notes.show else ""
    across = label + sum(columns) + gaps
    lines = wrap(f"Notes: {notes}", across, notes_size, notes_font) if notes else []
    probe = heading_of(Description(size=(width, 10.0), title=table.title,
                                   subtitle=table.subtitle), style)
    above = 10.0 - probe.bottom + PLOT_GAP_INCHES * style.size.scale
    below = ((NOTES_GAP_INCHES * style.size.scale
              + len(lines) * notes_size * NOTES_LINE_HEIGHT / 72.0) if lines else 0.0) + margin
    height = above + table_height + below
    description = Description(size=(width, height), title=table.title, subtitle=table.subtitle)
    top = height - above
    notes_text = (Text("\n".join(lines), left, top - table_height - NOTES_GAP_INCHES
                       * style.size.scale, notes_size, notes_font, "text-primary", ha="left",
                       va="top", line_height=NOTES_LINE_HEIGHT, id="beacon-notes")
                  if lines else None)
    layout = framed(description, style, heading_of(description, style), [], None, left,
                    notes_text)
    geometry = Geometry(layout)
    Placer(model, style, sizes, [label, *columns], rows, left, top, geometry).place()

    return geometry


def _designed(style: Any) -> tuple[float, float]:
    from .. import design

    with design.scoped(style):
        return sized(6.0, 4.5)


def _column_widths(model: TableModel,
                   style: Any,
                   sizes: Sizes) -> list[float]:
    """Each column's width, in inches: the row names', then each value
    column's, with a narrow gap before each group after the first where
    groups are boxed."""
    table = style.table
    pad = 2.0 * sizes.cell_pad / 72.0
    names = [_wide(row.name, sizes.body, sizes.font,
                   BOLD_WEIGHT if row.bold else table.name_weight)
             for group in model.row_groups for row in group.rows]
    widths = [max(names) + (sizes.name_space + sizes.cell_pad) / 72.0]

    for index, column in enumerate(model.columns):
        values = [_wide(row.cells[index].text, sizes.body, sizes.font,
                        BOLD_WEIGHT if row.bold else VALUE_WEIGHT)
                  for group in model.row_groups for row in group.rows]
        widths.append(max([_wide(column.name, sizes.head, sizes.font, table.header_weight),
                           *values]) + pad)

    for group, heading in enumerate(model.groups):
        members = [index + 1 for index, column in enumerate(model.columns)
                   if column.group == group]
        need = _wide(heading, sizes.group, sizes.font, GROUP_WEIGHT) + pad
        have = sum(widths[index] for index in members)
        for index in members:
            widths[index] += max(need - have, 0.0) / len(members)

    return widths


def _label_width(model: TableModel,
                 sizes: Sizes) -> float:
    """The width of the column the row groups' headings stand in, at the
    table's left, with the space after it; none where they are not shown.
    The widest heading is as far from the column's rule as its rule is from
    the rows."""
    labels = [_wide(group.label, sizes.group, sizes.font, GROUP_WEIGHT)
              for group in model.row_groups if group.label]
    if not (model.group_headings and labels):
        return 0.0

    return max(labels) + 3.0 * rule_gap(sizes)


def _row_heights(model: TableModel,
                 sizes: Sizes,
                 rules: dict[str, bool]) -> list[tuple[str, float]]:
    """Each row's kind and height, top to bottom, in inches: boxed row
    groups with no headings are kept apart by a narrow space."""
    line = LINE / 72.0
    rows: list[tuple[str, float]] = []

    if model.groups:
        rows.append(("groups", sizes.group * line + 1.5 * sizes.row_pad / 72.0))
    rows.append(("header", sizes.head * line + 2.0 * sizes.row_pad / 72.0))

    for index, group in enumerate(model.row_groups):
        if rules["row_boxes"] and index:
            rows.append(("space", 2.0 * sizes.row_pad / 72.0))
        rows.extend(("row", sizes.body * line + 2.0 * sizes.row_pad / 72.0)
                    for _ in group.rows)

    return rows
