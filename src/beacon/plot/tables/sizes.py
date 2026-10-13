# src/beacon/plot/tables/sizes.py
"""
The pieces a table is drawn from, in inches, and its sizes in a style.
"""
# BN-302.
from dataclasses import dataclass, field
from typing import Any

from ..layout import Box, Layout, Text
from ..panels import SCALE
from .model import Fill, TableModel

# A line of the table's text, as a multiple of its size.
LINE = 1.25

# The weights the values and the group headings are set in, and a
# highlighted row's.
VALUE_WEIGHT = 300
GROUP_WEIGHT = 400
BOLD_WEIGHT = 500


@dataclass(frozen=True)
class Rule:
    """A straight line: from (x0, y0) to (x1, y1) in inches, its weight in
    points, its colour by name or hex."""

    x0: float
    y0: float
    x1: float
    y1: float
    width: float
    colour: str


@dataclass(frozen=True)
class Patch:
    """A filled box: a stripe or a heatmap cell."""

    box: Box
    colour: str
    fill: Fill | None = None


@dataclass(frozen=True)
class Blend:
    """Heatmap cells whose colours blend into each other: a grid of fills,
    row by row, over a box, blended across, down or both."""

    box: Box
    fills: tuple[tuple[Fill, ...], ...]
    direction: str


@dataclass
class Geometry:
    """A table drawn as an image: the frame's layout, and the table's text,
    rules and fills, and every column's width with the spaces between them,
    in inches."""

    layout: Layout
    texts: list[Text] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    patches: list[Patch] = field(default_factory=list)
    blends: list[Blend] = field(default_factory=list)
    heat_texts: list[Any] = field(default_factory=list)
    widths: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Sizes:
    """A table's text sizes, paddings and line weights, scaled."""

    body: float
    head: float
    group: float
    row_pad: float
    cell_pad: float
    name_space: float
    rule: float
    row_rule: float
    box: float
    font: str

    @classmethod
    def of(cls, style: Any) -> "Sizes":
        table, zoom = style.table, style.size.scale
        text = zoom * style.text.scale
        k = SCALE * zoom

        return cls(table.body_size * text, table.header_size * text, table.group_size * text,
                   table.row_padding * zoom, table.cell_padding * zoom, table.name_space * zoom,
                   table.rule_width * k, table.row_rule_width * k, table.box_width * k,
                   style.text.font)


# The space either side of a row group's heading, in cell paddings: also
# between the rule under it and the rule between the rows beside it.
ROW_RULE_GAP = 1.5


def rule_gap(sizes: Sizes) -> float:
    """The space either side of a row group's heading, in inches."""
    return ROW_RULE_GAP * sizes.cell_pad / 72.0


def blocks(model: TableModel,
           rules: dict[str, bool]) -> bool:
    """Whether the row groups stand apart, boxed or headed: then stripes,
    row rules and blended heatmaps start afresh in each."""
    return model.grouped_rows and (rules["row_boxes"] or model.group_headings)


def gap_before(model: TableModel,
               index: int,
               rules: dict[str, bool]) -> bool:
    """Whether a narrow gap comes before a value column: where column groups
    are boxed, before each group but the first column."""
    column = model.columns[index]

    return rules["group_boxes"] and column.group is not None and column.first and index > 0
