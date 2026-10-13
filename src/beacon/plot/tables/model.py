# src/beacon/plot/tables/model.py
"""
A table's content as its renderers draw it: every cell's text, colour and
fill, its column groups and row groups, worked out once from a DataFrame and
the table's options, without a renderer.

Colours are names or hex codes (beacon.plot.colours), except a heatmap's
fills, which carry the colour map's colour and how strongly it shows, so a
faded cell can be laid over whatever the table's background is.
"""
# BN-302.
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .. import colormaps
from ..ticks import MINUS

# What a cell with no value says.
MISSING = "\N{EN DASH}"

# A value's format: a format string for its size ("{:,.1f}"), a mapping of
# decimals, prefix, suffix, thousands and percent, or a function.
Format = str | Mapping[str, Any] | Callable[[float], str]


@dataclass(frozen=True)
class Column:
    """One column of values: its heading and the group it is in."""

    name: str
    group: int | None = None
    first: bool = False
    last: bool = False


@dataclass(frozen=True)
class Fill:
    """A heatmap cell's colour: the colour map's colour and how strongly it
    shows over the table's background, 1 for fully."""

    colour: str
    alpha: float = 1.0


@dataclass(frozen=True)
class Cell:
    """One value cell as drawn.

    Attributes:
        text: What it says.
        value: The number, for sorting; None for text or a missing value.
        colour: Its text's colour, by name or hex; None for a heatmap cell's
            own ink, worked out on its fill.
        fill: Its heatmap fill, or None.
        bold: Whether its row is highlighted.
        hidden: Whether its text is left out (a heatmap of colour alone).
    """

    text: str
    value: float | None
    colour: str | None
    fill: Fill | None = None
    bold: bool = False
    hidden: bool = False


@dataclass(frozen=True)
class Row:
    """One row: its name, its cells, and whether it is highlighted."""

    name: str
    cells: tuple[Cell, ...]
    bold: bool = False


@dataclass(frozen=True)
class RowGroup:
    """Rows kept together, with a heading when the table shows one."""

    label: str
    rows: tuple[Row, ...]


@dataclass(frozen=True)
class Heat:
    """Which cells are a heatmap and how their colours blend.

    Attributes:
        columns: The heatmap's columns, by position.
        blended: Whether each colour blends into its neighbours.
        blend: "columns" (across), "rows" (down) or "both".
    """

    columns: tuple[int, ...] = ()
    blended: bool = False
    blend: str = "columns"


@dataclass(frozen=True)
class TableModel:
    """A table's content.

    Attributes:
        columns: The value columns, left to right.
        groups: The column groups' headings, in order; empty for none.
        row_groups: The rows, in their groups (one unnamed group for none).
        group_headings: Whether the row groups' headings are shown.
        heat: The heatmap, if any.
    """

    columns: tuple[Column, ...]
    groups: tuple[str, ...]
    row_groups: tuple[RowGroup, ...]
    group_headings: bool = False
    heat: Heat = field(default_factory=Heat)

    @property
    def grouped_rows(self) -> bool:
        """Whether the rows are in named groups."""
        return len(self.row_groups) > 1 or bool(self.row_groups and self.row_groups[0].label)


def column_groups(columns: Sequence[str],
                  groups: Mapping[str, int | Sequence[str]] | None) -> tuple[tuple[str, ...],
                                                                                 dict[str, int]]:
    """The groups' headings and each column's group, from groups= given as
    {heading: [columns]} or {heading: how many columns, in order}.

    Raises:
        ValueError: If a group names a column the table does not have.
    """
    if not groups:
        return (), {}

    found: dict[str, int] = {}
    start = 0

    for index, (heading, members) in enumerate(groups.items()):
        if isinstance(members, int):
            chosen = list(columns[start:start + members])
            start += members
        else:
            chosen = [str(member) for member in members]
        for name in chosen:
            if name not in columns:
                raise ValueError(f"group {heading!r} names {name!r}, which is not a column")
            found[name] = index

    return tuple(groups), found


def formatter(format_: Format | None,
              negatives: str,
              whole: bool) -> Callable[[Any], str]:
    """A function writing one column's values.

    Args:
        format_: The column's format; by default as many decimals as a
            number of its kind needs, with thousands separated.
        negatives: "minus" for a true minus sign, "brackets" for (1.2).
        whole: Whether every value in the column is a whole number.
    """
    if callable(format_):
        return lambda value: MISSING if _missing(value) else str(format_(value))

    def write(value: Any) -> str:
        if _missing(value):
            return MISSING
        if not is_number(value):
            return str(value)

        size = _written(abs(float(value)), format_, whole)
        # A value that rounds to zero is written without a sign.
        if value < 0 and any(digit in "123456789" for digit in size):
            return f"({size})" if negatives == "brackets" else f"{MINUS}{size}"

        return size

    return write


def _written(size: float,
             format_: Format | None,
             whole: bool) -> str:
    if isinstance(format_, str):
        return format_.format(size)

    if isinstance(format_, Mapping):
        decimals = int(format_.get("decimals", 2))
        scale = 100.0 if format_.get("percent") else 1.0
        separator = "," if format_.get("thousands", False) else ""
        number = f"{size * scale:{separator}.{decimals}f}"
        suffix = "%" if format_.get("percent") else ""
        return f"{format_.get('prefix', '')}{number}{suffix}{format_.get('suffix', '')}"

    return f"{size:,.0f}" if whole else f"{size:,.2f}"


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def is_number(value: Any) -> bool:
    """Whether a cell's value is a number (numpy's included), not a flag."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def heat_fill(value: float,
              low: float,
              high: float,
              stops: Sequence[str],
              fade: bool,
              centred: bool) -> Fill:
    """A heatmap cell's colour, a share of the way along its map.

    Faded, near-neutral cells take the table's own background: a map lighter
    in its middle than at either end fades from the middle out to each end's
    colour, and a one-way map from its light end to its deep end, so no pale
    cell stands out on a dark table.
    """
    share = min(max((value - low) / ((high - low) or 1.0), 0.0), 1.0)

    if not fade:
        return Fill(colormaps.colour_at(stops, share))

    first, last = colormaps.colour_at(stops, 0.0), colormaps.colour_at(stops, 1.0)
    middle = colormaps.colour_at(stops, 0.5)

    if _luminance(middle) > max(_luminance(first), _luminance(last)):
        return Fill(first if share < 0.5 else last, abs(share - 0.5) * 2.0)

    light_at_top = _luminance(last) > _luminance(first)
    deep = first if light_at_top else last
    alpha = abs(share - 0.5) * 2.0 if centred else (1.0 - share if light_at_top else share)

    return Fill(deep, alpha)


def _luminance(colour: str) -> float:
    value = colour.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def over(fill: Fill,
         ground: str) -> str:
    """A fill laid over the table's background, as one colour."""
    top, base = fill.colour.lstrip("#"), ground.lstrip("#")
    mixed = [round(int(top[i:i + 2], 16) * fill.alpha + int(base[i:i + 2], 16) * (1 - fill.alpha))
             for i in (0, 2, 4)]

    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def frame_of(data: pd.DataFrame) -> pd.DataFrame:
    """A table's data, checked.

    Raises:
        TypeError: If it is not a DataFrame.
        ValueError: If it is empty.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"a table takes a pandas DataFrame, not {type(data).__name__}")
    if data.empty:
        raise ValueError("there is nothing to draw: the table is empty")

    named = data.copy()
    named.columns = [str(column) for column in named.columns]
    named.index = [str(name) for name in named.index]

    return named
