# src/beacon/analysis/analytics/shape.py
"""
A statistic's value in the shape its analytic gives it: a single figure as a
float, and each dimension added makes it a Series, then a DataFrame.

    figures {"fund": 0.12, "index": 0.10}       of="fund" -> 0.12
                                                 of="all"  -> Series by portfolio
    figures {"fund/daily": .5, "fund/monthly": .6}  of="fund" -> Series by frequency

A figure's name is its parts joined with ``/``, the first of them the
portfolio or comparator it is of when the statistic gives one each.
"""
# BN-434 (Karan, 2026-11-24).
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import pandas as pd

from ..reports.statistics import Summarised

ALL = "all"


def chosen(names: Sequence[str],
           choice: str,
           what: str) -> list[str]:
    """The figure names whose first part is *choice*, or all of them.

    Raises:
        ValueError: If none is.
    """
    if choice == ALL:
        return list(names)

    found = [name for name in names if name.split("/", 1)[0] == choice]

    if not found:
        known = list(dict.fromkeys(name.split("/", 1)[0] for name in names))
        raise ValueError(f"No {what} {choice!r} here. Known: {', '.join(known) or 'none'}.")

    return found


def of_figures(figures: Mapping[str, Any],
               parts: Sequence[str],
               choice: str | None,
               what: str) -> Any:
    """Named figures in the shape their analytic gives: *choice* picks the
    first part when the figures are given one each of *what*, None when
    they are not.

    Args:
        figures: By name, the parts joined with ``/``.
        parts: What each part of a name is, such as ``("book", "frequency")``.
        choice: The one picked, ``all``, or None.
        what: What the first part is, as an error names it.
    """
    if choice is None:
        return _shaped(dict(figures), list(parts))

    names = chosen(list(figures), choice, what)

    if choice != ALL:
        return _shaped({name.split("/", 1)[1] if "/" in name else "": figures[name]
                        for name in names}, list(parts[1:]))

    found = _shaped({name: figures[name] for name in names}, list(parts))

    # One portfolio a column, the figures down the side.
    if isinstance(found, pd.Series) and isinstance(found.index, pd.MultiIndex):
        return found.unstack(0).reindex(columns=list(dict.fromkeys(
            found.index.get_level_values(0))))

    return found


def _shaped(figures: dict[str, Any],
            parts: list[str]) -> Any:
    """A float for one unnamed figure, else a Series by name, its index a
    level a part."""
    if list(figures) == [""]:
        return number(figures[""])

    if not parts or len(parts) == 1:
        series = pd.Series(figures, dtype=object if _textual(figures) else float)
        series.index.name = parts[0] if parts else None

        return series

    split = [name.split("/") for name in figures]

    if any(len(name) != len(parts) for name in split):
        return pd.Series(figures, dtype=object if _textual(figures) else float)

    index = pd.MultiIndex.from_tuples([tuple(name) for name in split], names=parts)

    return pd.Series(list(figures.values()), index=index,
                     dtype=object if _textual(figures) else float)


def _textual(figures: Mapping[str, Any]) -> bool:
    return any(not isinstance(value, Real) and value is not None for value in figures.values())


def number(value: Any) -> Any:
    """A number as a float, numpy's included; anything else as it is."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)

    return value


def plain(value: Any) -> Any:
    """A statistic's value as it was worked out, a summarised table as its
    table with the summary in its ``attrs``."""
    if isinstance(value, Summarised):
        table = value.table.copy()
        table.attrs["summary"] = dict(value.summary)

        return table

    return value


def of_table(table: pd.DataFrame,
             choice: str | None) -> Any:
    """A table with a column each portfolio, or each portfolio's fields
    (``fund/MARKET_CAP``): *choice*'s columns, a Series when that leaves
    one."""
    if choice is None or choice == ALL:
        return table

    columns = chosen([str(column) for column in table.columns], choice, "portfolio")
    found = table[columns]

    if columns == [choice]:
        return found[choice]

    found = found.copy()
    found.columns = [column.split("/", 1)[1] for column in columns]

    return found


def of_row(table: pd.DataFrame | pd.Series,
           position: int) -> Any:
    """Row *position* of a dated table or series: a Series of a table's
    columns, a series' value."""
    row = table.iloc[position]

    return number(row) if isinstance(table, pd.Series) else row
