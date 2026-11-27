# src/beacon/analysis/analytics/display.py
"""
A report's formatted display: in Jupyter, printed, and as a page to save or
send.

    report = result.analytics.reports.factsheet()
    report                     # in Jupyter: the formatted page
    print(report)              # the same as plain text
    html = report.to_html()    # the page as a string

The page opens with the report, its level, the window, the currency and the
portfolios. Then a section for each category, in the report's order: its
single figures as a table, a row a statistic and a column a portfolio; each
table statistic to its first rows, with a count of the rest; each statistic
through time and each rolling view as its latest value. Each figure is
formatted from its unit. A footer lists what the run could not give and why.
"""
# BN-458 (Karan, 2026-11-24). Charts are left out until the default charts
# for a backtest are agreed.
import html
import numbers
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from ..placement import CATEGORIES, INSTRUMENT, PART_OF
from ..reports.catalogue import CATALOGUE, FIELD_UNITS
from ..reports.statistics import DECIMALS
from .tables import single_figure

# How many rows of a table are shown; the rest are counted.
ROWS = 5

BLANK = "\N{EN DASH}"

PART_LABELS = {category: category.capitalize() for category in CATEGORIES} | {
    "relative": "Relative", INSTRUMENT: "By name"}
PORTFOLIO_LABELS = {"fund": "Fund", "index": "Index", "optimised": "Optimised index",
               "benchmark": "Benchmark", "active": "Active"}


@dataclass(frozen=True)
class Block:
    """A grid of formatted cells under a title, and a note beneath it."""
    title: str | None
    header: list[str]
    rows: list[list[str]]
    note: str | None = None


@dataclass(frozen=True)
class Section:
    """One category's blocks."""
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(frozen=True)
class Page:
    """What a report's display shows, before it is drawn as HTML or text."""
    title: str
    facts: list[tuple[str, str]]
    sections: list[Section]
    unavailable: list[tuple[str, str]]


def formatted(value: Any,
              unit: str | None) -> str:
    """*value* as shown, from its *unit*: a fraction as a percentage, a ratio
    to two decimals, basis points with ``bps``, a count whole."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return BLANK

    if isinstance(value, pd.Timestamp | date):
        return str(pd.Timestamp(value).strftime("%Y-%m-%d"))

    if isinstance(value, bool | np.bool_):
        return "Yes" if value else "No"

    if not isinstance(value, numbers.Number):
        return str(value)

    number = float(value)  # type: ignore[arg-type]

    if unit is None:
        return _plain(number)

    decimals = DECIMALS.get(unit, 4)

    if unit == "fraction":
        return f"{number * 100:,.{decimals - 2}f}%"

    if unit == "bps":
        return f"{number:,.{decimals}f} bps"

    return f"{number:,.{decimals}f}"


def _plain(number: float) -> str:
    """A number with no unit: whole when it is, an amount without decimals,
    else to four."""
    if number.is_integer() or abs(number) >= 1000:
        return f"{number:,.0f}"

    return f"{number:,.4f}"


def page_of(report: Any) -> Page:
    """What *report*'s display shows."""
    context = report._analytics._context
    returns = context.performance.returns
    window = (f"{formatted(returns.index[0], None)} to {formatted(returns.index[-1], None)}"
              if len(returns) else BLANK)
    facts = [("Level", report.level.capitalize()), ("Window", window),
             ("Currency", str(context.run.currency or BLANK)),
             ("Portfolios", ", ".join(_book_label(book) for book in context.portfolios()))]
    sections = [Section(PART_LABELS.get(part, part), _blocks(report, ids))
                for part, ids in _by_part(report).items()]
    unavailable = [(CATALOGUE[statistic_id].label, reason)
                   for statistic_id, reason in report.unavailable.items()]

    return Page(f"{report.label} report", facts, sections, unavailable)


def _by_part(report: Any) -> dict[str, list[str]]:
    """The report's statistics it holds by category, in the report's
    order."""
    parts: dict[str, list[str]] = {}

    for statistic_id in report.statistics:
        if statistic_id in report:
            parts.setdefault(PART_OF.get(statistic_id, INSTRUMENT), []).append(statistic_id)

    return parts


def _blocks(report: Any,
            ids: list[str]) -> list[Block]:
    """One category's blocks: its single figures together, then each other
    statistic, then the latest of each rolling view."""
    singles = [statistic_id for statistic_id in ids if single_figure(CATALOGUE[statistic_id])]
    blocks = [_figures(report, singles)] if singles else []

    blocks.extend(_block(report[statistic_id]) for statistic_id in ids
                  if statistic_id not in singles)

    rolling = [_latest_row(report[statistic_id]) for statistic_id in ids
               if report[statistic_id].rolling is not None]

    if rolling:
        days = report._analytics._context.options.rolling.window
        blocks.append(Block(f"Rolling {days} days, latest", ["", "Value", "As of"], rolling))

    return blocks


def _figures(report: Any,
             ids: list[str]) -> Block:
    """Single figures side by side, a column a portfolio with any."""
    table = report._side_by_side(ids).dropna(axis=1, how="all")
    rows = [[CATALOGUE[statistic_id].label,
             *(formatted(value, CATALOGUE[statistic_id].unit) for value in table.loc[statistic_id])]
            for statistic_id in ids]

    return Block(None, ["", *(_book_label(str(column)) for column in table.columns)], rows)



def _block(entry: Any) -> Block:
    """A statistic of several figures, a table, or a figure through time."""
    value = entry.headline
    spec = CATALOGUE[entry.id]

    if isinstance(value, pd.Series | pd.DataFrame) and _dated(value):
        return _latest(entry, value)

    if isinstance(value, pd.Series):
        value = value.to_frame("value")

    if not isinstance(value, pd.DataFrame):
        return Block(entry.label, [], [[formatted(value, spec.unit)]])

    return _table(entry.label, value, spec)


def _table(title: str,
           table: pd.DataFrame,
           spec: Any) -> Block:
    """*table*'s first rows, each figure formatted from its unit."""
    if table.empty:
        return Block(title, [], [], "No rows.")

    shown = table.head(ROWS)
    numbered = isinstance(table.index, pd.RangeIndex)
    rows = [[str(number + 1) if numbered else _name_label(label),
             *(formatted(value, _unit_of(spec, label, column))
               for value, column in zip(row, table.columns, strict=True))]
            for number, (label, row) in enumerate(zip(shown.index,
                                                      shown.itertuples(index=False),
                                                      strict=True))]
    rest = len(table) - len(shown)
    header = ["", *(_column_label(spec, column) for column in table.columns)]

    note = f"And {rest} more {'row' if rest == 1 else 'rows'}." if rest else None

    return Block(title, header, rows, note)


def _latest(entry: Any,
            value: pd.Series | pd.DataFrame) -> Block:
    """A statistic through time, as its latest value."""
    spec = CATALOGUE[entry.id]
    found = value.dropna(how="all")

    if found.empty:
        return Block(entry.label, [], [[BLANK]])

    if isinstance(found, pd.Series):
        found = found.to_frame("value")

    return _table(f"{entry.label}, latest", found.iloc[[-1]], spec)


def _latest_row(entry: Any) -> list[str]:
    """A rolling view's latest value and its day: of its first figure, when
    it has several."""
    found = entry.rolling

    if isinstance(found, pd.DataFrame):
        found = found.iloc[:, 0]

    found = found.dropna()

    if found.empty:
        return [entry.label, BLANK, BLANK]

    return [entry.label, formatted(found.iloc[-1], entry.unit), formatted(found.index[-1], None)]


def _unit_of(spec: Any,
             row: Any,
             column: Any) -> str | None:
    """The unit of a table's figure: its own, by the name of its row or
    column, its column's, or the statistic's."""
    own = FIELD_UNITS.get(spec.id, {})
    names = [*_parts(row), *_parts(column)]

    for name in reversed(names):
        if name in own:
            return own[name]

    listed = _listed(spec, column)
    unit: str | None = listed.unit if listed is not None else spec.unit

    return unit


def _column_label(spec: Any,
                  column: Any) -> str:
    listed = _listed(spec, column)

    return listed.label if listed is not None else _name_label(column)


def _listed(spec: Any,
            column: Any) -> Any:
    """The column of *spec* that *column* is: by its name, or the fund's of
    it, when the fund's alone are given."""
    columns = {listed.name: listed for listed in spec.columns}

    return columns.get(str(column)) or columns.get(f"fund/{column}")


def _parts(name: Any) -> list[str]:
    """The names *name* is made of: a tuple's, or a ``portfolio/field``'s."""
    if isinstance(name, tuple):
        return [part for item in name for part in _parts(item)]

    return [part.split(" ")[-1] for part in str(name).split("/")]


def _name_label(name: Any) -> str:
    """*name* as read: its parts joined, a portfolio by its label, a figure's
    underscores as spaces, the first letter a capital."""
    parts = (name if isinstance(name, tuple) else
             str(name).split("/") if isinstance(name, str) else (name,))
    text = ", ".join(_book_label(formatted(part, None)).replace("_", " ") for part in parts)

    return text[:1].upper() + text[1:]


def _dated(value: pd.Series | pd.DataFrame) -> bool:
    return isinstance(value.index, pd.DatetimeIndex) and len(value) > 1


def _book_label(name: str) -> str:
    if name.startswith("class/"):
        return f"Class {name.removeprefix('class/')}"

    return PORTFOLIO_LABELS.get(name, name)


# -- drawn --------------------------------------------------------------------

STYLE = """
.beacon-report { font: 13px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  color: inherit; max-width: 60rem; }
.beacon-report h2 { font-size: 1.25rem; margin: 0 0 .25rem; padding-left: .6rem;
  border-left: 3px solid #2f6fdf; }
.beacon-report .facts { display: flex; flex-wrap: wrap; gap: .25rem 1.5rem; margin: 0 0 1rem;
  opacity: .8; }
.beacon-report .facts b { font-weight: 600; margin-right: .3rem; }
.beacon-report details { border-top: 1px solid rgba(127,127,127,.3); padding: .4rem 0; }
.beacon-report summary { cursor: pointer; font-weight: 600; font-size: 1rem; }
.beacon-report h4 { font-size: .85rem; font-weight: 600; margin: .8rem 0 .25rem; }
.beacon-report .scroll { overflow-x: auto; }
.beacon-report table { border-collapse: collapse; font-variant-numeric: tabular-nums;
  margin: .25rem 0; }
.beacon-report th, .beacon-report td { padding: .15rem .8rem .15rem 0; text-align: right;
  white-space: nowrap; }
.beacon-report th:first-child, .beacon-report td:first-child { text-align: left; }
.beacon-report thead th { font-weight: 600; border-bottom: 1px solid rgba(127,127,127,.4); }
.beacon-report .note, .beacon-report footer { opacity: .7; font-size: .85rem; }
.beacon-report footer { border-top: 1px solid rgba(127,127,127,.3); padding-top: .5rem; }
.beacon-report footer ul { margin: .25rem 0; padding-left: 1.2rem; }
"""


def to_html(page: Page) -> str:
    """*page* as an HTML fragment, with its styles."""
    text = html.escape
    facts = "".join(f"<span><b>{text(name)}</b>{text(value)}</span>" for name, value in page.facts)
    sections = "".join(f"<details open><summary>{text(section.title)}</summary>"
                       f"{''.join(_html_block(block) for block in section.blocks)}</details>"
                       for section in page.sections)
    footer = ""

    if page.unavailable:
        items = "".join(f"<li><b>{text(label)}</b>: {text(reason)}</li>"
                        for label, reason in page.unavailable)
        footer = f"<footer>Not available:<ul>{items}</ul></footer>"

    return (f'<div class="beacon-report"><style>{STYLE}</style><h2>{text(page.title)}</h2>'
            f'<div class="facts">{facts}</div>{sections}{footer}</div>')


def to_document(page: Page) -> str:
    """*page* as a whole HTML page, to save or send."""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(page.title)}</title></head>"
            f'<body style="margin: 2rem;">{to_html(page)}</body></html>')


def _html_block(block: Block) -> str:
    text = html.escape
    title = f"<h4>{text(block.title)}</h4>" if block.title else ""
    head = ("<thead><tr>" + "".join(f"<th>{text(cell)}</th>" for cell in block.header)
            + "</tr></thead>") if block.header else ""
    body = "".join("<tr>" + "".join(f"<td>{text(cell)}</td>" for cell in row) + "</tr>"
                   for row in block.rows)
    note = f'<div class="note">{text(block.note)}</div>' if block.note else ""

    if not block.header and not block.rows:
        return f"{title}{note}"

    return f'{title}<div class="scroll"><table>{head}<tbody>{body}</tbody></table></div>{note}'


def to_text(page: Page) -> str:
    """*page* as plain text."""
    lines = [page.title, "=" * len(page.title),
             "  ".join(f"{name}: {value}" for name, value in page.facts)]

    for section in page.sections:
        lines += ["", section.title, "-" * len(section.title)]

        for block in section.blocks:
            lines += _text_block(block)

    if page.unavailable:
        lines += ["", "Not available:", *(f"  {label}: {reason}"
                                          for label, reason in page.unavailable)]

    return "\n".join(lines)


def _text_block(block: Block) -> list[str]:
    grid = [block.header, *block.rows] if block.header else block.rows
    lines = [f"{block.title}:"] if block.title else []
    note = [f"  {block.note}"] if block.note else []

    if not grid:
        return [*lines, *note, ""]

    widths = [max(len(row[index]) for row in grid if index < len(row))
              for index in range(max(len(row) for row in grid))]

    for row in grid:
        cells = [cell.ljust(widths[0]) if index == 0 else cell.rjust(widths[index])
                 for index, cell in enumerate(row)]
        lines.append("  " + "  ".join(cells).rstrip())

    return [*lines, *note, ""]


__all__ = ["Block", "Page", "Section", "formatted", "page_of", "to_document", "to_html",
           "to_text"]
