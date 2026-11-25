# src/beacon/analysis/reports/catalogue/style.py
"""
The style box statistics: where each name the run holds sits by size and
style, each portfolio's weight in the nine cells, and each portfolio's
centre (`beacon.analysis.performance.style`), on the fund's last day.
"""
# BN-427.
from typing import Any

import pandas as pd

from ....data.base import as_of_position
from ...panels import CHARACTERISTICS, RATES
from ...performance import style
from ...performance.portfolio import BOOKS, FUND
from ..context import Context
from ..statistics import (
    NO_DOLLAR_RATE,
    PORTFOLIO,
    RUN_PREDATES,
    TABLE,
    Column,
    StatisticSpec,
    Summarised,
    Unavailable,
)
from .build import memo

HOLDINGS = "portfolio"
USD = "USD"

STYLE_SETTINGS = ("style.large_from", "style.mid_from", "style.top", "style.bottom",
                  "style.growth")

SIZE_LABELS = {"large": "Large", "mid": "Mid", "small": "Small"}
STYLE_LABELS = {"value": "Value", "blend": "Blend", "growth": "Growth"}
BOOK_COLUMNS = (Column(FUND, "Fund", "fraction"), Column("index", "Index", "fraction"),
                Column("benchmark", "Benchmark", "fraction"))

POSITION_COLUMNS = (Column("name", "Name"), Column("size", "Size position"),
                    Column("style", "Style position"), *BOOK_COLUMNS)
BOX_COLUMNS = (Column("size", "Size", labels=SIZE_LABELS),
               Column("style", "Style", labels=STYLE_LABELS), *BOOK_COLUMNS)
CLASSIFICATION_COLUMNS = (Column("book", "Portfolio", labels={FUND: "Fund", "index": "Index",
                                                              "benchmark": "Benchmark"}),
                          Column("size", "Size", labels=SIZE_LABELS),
                          Column("style", "Style", labels=STYLE_LABELS),
                          Column("size_centre", "Size centre"),
                          Column("style_centre", "Style centre"))


def _placed(context: Context) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """The positions on the fund's last day and each portfolio's coverage,
    worked out once for the three statistics."""
    found: tuple[pd.DataFrame, dict[str, float | None]] = memo(
        context, "style_positions", lambda: _work_out(context))

    return found


def _work_out(context: Context) -> tuple[pd.DataFrame, dict[str, float | None]]:
    tables = context.run.tables

    if CHARACTERISTICS not in tables:
        raise Unavailable(RUN_PREDATES)

    snapshots = tables[CHARACTERISTICS]
    weights = {book: context.portfolio.weights(book) for book in BOOKS}
    fund = weights[FUND]

    if fund is None or fund.empty or snapshots.empty:
        raise Unavailable(RUN_PREDATES)

    last = fund.index[-1]
    dates = pd.DatetimeIndex(snapshots.index.unique()).sort_values()
    taken = dates[as_of_position(dates, last) or 0]
    snapshot = snapshots.loc[[taken]].set_index("IDENTIFIER")
    snapshot.index = snapshot.index.astype(str)
    held = {book: (None if found is None else _on(found, last)) for book, found in weights.items()}
    positions = style.style_positions(snapshot, held, context.options.style,
                                      _usd_rate(context, taken))

    return positions, style.coverage(positions, held)


def _on(weights: pd.DataFrame,
        day: pd.Timestamp) -> pd.Series:
    """A portfolio's weights on *day*, or its latest before it."""
    position = as_of_position(weights.index, day)
    row = weights.iloc[position if position is not None else 0].fillna(0.0)
    row.index = row.index.astype(str)

    return row


def _usd_rate(context: Context,
              day: pd.Timestamp) -> float:
    """A US dollar in the run's currency on *day*: 1 for a dollar run, else
    the rate the run kept, else the data store's."""
    if context.run.currency == USD:
        return 1.0

    rates = context.run.tables.get(RATES)

    if rates is not None and USD in rates.columns:
        kept = rates[USD].dropna()
        position = as_of_position(kept.index, day)

        if position is not None:
            return float(kept.iloc[position])

    if context.source is not None:
        found = context.source.fetch_fx_rates(USD, context.run.currency,
                                              end_date=str(day.date())).dropna()

        if not found.empty:
            return float(found.iloc[-1])

    raise Unavailable(NO_DOLLAR_RATE)


def _positions(context: Context) -> pd.DataFrame:
    return _placed(context)[0]


def _box(context: Context) -> Summarised:
    positions, covered = _placed(context)

    return Summarised(style.style_box(positions, tuple(BOOKS)), {"coverage": covered})


def _classification(context: Context) -> pd.DataFrame:
    return style.style_classification(_placed(context)[0], tuple(BOOKS))


def statistics() -> list[StatisticSpec]:
    """The style box statistics, in order."""
    def table(id_: str,
              label: str,
              description: str,
              read: Any,
              columns: tuple[Column, ...]) -> StatisticSpec:
        return StatisticSpec(id_, label, HOLDINGS, TABLE, PORTFOLIO, description, read,
                             unit="fraction", settings=STYLE_SETTINGS, columns=columns)

    return [
        table("style_positions", "Style positions", "Where each name held sits by size and "
              "style, from 0 to 3 on each, with each portfolio's weight in it.", _positions,
              POSITION_COLUMNS),
        table("style_box", "Style box", "Each portfolio's weight in each of the nine size "
              "and style cells.", _box, BOX_COLUMNS),
        table("style_classification", "Style classification", "The size and style each "
              "portfolio's weight is centred on, such as large blend.", _classification,
              CLASSIFICATION_COLUMNS),
    ]
