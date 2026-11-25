# src/beacon/analysis/analytics/instrument_table.py
"""
Every per-name figure in one table, a row a name: `instrument.table()`.

    instrument.table()                                    # every column
    instrument.table(columns=["total_return", "beta", "weight"], against="benchmark")

Its columns, in order (`COLUMNS`):

- ``total_return``, ``volatility``, and against the comparator ``beta``,
  ``correlation``, ``tracking_error`` and ``residual_risk``, on the basis
  ``frequency`` names (daily unless given);
- ``weight/<portfolio>``, the latest weight in each portfolio over the
  window, and ``average_weight/<portfolio>``, its mean daily weight;
- ``contribution``, its contribution to the fund's return
  (`attribution.contribution`), and ``risk_contribution``, to its expected
  risk at the last rebalance (`risk.risk_contributions`);
- ``days_to_sell``, at the last rebalance or month end.

``columns`` picks some, a weight column by its name alone for every
portfolio's. A column the run cannot give is left out, and ``attrs`` say why
under ``unavailable``, beside the ``basis`` and the comparator (``against``).
"""
# BN-454 (Karan, 2026-11-24): one table, so Drilldown and the risk-return
# cloud read the same numbers (beacon-ui, 2026-11-24).
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from ..reports.statistics import Unavailable
from ..risk.liquidity import BOOKS
from .shape import ALL

FIGURES = ("total_return", "volatility", "beta", "correlation", "tracking_error",
           "residual_risk")
WEIGHTS = ("weight", "average_weight")
COLUMNS = (*FIGURES, *WEIGHTS, "contribution", "risk_contribution", "days_to_sell")


def per_name(instrument: Any,
             columns: Sequence[str] | None,
             names: Sequence[str] | str | None,
             start: Any,
             end: Any,
             against: str | None,
             frequency: str) -> pd.DataFrame:
    """The table `instrument.table()` gives.

    Raises:
        ValueError: If a column is not one of `COLUMNS`.
    """
    wanted = list(COLUMNS) if columns is None else _checked(columns)
    window = {"start": start, "end": end}
    parts: dict[str, pd.Series] = {}
    unavailable: dict[str, str] = {}
    comparator = None

    for column in wanted:
        work = _part(instrument, column, window, against, frequency)

        try:
            found = work()
        except (Unavailable, ValueError, KeyError) as error:
            unavailable[column] = getattr(error, "reason", None) or str(error)
            continue

        if column in FIGURES and "against" in found.attrs:
            comparator = found.attrs["against"]

        parts.update(_named(column, found))

    table = pd.DataFrame(parts)
    table.index.name = "name"
    table = _picked(table, names)
    table.attrs = {"basis": frequency, **({"against": comparator} if comparator else {}),
                   "unavailable": unavailable}

    return table


def _checked(columns: Sequence[str]) -> list[str]:
    unknown = [column for column in columns
               if column not in COLUMNS and column.split("/")[0] not in WEIGHTS]

    if unknown:
        raise ValueError(f"Unknown column {', '.join(unknown)}. Known: {', '.join(COLUMNS)}.")

    return list(columns)


def _part(instrument: Any,
          column: str,
          window: dict[str, Any],
          against: str | None,
          frequency: str) -> Callable[[], Any]:
    """What works column *column* out."""
    if column in FIGURES:
        relative = column not in ("total_return", "volatility")
        keywords = {**window, "frequency": frequency, **({"against": against} if relative else {})}

        return lambda: getattr(instrument, column)(**keywords)

    kind, _, book = column.partition("/")

    if kind == "weight":
        return lambda: _latest_weights(instrument, book or ALL, window)

    if kind == "average_weight":
        return lambda: instrument.average_weight(of=book or ALL, **window)

    if column == "contribution":
        return lambda: _by_name(instrument._analytics._value("contribution", dict(window)),
                                "contribution")

    if column == "risk_contribution":
        return lambda: instrument._analytics._value("risk_contributions", {})["contribution"]

    return lambda: instrument.days_to_sell()["days"]


def _named(column: str,
           found: pd.Series | pd.DataFrame) -> dict[str, pd.Series]:
    """*found* as named columns: a table of portfolios a column each, named
    ``<column>/<portfolio>``."""
    if isinstance(found, pd.DataFrame):
        kind = column.partition("/")[0]

        return {f"{kind}/{book}": found[book] for book in found.columns}

    if column in WEIGHTS:
        return {f"{column}/fund": found}

    return {column: found}


def _latest_weights(instrument: Any,
                    of: str,
                    window: dict[str, Any]) -> pd.DataFrame | pd.Series:
    """Each name's weight in each portfolio on the window's last day."""
    context = instrument._analytics._context_for(**window)
    last = context.performance.returns.index[-1]
    books = list(BOOKS) if of == ALL else [of]
    found = {}

    for book in books:
        weights = context.run.tables.get(BOOKS[book]) if book in BOOKS else None

        if weights is None or weights.empty:
            if of == ALL:
                continue
            raise ValueError(f"The run kept no weights for {book!r}.")

        found[book] = weights.loc[weights.index <= last].iloc[-1].fillna(0.0)

    return pd.DataFrame(found).fillna(0.0)


def _by_name(table: pd.DataFrame,
             column: str) -> pd.Series:
    """*column* of the rows of *table* a name each."""
    names = table[table["level"] == "name"] if "level" in table.columns else table

    return names.set_index("asset_id")[column] if "asset_id" in names.columns else names[column]


def _picked(table: pd.DataFrame,
            names: Sequence[str] | str | None) -> pd.DataFrame:
    """*table*'s rows of *names*, every name when None."""
    if names is None:
        return table

    wanted = [names] if isinstance(names, str) else list(names)
    unknown = [name for name in wanted if name not in table.index]

    if unknown:
        raise ValueError(f"The run measured no {', '.join(unknown[:5])}"
                         f"{' and others' if len(unknown) > 5 else ''}.")

    return table.loc[wanted]


__all__ = ["COLUMNS", "per_name"]
