# src/beacon/analysis/analytics/instrument_table.py
"""
Every per-name figure in one table, a row a name held by any portfolio over
the window: `instrument.table()`.

    instrument.table()                                    # every column
    instrument.table(columns=["total_return", "beta", "weight"], against="benchmark")

Its columns, in order (`COLUMNS`):

- ``first_held`` and ``last_held``, the first and last day over the window
  any portfolio held it, ``in_index_at_end``, whether the index held it on
  the window's last day, and ``observations``, the returns it has over the
  window on the basis measured;
- ``total_return``, against the comparator ``excess_return``,
  ``annualised_return`` (the mean return times the periods
  a year, as the efficient frontier measures it, blank for a name with
  returns over less than a year), ``volatility``, and against the comparator
  ``beta``, ``correlation``, ``tracking_error`` and ``residual_risk``, on
  the basis ``frequency`` names (daily unless given; weekly as the frontier
  measures them);
- ``weight/<portfolio>``, the latest weight in each portfolio over the
  window, and ``average_weight/<portfolio>``, its mean daily weight;
- ``contribution``, its contribution to the fund's return
  (`attribution.contribution`), and ``risk_contribution``, to its expected
  risk at the last rebalance (`risk.risk_contributions`);
- ``days_to_sell``, at the last rebalance or month end.

``columns`` picks some, a weight column by its name alone for every
portfolio's. `columns_of` gives each column's label and unit. A column the
run cannot give is left out, and ``attrs`` say why
under ``unavailable``, beside the ``basis``, the comparator (``against``),
the ``currency`` the returns are in and the window's ``start`` and ``end``.
"""
# BN-454 (Karan, 2026-11-24): one table, so Drilldown and the risk-return
# cloud read the same numbers (beacon-ui, 2026-11-24).
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from ...catalogue import COUNT, DATE, DAYS, FRACTION, RATIO
from ..performance.name_figures import BASES, DAILY
from ..reports.statistics import Column, Unavailable
from ..risk.liquidity import PORTFOLIOS
from .shape import ALL

HELD = ("first_held", "last_held", "in_index_at_end")
OBSERVATIONS = "observations"
FIGURES = ("total_return", "excess_return", "annualised_return", "volatility", "beta",
           "correlation", "tracking_error", "residual_risk")
WEIGHTS = ("weight", "average_weight")
COLUMNS = (*HELD, OBSERVATIONS, *FIGURES, *WEIGHTS, "contribution", "risk_contribution",
           "days_to_sell")

# Each column's label and unit; a weight column's is its kind's.
LABELS = {"first_held": ("First held", DATE), "last_held": ("Last held", DATE),
          "in_index_at_end": ("In the index at the end", None),
          OBSERVATIONS: ("Returns measured", COUNT),
          "total_return": ("Total return", FRACTION),
          "excess_return": ("Excess return", FRACTION),
          "annualised_return": ("Annualised return", FRACTION),
          "volatility": ("Volatility", FRACTION), "beta": ("Beta", RATIO),
          "correlation": ("Correlation", RATIO), "tracking_error": ("Tracking error", FRACTION),
          "residual_risk": ("Residual risk", FRACTION), "weight": ("Weight", FRACTION),
          "average_weight": ("Average weight", FRACTION),
          "contribution": ("Contribution to return", FRACTION),
          "risk_contribution": ("Contribution to risk", FRACTION),
          "days_to_sell": ("Days to sell", DAYS)}
OWN = ("total_return", "annualised_return", "volatility")


def per_name(instrument: Any,
             columns: Sequence[str] | None,
             names: Sequence[str] | str | None,
             start: Any,
             end: Any,
             against: str | None,
             frequency: str) -> pd.DataFrame:
    """The table `instrument.table()` gives.

    Raises:
        ValueError: If a column is not one of `COLUMNS`, or *frequency*
            is not daily, weekly or monthly.
    """
    if frequency != DAILY and frequency not in BASES:
        raise ValueError(f"frequency is {DAILY!r}, {', '.join(map(repr, BASES))}; "
                         f"not {frequency!r}.")

    wanted = list(COLUMNS) if columns is None else _checked(columns)
    window = {"start": start, "end": end}
    context = instrument._analytics._context_for(**window)
    days = context.performance.returns.index
    held = _held(context, days)
    parts: dict[str, pd.Series] = {}
    unavailable: dict[str, str] = {}
    comparator = None

    for column in wanted:
        if column in HELD:
            parts[column] = held[column]
            continue

        work = _part(instrument, column, window, against, frequency)

        try:
            found = work()
        except (Unavailable, ValueError, KeyError) as error:
            unavailable[column] = getattr(error, "reason", None) or str(error)
            continue

        if column in FIGURES and "against" in found.attrs:
            comparator = found.attrs["against"]

        parts.update(_named(column, found))

    table = pd.DataFrame(parts).reindex(held.index if len(held) else None)
    table.index.name = "name"
    table = _picked(table, names)
    table.attrs = {"basis": frequency, **({"against": comparator} if comparator else {}),
                   "currency": context.run.currency,
                   "start": days[0] if len(days) else None,
                   "end": days[-1] if len(days) else None,
                   "unavailable": unavailable}

    return table


def columns_of(table: pd.DataFrame) -> list[Column]:
    """The label and unit of each of *table*'s columns, in its order: a
    weight column's label names its portfolio."""
    found = []

    for name in table.columns:
        kind, _, book = str(name).partition("/")
        label, unit = LABELS[kind]
        found.append(Column(str(name), f"{label} ({book})" if book else label, unit))

    return found


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
        relative = column not in OWN
        keywords = {**window, "frequency": frequency, **({"against": against} if relative else {})}

        return lambda: getattr(instrument, column)(**keywords)

    if column == OBSERVATIONS:
        return lambda: instrument._observations(window, frequency)

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


def _held(context: Any,
          days: pd.DatetimeIndex) -> pd.DataFrame:
    """The names any portfolio held over *days*, a row each, with the first
    and last day held and whether the index held it on the last day; no
    rows when the run kept no weights."""
    held: list[pd.DataFrame] = []

    for table_name in PORTFOLIOS.values():
        weights = context.run.tables.get(table_name)

        if weights is not None and not weights.empty and len(days):
            held.append(weights.loc[(weights.index >= days[0]) & (weights.index <= days[-1])]
                        .fillna(0.0) > 0.0)

    if not held:
        return pd.DataFrame(columns=list(HELD))

    any_held = pd.concat(held).groupby(level=0).max()
    any_held = any_held.loc[:, any_held.any()]
    dates = any_held.index.to_series()
    index = context.run.tables.get(PORTFOLIOS["index"])
    at_end = (index.loc[index.index <= days[-1]].iloc[-1].fillna(0.0) > 0.0
              if index is not None and not index.empty else pd.Series(dtype=bool))

    return pd.DataFrame({
        "first_held": {name: dates[column].iloc[0] for name, column in any_held.items()},
        "last_held": {name: dates[column].iloc[-1] for name, column in any_held.items()},
        "in_index_at_end": at_end.reindex(any_held.columns, fill_value=False)})


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
    books = list(PORTFOLIOS) if of == ALL else [of]
    found = {}

    for book in books:
        weights = context.run.tables.get(PORTFOLIOS[book]) if book in PORTFOLIOS else None

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


__all__ = ["COLUMNS", "LABELS", "columns_of", "per_name"]
