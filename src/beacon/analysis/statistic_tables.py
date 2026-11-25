# src/beacon/analysis/statistic_tables.py
"""
Named tables of single-figure statistics side by side, which an analysis
asks for and its block gives under ``statistic_tables``: `analytics.table()`
for the API.

    AnalysisRequest(statistic_tables={"returns": StatisticTable(
        statistics=["total_return", "sharpe_ratio"], trailing=["1Y", "3Y"], of="all")})

A table's rows are a statistic each, or a statistic and window (or period)
each, named in a ``statistic`` column and a ``window`` (or ``period``)
column, with the statistic's ``unit``; its columns are the portfolios and
comparators, or ``value`` for one portfolio. Every analysis holds two
unless it asks for other tables: a ``summary``, the main figures of every
portfolio over each trailing window, and ``risk``, their risk over the last
1, 3 and 5 years and since inception.
"""
# BN-467 (Karan, 2026-11-25), in place of trailing_statistics. Agreed with
# beacon-ui for its Summary; the risk table follows its Risk view, less the
# capture ratios, which give three figures each.
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .performance.return_figures import INCEPTION, TO_DATE, TRAILING

# The windows a trailing table can take, in order, and how they read.
WINDOWS = (*TRAILING, *TO_DATE, INCEPTION)
WINDOW_LABELS = {"1M": "1 month", "3M": "3 months", "6M": "6 months", "1Y": "1 year",
                 "3Y": "3 years", "5Y": "5 years", "10Y": "10 years", "MTD": "Month to date",
                 "QTD": "Quarter to date", "YTD": "Year to date",
                 INCEPTION: "Since inception"}

STATISTIC, WINDOW, PERIOD, UNIT, VALUE = "statistic", "window", "period", "unit", "value"


@dataclass(frozen=True)
class StatisticTable:
    """Single-figure statistics side by side, as `analytics.table()` gives
    them.

    Attributes:
        statistics: Ids or `Statistic` members, each giving one figure.
        trailing: Trailing windows, a row each: any of `WINDOWS`.
        by: Calendar periods, a row each: ``month``, ``quarter`` or
            ``year``.
        of: The portfolio, or ``all`` for a column each, with the figures
            against a comparator under the comparator's.
        against: The comparator relative figures are measured against.
        annualise: How annualised figures are given
            (`beacon.analysis.annualising`).

    Raises:
        ValueError: If both *trailing* and *by* are given.
    """
    statistics: Sequence[str]
    trailing: Sequence[str] | None = None
    by: str | None = None
    of: str | None = None
    against: str | None = None
    annualise: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "statistics", tuple(str(id_) for id_ in self.statistics))

        if self.trailing is not None:
            object.__setattr__(self, "trailing", tuple(self.trailing))

        if self.trailing is not None and self.by is not None:
            raise ValueError("A table takes trailing windows or periods, not both.")

    def check(self) -> None:
        """Refuse a statistic that is unknown.

        Raises:
            ValueError: If one is.
        """
        from .reports.catalogue import CATALOGUE  # noqa: PLC0415

        unknown = [statistic_id for statistic_id in self.statistics
                   if statistic_id not in CATALOGUE]

        if unknown:
            raise ValueError(f"Unknown statistic {', '.join(unknown)}.")

    def form(self) -> dict[str, Any]:
        """The keywords `analytics.table()` takes for it."""
        return {"trailing": None if self.trailing is None else list(self.trailing),
                "by": self.by, "of": self.of, "against": self.against,
                "annualise": self.annualise}

    def as_data(self) -> dict[str, Any]:
        """The table asked for, as plain data."""
        return {"statistics": list(self.statistics), **self.form()}


# The main figures of every portfolio over each trailing window, which an
# analysis holds with `RISK` unless it asks for other tables.
SUMMARY = StatisticTable(statistics=("total_return", "annualised_return", "volatility",
                                     "max_drawdown", "sharpe_ratio", "tracking_difference",
                                     "tracking_error"),
                         trailing=WINDOWS, of="all")
# The risk of every portfolio over the longer windows.
RISK = StatisticTable(statistics=("volatility", "max_drawdown", "downside_deviation", "skew",
                                  "excess_kurtosis", "tail_ratio", "beta",
                                  "downside_tracking_error"),
                      trailing=("1Y", "3Y", "5Y", INCEPTION), of="all")
DEFAULT = {"summary": SUMMARY, "risk": RISK}


@dataclass
class _Windows:
    """A run's context over each window a table asks for, built once."""
    context: Any
    built: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def __call__(self,
                 start: Any = None,
                 end: Any = None,
                 tracked: str | None = None,
                 rolling: Any = None,
                 annualise: bool | None = None) -> Any:
        from .analytics.forms import windowed  # noqa: PLC0415

        key = (start and str(start), end and str(end), tracked, rolling, annualise)

        if key not in self.built:
            self.built[key] = windowed(self.context, start, end, tracked, rolling, annualise)

        return self.built[key]


def worked_out(table: StatisticTable,
               context: Any) -> pd.DataFrame:
    """*table* of the run *context* is of, a row a statistic (and window or
    period) and a column a portfolio."""
    # The analytics read the catalogue, which imports this package.
    from .analytics.forms import value  # noqa: PLC0415
    from .analytics.tables import table as side_by_side  # noqa: PLC0415
    from .reports.catalogue import CATALOGUE  # noqa: PLC0415

    windows = _Windows(context)
    found = side_by_side(table.statistics, table.form(),
                         lambda statistic_id, form: value(CATALOGUE[statistic_id], form,
                                                          windows))

    return _long(found, WINDOW if table.trailing is not None
                 else PERIOD if table.by is not None else None)


def _long(found: pd.DataFrame,
          time: str | None) -> pd.DataFrame:
    """*found* as rows of a statistic (and *time*) each, with its unit."""
    from .reports.catalogue import CATALOGUE  # noqa: PLC0415

    if isinstance(found.index, pd.MultiIndex):
        long = found.reset_index()
    elif time is not None:
        long = (found.rename_axis(columns=time).stack(future_stack=True)
                .rename(VALUE).reset_index())
    else:
        long = found.reset_index()

    long.columns = [STATISTIC, *long.columns[1:]]

    if time is not None:
        long = long.rename(columns={long.columns[1]: time})

    at = 2 if time is not None else 1
    long.insert(at, UNIT, [CATALOGUE[statistic_id].unit for statistic_id in long[STATISTIC]])

    return long


def tables_of(asked: Mapping[str, Any] | None) -> dict[str, StatisticTable]:
    """The tables asked for, each a `StatisticTable` or its plain data, by
    name; `DEFAULT` when None."""
    if asked is None:
        return dict(DEFAULT)

    return {name: table if isinstance(table, StatisticTable) else StatisticTable(**table)
            for name, table in asked.items()}


__all__ = ["DEFAULT", "RISK", "SUMMARY", "WINDOWS", "WINDOW_LABELS", "StatisticTable", "tables_of",
           "worked_out"]
