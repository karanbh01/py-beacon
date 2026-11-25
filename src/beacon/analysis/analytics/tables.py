# src/beacon/analysis/analytics/tables.py
"""
Several single-figure statistics side by side: a row a statistic, a column a
window, period or portfolio.

    analytics.table([Statistic.TOTAL_RETURN, Statistic.SHARPE_RATIO],
                    trailing=["1Y", "3Y", "5Y"])
    analytics.table(["total_return", "volatility"], by="year")
    analytics.table(["total_return", "tracking_error"], of="all")

Each keyword goes to the statistics that take it, and ``of="all"`` is also
``against="all"`` for a figure of the fund against each comparator, so a
portfolio's figures and the fund's against each comparator share a table.
Given a window or period and every portfolio, a row is a statistic and a
window, and a column a portfolio or comparator.
"""
# BN-445 (Karan, 2026-11-24).
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd

from ..reports.catalogue import CATALOGUE
from ..reports.catalogue.forms import (
    AGAINST,
    ANNUALISE,
    AS_OF,
    BOOK,
    COMPARATOR,
    OF,
    ROLLING,
    STEP,
)
from ..reports.statistics import FIGURES, StatisticSpec
from .shape import ALL

# Keywords a table does not take: a rolling or dated figure is a series, not
# a figure, so it goes on its own.
NOT_IN_A_TABLE = (ROLLING, STEP, AS_OF)


def single_figure(statistic: StatisticSpec) -> bool:
    """Whether *statistic* gives one figure, a portfolio's or the fund's
    against a comparator."""
    return statistic.kind == FIGURES and statistic.figure_names in ((BOOK,), (COMPARATOR,))


def table(statistics: Iterable[str],
          form: Mapping[str, Any],
          value: Callable[[str, dict[str, Any]], Any]) -> pd.DataFrame:
    """*statistics* side by side, each in the form *form*'s keywords ask for.

    Args:
        statistics: Ids or `Statistic` members, each giving one figure.
        form: The keywords, those left as None not given.
        value: Works out a statistic by id in a form.

    Raises:
        ValueError: If a statistic gives more than one figure, or takes a
            window or period asked for that it does not.
        TypeError: If a keyword is one no table takes, or none of the
            statistics takes.
    """
    ids = [str(statistic_id) for statistic_id in statistics]
    form = {keyword: given for keyword, given in form.items() if given is not None}
    _check(ids, form)
    rows = {statistic_id: value(statistic_id, _own(CATALOGUE[statistic_id], form))
            for statistic_id in ids}

    return _joined(rows)


def _check(ids: list[str],
           form: Mapping[str, Any]) -> None:
    unknown = [statistic_id for statistic_id in ids if statistic_id not in CATALOGUE]

    if unknown:
        raise ValueError(f"Unknown statistic {', '.join(unknown)}.")

    several = [statistic_id for statistic_id in ids if not single_figure(CATALOGUE[statistic_id])]

    if several:
        verb = "gives" if len(several) == 1 else "give"
        raise ValueError(f"A table puts single figures side by side; {', '.join(several)} "
                         f"{verb} more than one. Ask for each on its own.")

    refused = [keyword for keyword in form if keyword in NOT_IN_A_TABLE]

    if refused:
        raise TypeError(f"A table does not take {', '.join(refused)}: ask for each statistic "
                        f"on its own.")

    unused = [keyword for keyword in form if not any(
        _takes(CATALOGUE[statistic_id], keyword, form[keyword]) for statistic_id in ids)]

    if unused:
        raise TypeError(f"None of these statistics takes {', '.join(unused)}.")


def _takes(statistic: StatisticSpec,
           keyword: str,
           given: Any) -> bool:
    """Whether *statistic* is given *keyword*: ``of="all"`` too when it is
    given as its ``against``."""
    own = _own(statistic, {keyword: given})

    return keyword in own or (keyword == OF and AGAINST in own)


def _own(statistic: StatisticSpec,
         form: Mapping[str, Any]) -> dict[str, Any]:
    """The keywords of *form* *statistic* is given: those it takes, with a
    portfolio's ``of`` its ``against`` when it measures the fund against a
    comparator, and ``annualise`` only when it is annualised; a window or
    period whether it takes it or not, so one it does not is refused."""
    own: dict[str, Any] = {}

    for keyword, given in form.items():
        if keyword == OF and OF not in statistic.forms:
            if AGAINST in statistic.forms and given == ALL and AGAINST not in form:
                own[AGAINST] = ALL
            continue

        if keyword in (AGAINST, ANNUALISE) and keyword not in statistic.forms:
            continue

        own[keyword] = given

    return own


def _joined(rows: Mapping[str, Any]) -> pd.DataFrame:
    """Each statistic's value a row: a figure in one column, a Series its
    index as columns, a DataFrame its rows under the statistic."""
    if all(not isinstance(row, (pd.Series, pd.DataFrame)) for row in rows.values()):
        return pd.DataFrame({"value": pd.Series(rows, dtype=float)}).rename_axis("statistic")

    if any(isinstance(row, pd.DataFrame) for row in rows.values()):
        found = pd.concat({statistic_id: row if isinstance(row, pd.DataFrame)
                           else pd.DataFrame({"value": [row]})
                           for statistic_id, row in rows.items()}, names=["statistic"])
        found.columns.name = None

        return found

    series = {statistic_id: row if isinstance(row, pd.Series) else pd.Series({"value": row})
              for statistic_id, row in rows.items()}
    # The columns in the order the statistics give them, not sorted.
    columns = list(dict.fromkeys(key for row in series.values() for key in row.index))
    found = pd.DataFrame(series).T.reindex(columns=columns)
    found.index.name = "statistic"
    found.columns.name = None

    return found
