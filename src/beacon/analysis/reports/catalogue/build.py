# src/beacon/analysis/reports/catalogue/build.py
"""
The shapes most statistics take: a figure of each portfolio's returns, a figure of
the fund against each comparator, and tables from one call.
"""
# BN-398.
import logging
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd

from ...performance.analytics import PerformanceAnalytics
from ...performance.relative_analytics import RelativeAnalytics
from ..context import FUND, Context
from ..statistics import NO_INDEX, Unavailable

logger = logging.getLogger(__name__)

ACTIVE = "active"


Reader = Callable[[PerformanceAnalytics, Context], Any]
RelativeReader = Callable[[RelativeAnalytics, Context], Any]


def per_portfolio(read: Reader,
             active: bool = False) -> Callable[[Context], dict[str, Any]]:
    """A statistic of each portfolio's returns: the fund's, each comparator's and
    each share class's; with *active*, the fund's less the tracked
    comparator's too."""
    def compute(context: Context) -> dict[str, Any]:
        found: dict[str, Any] = {FUND: read(context.performance, context)}

        for name in context.portfolios()[1:]:
            found[name] = _quietly(name, partial(_read_portfolio, read, context, name))

        tracked = context.tracked

        if active and tracked is not None and _numbers(found[FUND], found.get(tracked)):
            found[ACTIVE] = found[FUND] - found[tracked]

        return found

    return compute


def per_comparator(read: RelativeReader) -> Callable[[Context], dict[str, Any]]:
    """A statistic of the fund against each comparator the run has."""
    def compute(context: Context) -> dict[str, Any]:
        if not context.comparators:
            raise Unavailable(NO_INDEX)

        return {name: read(context.relative(name), context) for name in context.comparators}

    return compute


def memo(analytics: Any,
         name: str,
         build: Callable[[], Any]) -> Any:
    """*build*'s answer, worked out once for *analytics* (a portfolio's or a
    comparator's), so statistics read from the same call share it."""
    kept: dict[str, Any] = analytics.__dict__.setdefault("_statistics", {})

    if name not in kept:
        kept[name] = build()

    return kept[name]


def fund_only(read: Reader) -> Callable[[Context], dict[str, Any]]:
    """A statistic of the fund alone."""
    return lambda context: {FUND: read(context.performance, context)}


def frame(table: pd.DataFrame) -> pd.DataFrame:
    """*table* with two-level column and row names joined with ``/``, ready
    to send."""
    table = table.copy()

    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [_joined(column) for column in table.columns]

    if isinstance(table.index, pd.MultiIndex):
        table.index = pd.Index([_joined(row) for row in table.index])

    return table


def _joined(parts: Any) -> str:
    return "/".join(str(part) for part in parts if str(part))


def _read_portfolio(read: Reader,
                    context: Context,
                    name: str) -> Any:
    return read(context.performance_of(name), context)


def _quietly(name: str,
             read: Callable[[], Any]) -> Any:
    """A comparator's or class's figure, None when its returns cannot give
    it; the fund's figure is not held back by another portfolio's."""
    try:
        return read()
    except Unavailable:
        return None
    except (ValueError, ZeroDivisionError, KeyError) as error:
        logger.debug(f"{name}: no figure ({error}).")
        return None


def _numbers(*values: Any) -> bool:
    return all(isinstance(value, (int, float)) and value == value for value in values)
