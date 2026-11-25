# src/beacon/analysis/reports/catalogue/risk_return.py
"""
The risk and return statistics: every name the index held, the fund's and
its comparators' measured the same way, and the efficient frontier through
them in hindsight (`beacon.analysis.performance.members`).
"""
# BN-429. Agreed with Karan on 2026-11-21 for beacon-ui's Risk & correlation view.
import pandas as pd

from ...panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS
from ...performance import members
from ..context import Context
from ..options import BENCHMARK, INDEX
from ..statistics import (
    NO_INDEX,
    RISK,
    RUN_PREDATES,
    TABLE,
    TOO_SHORT,
    Column,
    StatisticSpec,
    Summarised,
    Unavailable,
)
from .build import memo

CATEGORY = "risk"

SETTINGS = ("risk_return.frequency",)
FRONTIER_SETTINGS = (*SETTINGS, "risk_return.frontier_names", "risk_return.frontier_points")

BOOK_LABELS = {"fund": "Fund", INDEX: "Index", BENCHMARK: "Benchmark"}
FIGURE_COLUMNS = (Column("return", "Return", "fraction"),
                  Column("volatility", "Volatility", "fraction"),
                  Column("observations", "Periods", "count"),
                  Column("first", "First", "date"), Column("last", "Last", "date"))
MEMBER_COLUMNS = (Column("name", "Name"), *FIGURE_COLUMNS,
                  Column("in_index_at_end", "In the index at the end"),
                  Column("fund_weight", "Fund weight", "fraction"),
                  Column("index_weight", "Index weight", "fraction"))
PORTFOLIO_COLUMNS = (Column("book", "Portfolio", labels=BOOK_LABELS), *FIGURE_COLUMNS)
FRONTIER_COLUMNS = (Column("volatility", "Volatility", "fraction"),
                    Column("return", "Return", "fraction"),
                    Column("minimum_variance", "Minimum variance"))

NO_FRONTIER = ("The frontier needs two or more of the largest index members with a return "
               "in every period, a year of periods, and more periods than names.")


def _window(context: Context,
            table: pd.DataFrame) -> pd.DataFrame:
    """*table*'s rows on the days the fund's returns are measured."""
    returns = context.performance.returns

    if returns.empty:
        raise Unavailable(TOO_SHORT)

    return table.loc[(table.index >= returns.index[0]) & (table.index <= returns.index[-1])]


def _tables(context: Context) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The names' returns and the index's weights over the window."""
    tables = context.run.tables

    if INDEX not in context.run.comparators:
        raise Unavailable(NO_INDEX)

    if NAME_RETURNS not in tables or INDEX_WEIGHTS not in tables:
        raise Unavailable(RUN_PREDATES)

    return _window(context, tables[NAME_RETURNS]), _window(context, tables[INDEX_WEIGHTS])


def _members(context: Context) -> Summarised:
    def work_out() -> Summarised:
        name_returns, index_weights = _tables(context)
        fund = context.run.tables.get(FUND_WEIGHTS)
        found = members.member_risk_return(name_returns, index_weights,
                                           None if fund is None else _window(context, fund),
                                           context.options.risk_return)

        return Summarised(found.table, {"measured": found.measured,
                                        "too_short": found.too_short})

    found: Summarised = memo(context, "member_risk_return", work_out)

    return found


def _portfolios(context: Context) -> pd.DataFrame:
    books = {"fund": context.performance.returns}
    books.update({name: context.book(name).returns for name in (INDEX, BENCHMARK)
                  if name in context.run.comparators})

    return members.portfolio_risk_return(books, context.options.risk_return)


def _frontier(context: Context) -> Summarised:
    name_returns, index_weights = _tables(context)
    found = members.hindsight_frontier(name_returns, index_weights, context.options.risk_return)

    if found is None:
        raise Unavailable(TOO_SHORT, NO_FRONTIER)

    return Summarised(found.table, {"names": found.names, "periods": found.periods,
                                    "first": found.first, "last": found.last})


def statistics() -> list[StatisticSpec]:
    """The risk and return statistics, in order."""
    return [
        StatisticSpec("member_risk_return", "Risk and return of every member", CATEGORY, TABLE,
                      RISK,
                      "The annualised return and volatility of every name the index held over "
                      "the run, on weekly returns; names with under a year are counted, not "
                      "shown.", _members, unit="fraction", settings=SETTINGS,
                      columns=MEMBER_COLUMNS),
        StatisticSpec("portfolio_risk_return", "Risk and return of each portfolio", CATEGORY,
                      TABLE, RISK, "The fund's, the index's and the benchmark's annualised return "
                      "and volatility, measured as the members' are.", _portfolios,
                      unit="fraction", settings=SETTINGS, columns=PORTFOLIO_COLUMNS),
        StatisticSpec("efficient_frontier", "Efficient frontier in hindsight", CATEGORY, TABLE,
                      RISK, "The long-only, fully invested mean-variance frontier of the largest "
                      "latest index members, drawn with the returns the run went on to see.",
                      _frontier, unit="fraction", settings=FRONTIER_SETTINGS,
                      columns=FRONTIER_COLUMNS),
    ]
