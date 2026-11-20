# src/beacon/analysis/reports/catalogue/rating.py
"""
The risk rating statistics: the fund's rating on the factsheet's 1 to 7
scale with its tracking error band, and the days it spent at each
(`beacon.analysis.performance.risk_rating`).
"""
# Agreed with Karan on 2026-11-20 for beacon-ui's Summary risk box (BU-294).
import pandas as pd

from ...performance import risk_rating as rating
from ..context import Context
from ..options import BENCHMARK, INDEX
from ..statistics import PERFORMANCE, TABLE, TOO_SHORT, Column, Statistic, Unavailable

RISK = "risk"

SETTINGS = ("risk_rating.volatility_bounds", "risk_rating.tracking_bounds",
            "risk_rating.years", "window.tracked")

BAND_LABELS = {"passive": "Passive", "enhanced": "Enhanced", "active": "Active",
               "unconstrained": "Unconstrained"}
BOOK_LABELS = {rating.FUND: "Fund", INDEX: "Index", BENCHMARK: "Benchmark"}

RATING_COLUMNS = (Column("book", "Portfolio", labels=BOOK_LABELS),
                  Column("rating", "Risk rating"),
                  Column("band", "Band", labels=BAND_LABELS),
                  Column("volatility", "Volatility", "fraction"),
                  Column("tracking_error", "Tracking error", "fraction"))
DISTRIBUTION_COLUMNS = (Column("rating", "Risk rating"),
                        Column("band", "Band", labels=BAND_LABELS),
                        Column("days", "Days", "count"),
                        Column("share", "Share of days", "fraction"))

SHORT = "The rating needs a year of daily returns."


def _tracked(context: Context) -> pd.Series | None:
    return context.run.comparators[context.tracked] if context.tracked is not None else None


def _rating(context: Context) -> pd.DataFrame:
    performance = context.performance
    books = {name: context.run.comparators[name] for name in (INDEX, BENCHMARK)
             if name in context.run.comparators}
    found = rating.risk_rating(performance.returns, _tracked(context), books,
                               context.options.risk_rating, performance.periods_per_year)

    if found is None:
        raise Unavailable(TOO_SHORT, SHORT)

    return found


def _distribution(context: Context) -> pd.DataFrame:
    performance = context.performance
    found = rating.risk_distribution(performance.returns, _tracked(context),
                                     context.options.risk_rating, performance.periods_per_year)

    if found is None:
        raise Unavailable(TOO_SHORT, SHORT)

    return found


def statistics() -> list[Statistic]:
    """The risk rating statistics, in order."""
    return [
        Statistic("risk_rating", "Risk rating", RISK, TABLE, PERFORMANCE,
                  "The factsheet's 1 to 7 risk rating, on the volatility of weekly returns "
                  "over the last five years, and the band the tracking error puts the fund "
                  "in, from passive to unconstrained.", _rating, unit="fraction",
                  settings=SETTINGS, columns=RATING_COLUMNS),
        Statistic("risk_distribution", "Risk rating over time", RISK, TABLE, PERFORMANCE,
                  "The days the fund spent at each rating and band, on each day's volatility "
                  "and tracking error over the year ending that day.", _distribution,
                  unit="fraction", settings=SETTINGS, columns=DISTRIBUTION_COLUMNS),
    ]
