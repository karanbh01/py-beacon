# src/beacon/analysis/performance/members.py
"""
Each name's risk and return, the fund's and its comparators' measured the
same way, and the efficient frontier through them in hindsight.

    member_risk_return(name_returns, index_weights, fund_weights, settings)
    portfolio_risk_return(books, settings)
    hindsight_frontier(name_returns, index_weights, settings)

## Risk and return

Daily returns are compounded into weeks ending on Friday (or calendar
months), and each name is measured over the weeks it has: its return the
mean weekly return times 52 (12 for months), its volatility the standard
deviation of its weekly returns times the square root of 52. The mean, not
the compounded return, is what a mean-variance frontier is drawn in, so a
single name sits on or inside it. A run's first and last weeks may be part
weeks.

Every name the index held on any day of the run is measured, not only its
latest members. A name with under a year of weeks is left out, and counted.

## The frontier

The long-only, fully invested mean-variance frontier of the 100 largest
latest index members with a return in every week measured, from their
sample covariance and mean returns: a name listed part way through is left
out, so one late listing cannot shorten every name's weeks. It is the least
volatility for each return between the minimum-variance portfolio's and the
best name's, drawn with what the run went on to know: a frontier in
hindsight, not one an investor could have held. It needs a year of weeks,
and more weeks than names.
"""
# BN-429. Agreed with Karan on 2026-11-21 for beacon-ui's Risk & correlation
# view: every name ever in the index, the frontier from the 100 largest latest
# members with a plain sample covariance, returns as the mean times 52.
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...exceptions import CalculationError
from ...optimise.frontier import efficient_frontier
from ...risk.model import estimate_risk_model

WEEKLY, MONTHLY = "weekly", "monthly"
FREQUENCIES = {WEEKLY: ("W-FRI", 52), MONTHLY: ("ME", 12)}

FIGURE_COLUMNS = ["return", "volatility", "observations", "first", "last"]
MEMBER_COLUMNS = ["name", *FIGURE_COLUMNS, "in_index_at_end", "fund_weight", "index_weight"]
PORTFOLIO_COLUMNS = ["book", *FIGURE_COLUMNS]
FRONTIER_COLUMNS = ["volatility", "return", "minimum_variance"]


@dataclass(frozen=True)
class RiskReturnSettings:
    """How names' risk and return, and the frontier, are measured.

    Attributes:
        frequency: The returns measured: ``weekly`` or ``monthly``.
        frontier_names: How many of the largest latest index members the
            frontier is drawn from.
        frontier_points: How many portfolios the frontier is traced at.
    """
    frequency: str = WEEKLY
    frontier_names: int = 100
    frontier_points: int = 50

    def __post_init__(self) -> None:
        if self.frequency not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {', '.join(FREQUENCIES)}; "
                             f"got {self.frequency!r}.")

        if self.frontier_names < 2:
            raise ValueError(f"frontier_names must be at least 2; got {self.frontier_names}.")

        if self.frontier_points < 2:
            raise ValueError(f"frontier_points must be at least 2; got {self.frontier_points}.")

    @property
    def per_year(self) -> int:
        """How many of these periods make a year."""
        return FREQUENCIES[self.frequency][1]


@dataclass(frozen=True)
class Members:
    """Every name the index held, measured.

    Attributes:
        table: A row a name measured.
        measured: How many names were measured.
        too_short: How many had under a year of periods and were left out.
    """
    table: pd.DataFrame
    measured: int
    too_short: int


@dataclass(frozen=True)
class Frontier:
    """The frontier in hindsight.

    Attributes:
        table: A row a point, by rising return; the first is the
            minimum-variance portfolio.
        names: How many names it was drawn from.
        periods: How many periods they all had.
        first: The first period's end.
        last: The last period's end.
    """
    table: pd.DataFrame
    names: int
    periods: int
    first: pd.Timestamp
    last: pd.Timestamp


def compounded(returns: pd.DataFrame | pd.Series,
               frequency: str = WEEKLY) -> pd.DataFrame | pd.Series:
    """Daily returns compounded into weeks ending on Friday, or months; a
    period with no return is blank."""
    if returns.empty:
        return returns

    rule = FREQUENCIES[frequency][0]

    return (1.0 + returns).resample(rule).prod(min_count=1) - 1.0


def risk_return(periods: pd.DataFrame,
                per_year: int) -> pd.DataFrame:
    """Each column's return (the mean times *per_year*), volatility (the
    standard deviation times its square root), periods measured, and the
    ends of the first and last of them."""
    known = periods.notna()
    counted = known.sum()
    ends = pd.Series(periods.index, index=periods.index)

    return pd.DataFrame({"return": periods.mean() * per_year,
                         "volatility": periods.std() * np.sqrt(per_year),
                         "observations": counted.astype(int),
                         "first": [ends[known[name]].min() for name in periods.columns],
                         "last": [ends[known[name]].max() for name in periods.columns]},
                        index=periods.columns)


def member_risk_return(name_returns: pd.DataFrame,
                       index_weights: pd.DataFrame,
                       fund_weights: pd.DataFrame | None,
                       settings: RiskReturnSettings) -> Members:
    """The risk and return of every name the index held on any day.

    Args:
        name_returns: Each name's daily returns, dates by name, over the
            days measured.
        index_weights: The index's daily weights, dates by name, over the
            same days.
        fund_weights: The fund's daily weights; None for a run without.
        settings: The periods measured.

    Returns:
        Members: A row a name with a year of periods or more, by name:
        ``name``, ``return``, ``volatility``, ``observations``, ``first``,
        ``last``, ``in_index_at_end`` and the latest ``fund_weight`` and
        ``index_weight``, blank for a name out of that portfolio.
    """
    held = _held(index_weights)
    members = [name for name in held.index if name in name_returns.columns]
    figures = risk_return(compounded(name_returns[members], settings.frequency),
                          settings.per_year)
    long_enough = figures[figures["observations"] >= settings.per_year]
    index_latest = _latest(index_weights)
    fund_latest = _latest(fund_weights) if fund_weights is not None else pd.Series(dtype=float)
    table = long_enough.rename_axis("name").reset_index()
    table["in_index_at_end"] = [bool(index_latest.get(name, 0.0) > 0)
                                for name in table["name"]]
    table["fund_weight"] = [_weight(fund_latest, name) for name in table["name"]]
    table["index_weight"] = [_weight(index_latest, name) for name in table["name"]]

    return Members(table=table[MEMBER_COLUMNS].sort_values("name", ignore_index=True),
                   measured=len(table), too_short=len(figures) - len(long_enough))


def portfolio_risk_return(books: Mapping[str, pd.Series],
                          settings: RiskReturnSettings) -> pd.DataFrame:
    """Each portfolio's risk and return, from its daily returns by name,
    measured as the names' are."""
    if not books:
        return pd.DataFrame(columns=PORTFOLIO_COLUMNS)

    daily = pd.DataFrame(dict(books))
    figures = risk_return(compounded(daily, settings.frequency), settings.per_year)

    return figures.rename_axis("book").reset_index()[PORTFOLIO_COLUMNS]


def hindsight_frontier(name_returns: pd.DataFrame,
                       index_weights: pd.DataFrame,
                       settings: RiskReturnSettings) -> Frontier | None:
    """The frontier of the largest latest index members with a return in
    every period measured; None with fewer than two such names, under a
    year of periods, or no more periods than names."""
    latest = _latest(index_weights)
    largest = latest[latest > 0].sort_values(ascending=False, kind="stable")
    candidates = [name for name in largest.index if name in name_returns.columns]
    periods = compounded(name_returns[candidates], settings.frequency)
    periods = periods[periods.notna().any(axis=1)]
    throughout = periods.columns[periods.notna().all()]
    chosen = list(throughout[:settings.frontier_names])
    common = periods[chosen]

    if len(chosen) < 2 or len(common) <= len(chosen) or len(common) < settings.per_year:
        return None

    model = estimate_risk_model(common, intensity=0.0, periods_per_year=settings.per_year)
    means = (common.mean() * settings.per_year).to_dict()

    try:
        traced = efficient_frontier(model, means, points=settings.frontier_points)
    except CalculationError:
        return None

    table = pd.DataFrame({"volatility": traced.volatilities,
                          "return": traced.expected_returns,
                          "minimum_variance": [index == 0 for index in range(len(traced.points))]},
                         columns=FRONTIER_COLUMNS)

    return Frontier(table=table, names=len(chosen), periods=len(common),
                    first=pd.Timestamp(common.index[0]), last=pd.Timestamp(common.index[-1]))


def _held(weights: pd.DataFrame) -> pd.Series:
    """Whether each name had weight on any day, for the names that did."""
    found = (weights.fillna(0.0) > 0).any()
    found.index = found.index.astype(str)

    return found[found]


def _latest(weights: pd.DataFrame) -> pd.Series:
    """The last day's weights, by name."""
    if weights.empty:
        return pd.Series(dtype=float)

    row = weights.iloc[-1].fillna(0.0).astype(float)
    row.index = row.index.astype(str)

    return row


def _weight(latest: pd.Series,
            name: str) -> float | None:
    value = float(latest.get(name, 0.0))

    return value if value > 0 else None
