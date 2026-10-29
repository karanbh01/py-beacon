# src/beacon/analysis/performance/comparisons.py
"""
A run against a comparator, and its figures by calendar period.

A run is given as its daily returns, the first running from its starting
point (a backtest's initial capital), and a comparator as its levels. Over
the run's whole window the comparator's first day returns nothing, so its
figures run from its level that day, as a run's tracking does; over a later
window both start there. Every figure is one of
`beacon.analysis.performance.formulas`, so a comparison against a run's own
index agrees with its tracking error and tracking difference.

Figures by period split the same returns into calendar years, quarters or
months, each period from the previous period's close and the first from the
start, so the periods compound to the whole window's figures.
"""
# BN-372: what `beacon.analysis.comparison` computed (BN-323, BN-324), on the
# formulas, reached through `PerformanceAnalytics`.
from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from ...data.base import as_of_position
from . import formulas


@dataclass(frozen=True)
class Comparison:
    """A run against one comparator over the window they share.

    Attributes:
        start: The window's first day.
        end: Its last day.
        observations: Its days.
        level: The comparator's level over it, rebased to 100 on its first day.
        own: The comparator's figures: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown and hit_rate_monthly.
        relative: The run against it: relative_return,
            annualised_relative_return, tracking_error, information_ratio,
            correlation, beta and hit_rate_monthly.
    """
    start: pd.Timestamp
    end: pd.Timestamp
    observations: int
    level: pd.Series
    own: dict[str, float | None]
    relative: dict[str, float | None]


@dataclass(frozen=True)
class Period:
    """A run's figures in one calendar period.

    Attributes:
        period: Its name: ``2024``, ``2024 Q1`` or ``2024-01``.
        start: Its first day.
        end: Its last day.
        returns: The run's returns in it.
        figures: The run's headline figures in it, and the share of its days
            with a positive return (``positive_days``).
        against: Each comparator's ``own`` and ``relative`` figures in it, by
            name; None for a comparator with no days in it.
    """
    period: str
    start: pd.Timestamp
    end: pd.Timestamp
    returns: pd.Series
    figures: dict[str, float]
    against: dict[str, dict[str, dict[str, float | None]] | None] = field(default_factory=dict)


def comparison_of(returns: pd.Series,
                  comparator_levels: pd.Series,
                  periods_per_year: int,
                  risk_free_rate: float = 0.0) -> Comparison | None:
    """A run against a comparator over the days both have; None when they
    share fewer than two.

    Args:
        returns: The run's daily returns, the first from its starting point.
        comparator_levels: The comparator's levels.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate Sharpe ratios are in excess of.
    """
    ours, theirs = aligned(returns, comparator_levels)

    if ours.empty:
        return None

    window = ours.index
    levels = comparator_levels.dropna().loc[window].astype(float)

    return Comparison(start=window[0],
                      end=window[-1],
                      observations=len(window),
                      level=levels / levels.iloc[0] * 100.0,
                      own=own_figures(theirs, levels, periods_per_year, risk_free_rate),
                      relative=relative_figures(ours, theirs, periods_per_year))


def aligned(returns: pd.Series,
            comparator_levels: pd.Series,
            before: pd.Timestamp | None = None) -> tuple[pd.Series, pd.Series]:
    """A run's and a comparator's daily returns on the days both have, empty
    when they share fewer than two: the comparator's first day returns
    nothing from its level that day, and so, when the comparator starts after
    the run, does the run's.

    For returns over a window that starts after the run, *before* is the
    run's close before the window's first day: the comparator's first return
    is then from its own close on or before it, as the run's is."""
    levels = comparator_levels.dropna()
    window = returns.index.intersection(levels.index)

    if len(window) < 2:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        return empty, empty

    ours = returns.loc[window].astype(float)
    theirs = formulas.returns_from_start(levels.loc[window].astype(float))
    # BN-398: a window that starts after the run measured the comparator's
    # first day as nothing, leaving a day of its move out of every windowed
    # relative figure.
    prior = as_of_position(levels.index, before) if before is not None else None

    if prior is not None and window[0] == returns.index[0] and levels.index[prior] < window[0]:
        theirs.iloc[0] = float(levels.loc[window[0]]) / float(levels.iloc[prior]) - 1.0

    # From a later start, the run starts there too, as the comparator does.
    if window[0] != returns.index[0]:
        ours = ours.copy()
        ours.iloc[0] = 0.0

    return ours, theirs


def periods_of(returns: pd.Series,
               frequency: str,
               periods_per_year: int,
               risk_free_rate: float = 0.0,
               comparators: Mapping[str, pd.Series] | None = None) -> list[Period]:
    """A run's figures in each calendar period, and each comparator's in it.

    Each period's returns are its own days', so the first runs from the run's
    starting point and each later one from the previous close.

    Args:
        returns: The run's daily returns, the first from its starting point.
        frequency: ``"annual"``, ``"quarterly"`` or ``"monthly"``.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate the Sharpe ratio is in excess of.
        comparators: Each comparator's daily returns, by name, its first day's
            nothing.

    Raises:
        ValueError: If the frequency is not one of the three.
    """
    periods = []

    for label, ours in formulas.grouped(returns, frequency):
        path = formulas.path_of(ours)
        figures = formulas.headline_figures(ours, float(path.iloc[-1] - 1.0), path,
                                            periods_per_year, risk_free_rate)
        against = {name: period_against(ours, theirs, periods_per_year, risk_free_rate)
                   for name, theirs in (comparators or {}).items()}
        periods.append(Period(period=label, start=ours.index[0], end=ours.index[-1],
                              returns=ours,
                              figures={**figures,
                                       "positive_days": formulas.positive_share(ours)},
                              against=against))

    return periods


def period_against(ours: pd.Series,
                   theirs: pd.Series,
                   periods_per_year: int,
                   risk_free_rate: float = 0.0) -> dict[str, dict[str, float | None]] | None:
    """A comparator's ``own`` figures and the run's ``relative`` ones over a
    period, from both's daily returns in it; None with no common days."""
    window = ours.index.intersection(theirs.index)

    if len(window) < 1:
        return None

    ours, theirs = ours.loc[window], theirs.loc[window]

    return {"own": own_figures(theirs, formulas.path_of(theirs), periods_per_year,
                               risk_free_rate),
            "relative": relative_figures(ours, theirs, periods_per_year)}


def own_figures(returns: pd.Series,
                path: pd.Series,
                periods_per_year: int,
                risk_free_rate: float) -> dict[str, float | None]:
    """A series' own figures, from its returns and the level they move."""
    total = formulas.total_return(float(path.iloc[0]), float(path.iloc[-1]))
    figures: dict[str, float | None] = {**formulas.headline_figures(returns, total, path,
                                                                    periods_per_year,
                                                                    risk_free_rate)}
    figures["hit_rate_monthly"] = formulas.hit_rate_monthly(returns)

    return figures


def relative_figures(ours: pd.Series,
                     theirs: pd.Series,
                     periods_per_year: int) -> dict[str, float | None]:
    """The run against the comparator, over the same days."""
    our_total = formulas.compounded_return(ours)
    their_total = formulas.compounded_return(theirs)
    annualised = formulas.annualised_relative_return(our_total, their_total, len(ours),
                                                     periods_per_year)
    active = ours - theirs

    return {"relative_return": our_total - their_total,
            "annualised_relative_return": annualised,
            "tracking_error": formulas.tracking_error(active, periods_per_year),
            "information_ratio": formulas.information_ratio(active, periods_per_year),
            "correlation": formulas.correlation(ours, theirs),
            "beta": formulas.beta(ours, theirs),
            "hit_rate_monthly": formulas.relative_hit_rate_monthly(ours, theirs)}
