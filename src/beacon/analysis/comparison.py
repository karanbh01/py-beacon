# src/beacon/analysis/comparison.py
"""
A strategy against a comparator: the comparator's own figures and the
strategy's relative to it, over the window they share.

The strategy is given as its daily returns, the first running from its
starting point (a backtest's initial capital), and the comparator as its
levels. Over the run's whole window the comparator's first day returns
nothing, so its figures run from its level that day, as a backtest's tracking
does; over a later window both start there. Every figure is defined as a
backtest's metrics are (see `beacon.backtest.metrics.headline_metrics`), so a
comparison against a backtest's own index agrees with its tracking error and
tracking difference.

Figures by period split the same returns into calendar years, quarters or
months, each period from the previous period's close and the first from the
start, so the periods compound to the whole window's figures.
"""
# BN-323 and BN-324: every figure a run shows, computed here rather than by
# each client.
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..assumptions import DEFAULT_PERIODS_PER_YEAR
from ..backtest.metrics import headline_metrics

# How a calendar period is named and grouped.
ANNUAL, QUARTERLY, MONTHLY = "annual", "quarterly", "monthly"
FREQUENCIES = (ANNUAL, QUARTERLY, MONTHLY)


@dataclass(frozen=True)
class Comparison:
    """A strategy against one comparator over the window they share.

    Attributes:
        start: The window's first day.
        end: Its last day.
        observations: Its days.
        level: The comparator's level over it, rebased to 100 on its first day.
        own: The comparator's figures: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown and hit_rate_monthly.
        relative: The strategy against it: relative_return,
            annualised_relative_return, tracking_error, information_ratio,
            correlation, beta and hit_rate_monthly.
    """
    start: pd.Timestamp
    end: pd.Timestamp
    observations: int
    level: pd.Series
    own: dict[str, float | None]
    relative: dict[str, float | None]


def hit_rate_monthly(returns: pd.Series) -> float | None:
    """The share of calendar months whose compounded return was positive;
    None with no returns."""
    monthly = _compounded(returns, MONTHLY)

    return float((monthly > 0).mean()) if len(monthly) else None


def compare(strategy_returns: pd.Series,
            comparator_levels: pd.Series,
            periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
            risk_free_rate: float = 0.0) -> Comparison | None:
    """A strategy against a comparator over the days both have.

    Args:
        strategy_returns: The strategy's daily returns, the first from its
            starting point.
        comparator_levels: The comparator's levels.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate Sharpe ratios are in excess of.

    Returns:
        Comparison | None: The comparison, or None when they share fewer than
        two days.
    """
    levels = comparator_levels.dropna()
    window = strategy_returns.index.intersection(levels.index)

    if len(window) < 2:
        return None

    levels = levels.loc[window].astype(float)
    ours = strategy_returns.loc[window].astype(float)
    theirs = levels.pct_change()
    theirs.iloc[0] = 0.0

    # From a later start, the strategy starts there too, as the comparator does.
    if window[0] != strategy_returns.index[0]:
        ours = ours.copy()
        ours.iloc[0] = 0.0

    return Comparison(start=window[0],
                      end=window[-1],
                      observations=len(window),
                      level=levels / levels.iloc[0] * 100.0,
                      own=_own(theirs, levels, periods_per_year, risk_free_rate),
                      relative=_relative(ours, theirs, periods_per_year))


def by_period(strategy_returns: pd.Series,
              frequency: str,
              periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
              risk_free_rate: float = 0.0) -> list[dict[str, Any]]:
    """The strategy's figures in each calendar period.

    Each period's returns are its own days', so the first runs from the
    strategy's starting point and each later one from the previous close.

    Args:
        strategy_returns: The strategy's daily returns, the first from its
            starting point.
        frequency: ``"annual"``, ``"quarterly"`` or ``"monthly"``.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate the Sharpe ratio is in excess of.

    Returns:
        list: One ``{period, start, end, returns, figures}`` per period, in
        order; ``returns`` are the period's own, for comparing it.

    Raises:
        ValueError: If the frequency is not one of the three.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"Unknown frequency {frequency!r}. Known: {', '.join(FREQUENCIES)}.")

    periods = []

    for label, returns in _grouped(strategy_returns, frequency):
        path = _path(returns)
        figures = headline_metrics(returns, float(path.iloc[-1] - 1.0), path,
                                   periods_per_year, risk_free_rate)
        periods.append({"period": label, "start": returns.index[0], "end": returns.index[-1],
                        "returns": returns,
                        "figures": {**figures, "positive_days": float((returns > 0).mean())}})

    return periods


def compare_returns(strategy_returns: pd.Series,
                    comparator_returns: pd.Series,
                    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
                    risk_free_rate: float = 0.0) -> dict[str, dict[str, float | None]] | None:
    """A strategy against a comparator over a period, from both's daily
    returns there: the comparator's ``own`` figures and the ``relative`` ones,
    or None when the period has no common days."""
    window = strategy_returns.index.intersection(comparator_returns.index)

    if len(window) < 1:
        return None

    ours = strategy_returns.loc[window]
    theirs = comparator_returns.loc[window]

    return {"own": _own(theirs, _path(theirs), periods_per_year, risk_free_rate),
            "relative": _relative(ours, theirs, periods_per_year)}


def daily_returns(levels: pd.Series) -> pd.Series:
    """A comparator's daily returns, the first day's nothing: the returns a
    comparison over the whole window uses."""
    returns = levels.astype(float).pct_change()

    if len(returns):
        returns.iloc[0] = 0.0

    return returns


def _own(returns: pd.Series,
         path: pd.Series,
         periods_per_year: int,
         risk_free_rate: float) -> dict[str, float | None]:
    """A series' own figures, from its returns and the level they move."""
    total = float(path.iloc[-1] / path.iloc[0] - 1.0)
    figures: dict[str, float | None] = {**headline_metrics(returns, total, path,
                                                           periods_per_year, risk_free_rate)}
    figures["hit_rate_monthly"] = hit_rate_monthly(returns)

    return figures


def _relative(ours: pd.Series,
              theirs: pd.Series,
              periods_per_year: int) -> dict[str, float | None]:
    """The strategy against the comparator, over the same days."""
    n = len(ours)
    years = n / float(periods_per_year)
    our_total = float((1.0 + ours).prod() - 1.0)
    their_total = float((1.0 + theirs).prod() - 1.0)
    annualised = (((1.0 + our_total) ** (1.0 / years) - (1.0 + their_total) ** (1.0 / years))
                  if years > 0 else 0.0)
    difference = ours - theirs
    tracking_error = (float(difference.std() * np.sqrt(periods_per_year)) if n > 1 else None)
    variance = float(theirs.var()) if n > 1 else 0.0
    monthly_ours = _compounded(ours, MONTHLY)
    monthly_theirs = _compounded(theirs, MONTHLY)

    return {
        "relative_return": our_total - their_total,
        "annualised_relative_return": float(annualised),
        "tracking_error": tracking_error,
        "information_ratio": (float(annualised / tracking_error)
                              if tracking_error else None),
        "correlation": (float(ours.corr(theirs))
                        if n > 1 and ours.std() > 0 and theirs.std() > 0 else None),
        "beta": float(ours.cov(theirs) / variance) if variance > 0 else None,
        "hit_rate_monthly": (float((monthly_ours > monthly_theirs).mean())
                             if len(monthly_ours) else None),
    }


def _path(returns: pd.Series) -> pd.Series:
    """The level a run of returns moves, from 1 before its first."""
    return pd.concat([pd.Series([1.0]), (1.0 + returns.reset_index(drop=True)).cumprod()],
                     ignore_index=True)


def _compounded(returns: pd.Series,
                frequency: str) -> pd.Series:
    """Returns compounded within each calendar period."""
    if returns.empty:
        return pd.Series(dtype=float)

    return pd.Series({label: float((1.0 + period).prod() - 1.0)
                      for label, period in _grouped(returns, frequency)})


def _grouped(returns: pd.Series,
             frequency: str) -> list[tuple[str, pd.Series]]:
    """Returns split into calendar periods, each named, in order."""
    index = pd.DatetimeIndex(returns.index)
    keys = {ANNUAL: index.strftime("%Y"),
            QUARTERLY: [f"{day.year} Q{(day.month - 1) // 3 + 1}" for day in index],
            MONTHLY: index.strftime("%Y-%m")}[frequency]

    return [(str(label), period) for label, period
            in returns.groupby(pd.Index(keys, name="period"), sort=False)]
