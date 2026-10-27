# src/beacon/analysis/performance/rolling.py
"""
A run's figures over a window that moves a day at a time: each day's figure
is the one its formula gives over the window ending that day, blank until
the first full window. And over every window, the worst, middle and best
return, and how often it gained or got ahead.
"""
# BN-378. The same definitions as the static figures: compounded returns,
# the decided Sharpe and information ratios, 0 for a correlation or beta
# against a series that does not vary.
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import formulas

# The windows given unless asked otherwise, in days: a quarter and a year.
WINDOWS = (63, 252)


def rolling_figures(returns: pd.Series,
                    window: int,
                    periods_per_year: int,
                    risk_free_rate: float = 0.0) -> pd.DataFrame:
    """Each day's return, volatility, Sharpe ratio, Sortino ratio and maximum
    drawdown over the *window* days ending that day."""
    excess = returns - formulas.daily_rate(risk_free_rate, periods_per_year)
    rolled = excess.rolling(window)
    spread = rolled.std()
    root = np.sqrt(periods_per_year)
    downside = np.sqrt(np.minimum(excess, 0.0).pow(2).rolling(window).mean())

    return pd.DataFrame({
        "return": _compounded(returns, window),
        "volatility": returns.rolling(window).std() * root,
        "sharpe_ratio": _ratio(rolled.mean(), spread) * root,
        "sortino_ratio": (rolled.mean() / downside.where(downside > formulas.NEGLIGIBLE_SPREAD)
                          * root),
        "max_drawdown": _max_drawdowns(returns, window),
    }, index=returns.index)


def rolling_relative(ours: pd.Series,
                     theirs: pd.Series,
                     window: int,
                     periods_per_year: int) -> pd.DataFrame:
    """Each day's relative return, tracking error, information ratio, beta
    and correlation against a comparator over the *window* days ending that
    day, the two on the same days."""
    active = ours - theirs
    rolled = active.rolling(window)
    spread = rolled.std()
    root = np.sqrt(periods_per_year)
    theirs_spread = theirs.rolling(window).std()
    ours_spread = ours.rolling(window).std()
    covariance = ours.rolling(window).cov(theirs)
    varies = theirs_spread > formulas.NEGLIGIBLE_SPREAD
    both_vary = varies & (ours_spread > formulas.NEGLIGIBLE_SPREAD)
    filled = active.rolling(window).count() == window

    return pd.DataFrame({
        "relative_return": _compounded(ours, window) - _compounded(theirs, window),
        "tracking_error": spread * root,
        "information_ratio": _ratio(rolled.mean(), spread, missing=True) * root,
        "beta": (covariance / theirs_spread ** 2).where(varies, 0.0).where(filled),
        "correlation": (covariance / (ours_spread * theirs_spread)).where(both_vary,
                                                                          0.0).where(filled),
    }, index=ours.index)


def outcomes(window_returns: pd.Series,
             relative_returns: pd.Series | None = None) -> dict[str, float | None]:
    """Over every full window: the worst, middle and best return, the share
    of windows that gained, and the share ahead of the comparator."""
    values = window_returns.dropna()

    if values.empty:
        return dict.fromkeys(("worst", "median", "best", "share_positive", "share_ahead"))

    ahead = relative_returns.dropna() if relative_returns is not None else None

    return {"worst": float(values.min()),
            "median": float(values.median()),
            "best": float(values.max()),
            "share_positive": float((values > 0).mean()),
            "share_ahead": (float((ahead > 0).mean())
                            if ahead is not None and len(ahead) else None)}


def _compounded(returns: pd.Series,
                window: int) -> pd.Series:
    """Each day's compounded return over the *window* days ending that
    day."""
    values = returns.to_numpy(dtype=float)
    found = np.full(len(values), np.nan)

    if len(values) >= window:
        found[window - 1:] = np.prod(1.0 + sliding_window_view(values, window), axis=1) - 1.0

    return pd.Series(found, index=returns.index)


def _max_drawdowns(returns: pd.Series,
                   window: int) -> pd.Series:
    """Each day's maximum drawdown over the *window* days ending that day,
    measured from 1 before the window's first day."""
    values = returns.to_numpy(dtype=float)
    found = np.full(len(values), np.nan)

    if len(values) >= window:
        paths = np.cumprod(1.0 + sliding_window_view(values, window), axis=1)
        peaks = np.maximum(np.maximum.accumulate(paths, axis=1), 1.0)
        found[window - 1:] = np.minimum((paths / peaks - 1.0).min(axis=1), 0.0)

    return pd.Series(found, index=returns.index)


def _ratio(mean: pd.Series,
           spread: pd.Series,
           missing: bool = False) -> pd.Series:
    """*mean* over *spread*: 0 where the spread is negligible, or blank when
    *missing*, as the static ratio has it."""
    ratio = mean / spread.where(spread > formulas.NEGLIGIBLE_SPREAD)
    filled = spread.notna()

    return ratio.where(ratio.notna() | ~filled, np.nan if missing else 0.0)
