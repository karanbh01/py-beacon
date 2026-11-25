# src/beacon/analysis/performance/name_figures.py
"""
Each name's return, risk and figures against a comparator, from a table of
names' returns, a column a name:

    figure("beta", returns, theirs, periods_per_year)        # a row a name
    rolled("volatility", returns, None, 63, periods_per_year)  # dates by names

A name is measured over the days it has a return, and against a comparator
over the days both have. Figures are measured as the fund's are
(`formulas`): the total return compounded, the volatility and tracking error
the standard deviation scaled by the square root of the periods a year, the
beta the covariance over the comparator's variance, and the residual risk
the regression's residuals' spread with two degrees of freedom spent.

Returns can be measured daily, or compounded into weeks ending on Friday or
calendar months first (`on_basis`).
"""
# BN-452 (Karan, 2026-11-24): daily by default, weekly or monthly asked for.
from collections.abc import Callable

import numpy as np
import pandas as pd

from .formulas import NEGLIGIBLE_SPREAD

DAILY, WEEKLY, MONTHLY = "daily", "weekly", "monthly"
BASES = {WEEKLY: ("W-FRI", 52), MONTHLY: ("ME", 12)}

OWN = ("total_return", "volatility", "max_drawdown")
RELATIVE = ("beta", "correlation", "tracking_error", "residual_risk")
FIGURES = (*OWN, *RELATIVE)


def on_basis(returns: pd.DataFrame,
             theirs: pd.Series | None,
             frequency: str,
             periods_per_year: int) -> tuple[pd.DataFrame, pd.Series | None, int]:
    """*returns* and *theirs* on the basis *frequency* names, with the
    periods a year it has.

    Raises:
        ValueError: If *frequency* is not daily, weekly or monthly.
    """
    if frequency == DAILY:
        return returns, theirs, periods_per_year

    if frequency not in BASES:
        raise ValueError(f"frequency is {DAILY!r}, {WEEKLY!r} or {MONTHLY!r}, got {frequency!r}.")

    rule, periods = BASES[frequency]

    def compounded(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        grown: pd.DataFrame | pd.Series = values + 1.0

        return grown.resample(rule).prod(min_count=1) - 1.0

    return (compounded(returns), None if theirs is None else compounded(theirs).rename(None),
            periods)


def figure(name: str,
           returns: pd.DataFrame,
           theirs: pd.Series | None,
           periods_per_year: int) -> pd.Series:
    """Figure *name* of each name over all of *returns*, a value a name.

    Raises:
        ValueError: If *name* is not one of `FIGURES`, or is measured
            against a comparator and *theirs* is None.
    """
    if name not in FIGURES:
        raise ValueError(f"Unknown figure {name!r}. Known: {', '.join(FIGURES)}.")

    if name in OWN:
        return _OWN[name](returns, periods_per_year)

    if theirs is None:
        raise ValueError(f"{name} is measured against a comparator.")

    return _relative(name, returns, theirs.reindex(returns.index), periods_per_year)


def rolled(name: str,
           returns: pd.DataFrame,
           theirs: pd.Series | None,
           window: int,
           periods_per_year: int,
           ends: list[int] | None = None) -> pd.DataFrame:
    """Figure *name* of each name over each trailing *window* of returns,
    dates by names: every full window, or those ending at positions *ends*.
    A name is measured only over windows it has a return on every day of."""
    if name == "max_drawdown" or ends is not None:
        positions = ends if ends is not None else list(range(window - 1, len(returns)))
        rows = {returns.index[end]: _full(figure(name, returns.iloc[end - window + 1:end + 1],
                                                 _cut(theirs, returns, end, window),
                                                 periods_per_year),
                                          returns.iloc[end - window + 1:end + 1])
                for end in positions}

        return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=returns.columns)

    if name == "total_return":
        return np.expm1(np.log1p(returns).rolling(window, min_periods=window).sum())

    if name == "volatility":
        return returns.rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)

    if theirs is None:
        raise ValueError(f"{name} is measured against a comparator.")

    return _rolling_relative(name, returns, theirs.reindex(returns.index), window,
                             periods_per_year)


def _total_return(returns: pd.DataFrame,
                  periods_per_year: int) -> pd.Series:
    return (1.0 + returns).prod(min_count=1) - 1.0


def _volatility(returns: pd.DataFrame,
                periods_per_year: int) -> pd.Series:
    return returns.std() * np.sqrt(periods_per_year)


def _max_drawdown(returns: pd.DataFrame,
                  periods_per_year: int) -> pd.Series:
    """The deepest fall of each name's level from its running peak, the
    level starting from 1 before its first return."""
    levels = (1.0 + returns.fillna(0.0)).cumprod()
    peaks = levels.cummax().clip(lower=1.0)
    found = (levels / peaks - 1.0).min().clip(upper=0.0)

    return found.where(returns.notna().any())


_OWN: dict[str, Callable[[pd.DataFrame, int], pd.Series]] = {
    "total_return": _total_return, "volatility": _volatility, "max_drawdown": _max_drawdown}


def _relative(name: str,
              returns: pd.DataFrame,
              theirs: pd.Series,
              periods_per_year: int) -> pd.Series:
    """A figure of each name against *theirs*, over the days both have."""
    both = returns.notna() & theirs.notna().to_numpy()[:, None]
    ours = returns.where(both)
    comparator = pd.DataFrame(np.where(both, theirs.to_numpy()[:, None], np.nan),
                              index=returns.index, columns=returns.columns)
    count = both.sum()

    if name == "tracking_error":
        return (ours - comparator).std() * np.sqrt(periods_per_year)

    ours_mean, theirs_mean = ours.mean(), comparator.mean()
    covariance = ((ours - ours_mean) * (comparator - theirs_mean)).sum() / (count - 1)
    variance = ((comparator - theirs_mean) ** 2).sum() / (count - 1)
    own = ((ours - ours_mean) ** 2).sum() / (count - 1)

    return _from_moments(name, covariance, variance, own, count, periods_per_year)


def _rolling_relative(name: str,
                      returns: pd.DataFrame,
                      theirs: pd.Series,
                      window: int,
                      periods_per_year: int) -> pd.DataFrame:
    """A figure of each name against *theirs* over each full window."""
    if name == "tracking_error":
        active = returns.sub(theirs, axis=0)

        return active.rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)

    rolling = returns.rolling(window, min_periods=window)
    covariance = rolling.cov(theirs)
    spread = theirs.rolling(window, min_periods=window).var().to_numpy()
    variance = pd.DataFrame(np.broadcast_to(spread[:, None], returns.shape),
                            index=returns.index, columns=returns.columns)
    count = pd.DataFrame(window, index=returns.index, columns=returns.columns)

    return _from_moments(name, covariance, variance, rolling.var(), count, periods_per_year)


def _from_moments(name: str,
                  covariance: pd.Series | pd.DataFrame,
                  variance: pd.Series | pd.DataFrame,
                  own: pd.Series | pd.DataFrame,
                  count: pd.Series | pd.DataFrame,
                  periods_per_year: int) -> pd.Series | pd.DataFrame:
    """Beta, correlation or residual risk from the covariance, the
    comparator's variance and the name's own, over *count* days each. A
    comparator that does not vary leaves nothing to be sensitive to: a beta
    of 0, as `formulas.beta` gives it."""
    varies = variance > NEGLIGIBLE_SPREAD ** 2
    beta = (covariance / variance).where(varies, 0.0)

    if name == "beta":
        return beta.where(count > 1)

    if name == "correlation":
        return (covariance / np.sqrt(variance * own)).where(varies & (own > 0))

    # The residuals' spread, with the degrees of freedom the regression
    # spent (`relative_analytics.regression`).
    unexplained = (own - beta * covariance).clip(lower=0.0) * (count - 1)
    freedom = (count - 2).where(varies, count - 1)

    return np.sqrt(unexplained / freedom * periods_per_year).where(count > 2)


def _cut(theirs: pd.Series | None,
         returns: pd.DataFrame,
         end: int,
         window: int) -> pd.Series | None:
    return None if theirs is None else theirs.reindex(returns.index).iloc[end - window + 1:end + 1]


def _full(found: pd.Series,
          window: pd.DataFrame) -> pd.Series:
    """*found*, blank for a name without a return on every day of the
    window."""
    return found.where(window.notna().all())


__all__ = ["BASES", "DAILY", "FIGURES", "MONTHLY", "OWN", "RELATIVE", "WEEKLY", "figure",
           "on_basis", "rolled"]
