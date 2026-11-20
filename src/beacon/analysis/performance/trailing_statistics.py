# src/beacon/analysis/performance/trailing_statistics.py
"""
A run's main figures over each trailing window, each period to date and the
whole run: the total and annualised return, volatility, maximum drawdown and
Sharpe ratio of the fund and of each comparator, and the fund's tracking
difference and tracking error against each.

The windows are `trailing_returns`'s (`return_figures.window_starts`), and
each figure is worked out as the whole run's is, from the window's daily
returns: a window starts from the close before its first day. A comparator's
figures are over its own window, and its figures against the fund only when
the two cover the same days, as in `trailing_table`.

- The annualised return is given only beyond a year, as in
  `trailing_returns`.
- The Sharpe ratio is not given over a window shorter than six months
  (`NO_SHARPE`): a month or a quarter of days says too little about it.
- A window longer than the run has no figures.
"""
# BN-422. Agreed with beacon-ui for its Summary (BU-294); the short windows'
# Sharpe ratio left out by Karan on 2026-11-20 (QTD with them, as it is
# shorter than 3M).
from collections.abc import Mapping

import pandas as pd

from . import formulas, return_figures

FUND, INDEX, BENCHMARK = "fund", "index", "benchmark"
COMPARATORS = (INDEX, BENCHMARK)

# Each book's own figures, then the fund's against a comparator, which are
# given under the comparator's column.
OWN = ("total_return", "annualised_return", "volatility", "max_drawdown", "sharpe_ratio")
AGAINST = ("tracking_difference", "tracking_error")

# The windows too short for a Sharpe ratio.
NO_SHARPE = ("1M", "3M", "MTD", "QTD", "YTD")

# What each figure is measured in, so a row says how it reads.
UNITS = {**dict.fromkeys((*OWN, *AGAINST), "fraction"), "sharpe_ratio": "ratio"}

COLUMNS = ("period", "statistic", "unit", FUND, *COMPARATORS)

Figures = dict[str, float | None]


def trailing_statistics(returns: pd.Series,
                        periods_per_year: int,
                        risk_free_rate: float,
                        comparators: Mapping[str, tuple[pd.Series, pd.Series]],
                        inception: Mapping[str, float] | None = None) -> pd.DataFrame:
    """The figures over each window, a row a window and figure.

    Args:
        returns: The fund's daily returns, the first from its starting point.
        periods_per_year: How many daily returns make a year.
        risk_free_rate: The annual rate the Sharpe ratio is in excess of.
        comparators: For the index and the benchmark, by name, the fund's and
            its daily returns on the days both have
            (`tables.comparator_returns`).
        inception: The fund's headline figures over the whole run, which
            start from what the fund started with; worked out from
            *returns* when None.

    Returns:
        pd.DataFrame: Columns ``period``, ``statistic``, ``unit`` (``fraction``,
        or ``ratio`` for the Sharpe ratio), ``fund``, ``index`` and
        ``benchmark``; a comparator the run lacks is blank.
    """
    starts = return_figures.window_starts(returns)
    ours = {window: _own(returns, window, start, periods_per_year, risk_free_rate)
            for window, start in starts.items()}

    if inception is not None and ours[return_figures.INCEPTION] is not None:
        ours[return_figures.INCEPTION] = _shown(dict(inception), return_figures.INCEPTION,
                                                returns.index[0], returns.index[-1])

    theirs = {name: _against(returns, pair, ours, periods_per_year, risk_free_rate)
              for name, pair in comparators.items() if name in COMPARATORS}
    rows = []

    for window in starts:
        for statistic in (*OWN, *AGAINST):
            row: dict[str, object] = {"period": window, "statistic": statistic,
                                      "unit": UNITS[statistic],
                                      FUND: _read(ours[window], statistic) if statistic in OWN
                                      else None}

            for name in COMPARATORS:
                row[name] = _read(theirs[name][window], statistic) if name in theirs else None

            rows.append(row)

    return pd.DataFrame(rows, columns=list(COLUMNS))


def _own(returns: pd.Series,
         window: str,
         start: pd.Timestamp | None,
         periods_per_year: int,
         risk_free_rate: float) -> Figures | None:
    """A book's own figures over *window*, from the close before *start*;
    None for a window longer than its returns."""
    if start is None:
        return None

    days = returns.loc[start:]
    path = formulas.path_of(days)
    figures = formulas.headline_figures(days, float(path.iloc[-1] - 1.0), path,
                                        periods_per_year, risk_free_rate)

    return _shown(dict(figures), window, days.index[0], days.index[-1])


def _shown(figures: Figures,
           window: str,
           start: pd.Timestamp,
           end: pd.Timestamp) -> Figures:
    """*figures* with those a window does not give left blank."""
    shown: Figures = {name: figures.get(name) for name in OWN}

    if not return_figures.annualised_over(window, start, end):
        shown["annualised_return"] = None

    if window in NO_SHARPE:
        shown["sharpe_ratio"] = None

    return shown


def _against(returns: pd.Series,
             pair: tuple[pd.Series, pd.Series],
             ours: Mapping[str, Figures | None],
             periods_per_year: int,
             risk_free_rate: float) -> dict[str, Figures | None]:
    """A comparator's own figures over each window, with the fund's against
    it where the two cover the same days."""
    aligned, theirs = pair
    starts = return_figures.window_starts(theirs)
    fund_starts = return_figures.window_starts(returns)
    found: dict[str, Figures | None] = {}

    for window, start in starts.items():
        own = _own(theirs, window, start, periods_per_year, risk_free_rate)
        mine = ours.get(window)
        same = (own is not None and mine is not None and start == fund_starts[window]
                and theirs.index[-1] == returns.index[-1])

        if own is not None:
            active = aligned.loc[start:] - theirs.loc[start:]
            own["tracking_difference"] = (_less(mine["total_return"], own["total_return"])
                                          if same and mine is not None else None)
            own["tracking_error"] = (formulas.tracking_error(active, periods_per_year)
                                     if same else None)

        found[window] = own

    return found


def _read(figures: Figures | None,
          statistic: str) -> float | None:
    return None if figures is None else figures.get(statistic)


def _less(ours: float | None,
          theirs: float | None) -> float | None:
    return ours - theirs if ours is not None and theirs is not None else None
