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

- The annualised return and the Sharpe ratio are left blank over a window
  too short to annualise them, a year and six months unless set
  (`beacon.analysis.annualising`).
- A window longer than the run has no figures.
"""
# BN-422. Agreed with beacon-ui for its Summary (BU-294). BN-465: the short
# windows' figures follow the rule every annualised figure does, in place of
# this table's own (no Sharpe ratio under 6M, set by Karan on 2026-11-20).
from collections.abc import Mapping

import pandas as pd

from ...assumptions import DEFAULT_MIN_ANNUALISED_RATE_WINDOW, DEFAULT_MIN_ANNUALISED_RATIO_WINDOW
from ..annualising import blanked_over
from . import formulas, return_figures

FUND, INDEX, BENCHMARK = "fund", "index", "benchmark"
COMPARATORS = (INDEX, BENCHMARK)

# Each book's own figures, then the fund's against a comparator, which are
# given under the comparator's column.
OWN = ("total_return", "annualised_return", "volatility", "max_drawdown", "sharpe_ratio")
AGAINST = ("tracking_difference", "tracking_error")

# The figures annualised, which a short window leaves blank.
ANNUALISED = ("annualised_return", "sharpe_ratio")

# What each figure is measured in, so a row says how it reads.
UNITS = {**dict.fromkeys((*OWN, *AGAINST), "fraction"), "sharpe_ratio": "ratio"}

COLUMNS = ("period", "statistic", "unit", FUND, *COMPARATORS)

Figures = dict[str, float | None]

# How short a window an annualised rate and ratio are blank over.
Windows = tuple[str, str]
DEFAULTS: Windows = (DEFAULT_MIN_ANNUALISED_RATE_WINDOW, DEFAULT_MIN_ANNUALISED_RATIO_WINDOW)


def trailing_statistics(returns: pd.Series,
                        periods_per_year: int,
                        risk_free_rate: float,
                        comparators: Mapping[str, tuple[pd.Series, pd.Series]],
                        inception: Mapping[str, float] | None = None,
                        annualising: Windows = DEFAULTS) -> pd.DataFrame:
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
        annualising: The shortest windows an annualised rate and ratio are
            given over.

    Returns:
        pd.DataFrame: Columns ``period``, ``statistic``, ``unit`` (``fraction``,
        or ``ratio`` for the Sharpe ratio), ``fund``, ``index`` and
        ``benchmark``; a comparator the run lacks is blank.
    """
    starts = return_figures.window_starts(returns)
    ours = {window: _own(returns, start, periods_per_year, risk_free_rate, annualising)
            for window, start in starts.items()}

    if inception is not None and ours[return_figures.INCEPTION] is not None:
        ours[return_figures.INCEPTION] = _shown(dict(inception), len(returns), periods_per_year,
                                                annualising)

    theirs = {name: _against(returns, pair, ours, periods_per_year, risk_free_rate,
                             annualising)
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
         start: pd.Timestamp | None,
         periods_per_year: int,
         risk_free_rate: float,
         annualising: Windows) -> Figures | None:
    """A book's own figures over a window from the close before *start*;
    None for a window longer than its returns."""
    if start is None:
        return None

    days = returns.loc[start:]
    path = formulas.path_of(days)
    figures = formulas.headline_figures(days, float(path.iloc[-1] - 1.0), path,
                                        periods_per_year, risk_free_rate)

    return _shown(dict(figures), len(days), periods_per_year, annualising)


def _shown(figures: Figures,
           days: int,
           periods_per_year: int,
           annualising: Windows) -> Figures:
    """*figures* over *days* daily returns, with those annualised over too
    few of them left blank."""
    shown: Figures = {name: figures.get(name) for name in OWN}

    for name in ANNUALISED:
        if blanked_over(name, days, *annualising, periods_per_year) is not None:
            shown[name] = None

    return shown


def _against(returns: pd.Series,
             pair: tuple[pd.Series, pd.Series],
             ours: Mapping[str, Figures | None],
             periods_per_year: int,
             risk_free_rate: float,
             annualising: Windows) -> dict[str, Figures | None]:
    """A comparator's own figures over each window, with the fund's against
    it where the two cover the same days."""
    aligned, theirs = pair
    starts = return_figures.window_starts(theirs)
    fund_starts = return_figures.window_starts(returns)
    found: dict[str, Figures | None] = {}

    for window, start in starts.items():
        own = _own(theirs, start, periods_per_year, risk_free_rate, annualising)
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
