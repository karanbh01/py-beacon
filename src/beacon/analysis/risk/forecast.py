# src/beacon/analysis/risk/forecast.py
"""
How well each rebalance's expected risk forecast what followed.

At each rebalance with a model (`RiskOverTime.dates`), the expected
volatility and tracking error (`RiskOverTime.expected`) are set beside the
ones measured over the days after it, up to and including the next
rebalance (or the run's last day): the standard deviation of the daily
returns (or of the fund's less the comparator's) over those days, a year.
Their ratio is above 1 where the model under-forecast.

The **bias statistic** pools the days: each day's return is divided by the
daily risk forecast in force at the close before (the expected figure over
the root of the days in a year), and the bias is the standard deviation of
those standardised returns. It is 1 when the forecasts were right on
average, above 1 when the risk was under-forecast. With few days it is noisy:
its standard error is about one over the root of twice the days.
"""
# BN-421: the check that BN-420 will revisit the risk model by.
import math
from typing import Any

import numpy as np
import pandas as pd

from ..data import RunData
from ..performance.comparisons import aligned

FUND = "fund"

# The columns of the check, a row a rebalance.
COLUMNS = ("days", "expected_volatility", "realised_volatility", "volatility_ratio",
           "expected_tracking_error", "realised_tracking_error", "tracking_error_ratio")


def book_returns(run: RunData,
                 book: str) -> pd.Series:
    """A book's daily returns: the fund's, or a comparator's from its levels."""
    if book == FUND:
        return run.returns.dropna().astype(float)

    if book not in run.comparators:
        raise KeyError(f"The run has no {book}: it has "
                       f"{', '.join(run.comparators) or 'no comparators'}.")

    _, theirs = aligned(run.returns, run.comparators[book])

    return theirs.dropna().astype(float)


def active_returns(run: RunData,
                   book: str,
                   against: str) -> pd.Series:
    """A book's daily returns less *against*'s, on the days both have."""
    ours, theirs = book_returns(run, book), book_returns(run, against)
    days = ours.index.intersection(theirs.index)

    return ours.loc[days] - theirs.loc[days]


def forecast_check(expected: dict[pd.Timestamp, dict[str, float]],
                   returns: pd.Series,
                   active: pd.Series | None,
                   periods_per_year: int) -> pd.DataFrame:
    """Each rebalance's expected volatility (and tracking error, when
    *active* is given) beside the ones measured until the next rebalance.

    Args:
        expected: Each rebalance's expected figures, as
            `RiskOverTime.expected` gives them.
        returns: The book's daily returns.
        active: Its daily returns less the comparator's; None for no
            tracking error.
        periods_per_year: The days in a year the figures are annualised by.
    """
    rebalances = sorted(expected)
    rows = {}

    for at, day in enumerate(rebalances):
        end = rebalances[at + 1] if at + 1 < len(rebalances) else None
        held = _after(returns, day, end)
        row: dict[str, Any] = {"days": len(held)}
        row.update(_compared(expected[day]["volatility"], held, periods_per_year,
                             "volatility"))

        if active is not None:
            row.update(_compared(expected[day]["tracking_error"], _after(active, day, end),
                                 periods_per_year, "tracking_error"))

        rows[day] = row

    table = pd.DataFrame.from_dict(rows, orient="index", columns=list(COLUMNS))

    return table if active is not None else table.drop(columns=list(COLUMNS[4:]))


def bias_statistic(forecasts: pd.Series,
                   returns: pd.Series,
                   periods_per_year: int) -> float | None:
    """The standard deviation of the daily returns after the first forecast,
    each over the daily forecast in force at the close before; None with
    fewer than two such days or a forecast of 0."""
    if forecasts.empty:
        return None

    days = returns.index[returns.index > forecasts.index[0]]
    # The forecast made at a rebalance's close is in force from the day after.
    standing = forecasts.reindex(forecasts.index.union(days)).ffill().shift(1).reindex(days)
    daily = standing / math.sqrt(periods_per_year)

    if len(days) < 2 or not (daily > 0.0).all():
        return None

    return float((returns.loc[days] / daily).std())


def _after(returns: pd.Series,
           start: pd.Timestamp,
           end: pd.Timestamp | None) -> pd.Series:
    """*returns* after *start*, up to and including *end*."""
    window = returns[returns.index > start]

    return window if end is None else window[window.index <= end]


def _compared(forecast: float,
              held: pd.Series,
              periods_per_year: int,
              name: str) -> dict[str, float | None]:
    """The forecast, the figure measured over *held*, and their ratio."""
    realised = (float(held.std() * np.sqrt(periods_per_year)) if len(held) > 1 else None)
    ratio = realised / forecast if realised is not None and forecast > 0.0 else None

    return {f"expected_{name}": forecast, f"realised_{name}": realised, f"{name}_ratio": ratio}
