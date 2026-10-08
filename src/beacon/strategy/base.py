# src/beacon/strategy/base.py
"""
What every strategy shares: what it may read at a rebalance, and the returns
and covariance it measures risk with.

The covariance is re-estimated at each rebalance from the names' trailing
daily returns in the book's currency (a year by default), shrunk toward
constant correlation. A name with too few returns to measure is left out of
it; each strategy says what it does with such a name.
"""
# BN-269 and BN-280, phase 7 of decisions/0006.
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..optimise.solver import covariance_matrix
from ..risk.model import CONSTANT_CORRELATION, RiskModel, estimate_risk_model

# Trading days of returns the covariance is estimated over, and the fewest a
# name needs to be measured at all.
DEFAULT_LOOKBACK_DAYS = 252
MINIMUM_OBSERVATIONS = 126


@dataclass(frozen=True)
class StrategyContext:
    """What a strategy may read at a rebalance.

    Attributes:
        fetcher: The run's data.
        currency: The book's currency, which returns and money fields are in.
    """
    fetcher: DataFetcher
    currency: str


def trailing_returns(names: list[str],
                     date: pd.Timestamp,
                     context: StrategyContext,
                     lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """Daily returns in the book's currency over the trading days before and
    including *date*, one column per name."""
    start = date - pd.Timedelta(days=int(lookback_days * 1.6) + 10)
    prices = context.fetcher.fetch_prices(names, start.strftime("%Y-%m-%d"),
                                          date.strftime("%Y-%m-%d"),
                                          currency=context.currency)

    if prices.empty:
        return pd.DataFrame()

    # A missed bar carries the last close rather than ending a name's history.
    return prices.ffill().pct_change(fill_method=None).iloc[1:].tail(lookback_days)


def measured(names: list[str],
             returns: pd.DataFrame,
             minimum_observations: int = MINIMUM_OBSERVATIONS) -> list[str]:
    """The names with enough returns to estimate their risk, in order."""
    return [name for name in names
            if name in returns and returns[name].count() >= minimum_observations]


def estimated_risk(returns: pd.DataFrame,
                   names: list[str]) -> RiskModel:
    """The shrunk covariance of *names*, over the days all have returns."""
    return estimate_risk_model(returns[names].dropna(), target=CONSTANT_CORRELATION)


def covariance_of(risk: RiskModel,
                  names: list[str]) -> np.ndarray:
    """The annualised covariance matrix, aligned to *names*."""
    return np.asarray(covariance_matrix(risk, names), dtype=float)
