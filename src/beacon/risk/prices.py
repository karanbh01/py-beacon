# src/beacon/risk/prices.py
"""
A risk model, and how a book's risk divides among its names, estimated from
the names' prices.

Each name's daily returns are its prices' changes; a day with no return for
any name is dropped. Below `MINIMUM_OBSERVATIONS` days a covariance over
hundreds of names is noise, so the decompositions give None rather than a
figure of an estimate that means nothing.
"""
# BN-401: moved from the server's weights, risk and optimisation views, which
# now call these.
import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .contribution import RiskContributions, active_risk_contributions, risk_contributions
from .model import RiskModel, estimate_risk_model

logger = logging.getLogger(__name__)

MINIMUM_OBSERVATIONS = 60


def returns_of_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Each name's daily returns from *prices* (names on the columns), with
    the days no name has one dropped."""
    return prices.pct_change().dropna(how="all")


def model_of_prices(prices: pd.DataFrame,
                    **options: Any) -> RiskModel:
    """`estimate_risk_model` over the returns of *prices*, with its
    *options*."""
    return estimate_risk_model(returns_of_prices(prices), **options)


def risk_of_prices(weights: Mapping[str, float],
                   prices: pd.DataFrame,
                   minimum_observations: int = MINIMUM_OBSERVATIONS
                   ) -> RiskContributions | None:
    """How the volatility of a book held at *weights* divides among its
    names, from a risk model of *prices*; None with too few days."""
    returns = returns_of_prices(prices)

    if len(returns) < minimum_observations:
        logger.warning(
            "Only %d return observation(s), fewer than the %d a covariance needs; "
            "no risk decomposition.", len(returns), minimum_observations)

        return None

    return risk_contributions(weights, estimate_risk_model(returns).covariance)


def active_risk_of_prices(weights: Mapping[str, float],
                          benchmark: Mapping[str, float],
                          prices: pd.DataFrame,
                          minimum_observations: int = MINIMUM_OBSERVATIONS
                          ) -> RiskContributions | None:
    """How a book's tracking error against *benchmark* divides among the
    names of either, from a risk model of *prices* (which should cover
    both); None with too few days."""
    returns = returns_of_prices(prices)

    if len(returns) < minimum_observations:
        logger.warning("Too few observations (%d) for an active decomposition.",
                       len(returns))

        return None

    return active_risk_contributions(weights, benchmark,
                                     estimate_risk_model(returns).covariance)
