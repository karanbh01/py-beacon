# src/beacon/analysis/attribution/model.py
"""
Factor attribution: how much of a book's return its factor exposures explain,
from the run's factors (`beacon.analysis.risk.factors`).

**Holdings-based.** Each day every name's return is the market's, plus its
exposures at the last rebalance before the day times the factor returns, plus
what is its own:

    r_i = f_market + sum_k z_ik f_k + e_i

with ``e_i`` defined as what is left, so the split is exact. A book, holding
``w`` at the close before, earns ``(sum w) f_market`` from the market,
``X_k f_k`` from each factor (``X_k = sum w_i z_ik``) and ``sum w_i e_i`` of
its own: the specific return.

**Returns-based.** The book's daily return over the risk-free rate,
regressed on the factor returns over the window (``alpha + sum beta_k f_k +
e``), splits into the risk-free return, the alpha, each beta times its
factor's return, and the regression's residual.

On the active return the same holds on the active weights, or the active
return's regression.
"""
# BN-395. Agreed with Karan on 2026-10-28: both methods.
import pandas as pd

from ...data.base import as_of_position
from ...risk.model import MARKET_FACTOR
from ..performance import formulas
from ..risk.factors import ALPHA, FactorAnalytics, regressed

SPECIFIC = "specific"
RISK_FREE = "risk_free"
RESIDUAL = "residual"
HOLDINGS = "holdings"
RETURNS = "returns"
METHODS = (HOLDINGS, RETURNS)


def holdings_parts(factors: FactorAnalytics,
                   lagged: pd.DataFrame,
                   name_returns: pd.DataFrame) -> pd.DataFrame:
    """Each day's market, factor and specific contributions of a book held
    at *lagged* (its weights at the close before), by day and part."""
    found = factors.factor_returns()
    days = lagged.index.intersection(found.index)
    rebalances = pd.DatetimeIndex(factors.dates)
    rows = {}

    for day in days:
        at = as_of_position(rebalances, day - pd.Timedelta(days=1))

        if at is None:
            continue

        exposures = factors.exposures(rebalances[at])
        held = lagged.loc[day]
        held = held[held != 0.0]
        names = held.index
        z = exposures.reindex(names).fillna(0.0)
        returns = name_returns.loc[day].reindex(names).fillna(0.0)
        factor_returns = found.loc[day]
        explained = factor_returns[MARKET_FACTOR] + z @ factor_returns[factors.names]
        own = returns - explained
        X = z.T @ held
        rows[day] = {MARKET_FACTOR: float(held.sum() * factor_returns[MARKET_FACTOR]),
                     **{name: float(X[name] * factor_returns[name]) for name in factors.names},
                     SPECIFIC: float((held * own).sum())}

    return pd.DataFrame.from_dict(rows, orient="index")


def returns_parts(target: pd.Series,
                  factor_returns: pd.DataFrame,
                  rate: float,
                  periods_per_year: int) -> pd.DataFrame:
    """Each day's risk-free, alpha, factor and residual parts of *target*
    (a book's daily return), from its regression on the factor returns."""
    days = target.index.intersection(factor_returns.dropna().index)
    excess = target.loc[days] - rate
    fit = regressed(excess, factor_returns.loc[days], periods_per_year)
    betas = fit.betas
    parts = factor_returns.loc[days].mul(betas, axis=1)
    parts.insert(0, ALPHA, fit.coefficients.loc[ALPHA, "coefficient"])
    parts.insert(0, RISK_FREE, rate)
    parts[RESIDUAL] = target.loc[days] - parts.sum(axis=1)

    return parts


def daily_rate(risk_free_rate: float,
               periods_per_year: int) -> float:
    """The risk-free rate a day."""
    return formulas.daily_rate(risk_free_rate, periods_per_year)
