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
    at *lagged* (its weights at the close before), by day and part.

    Worked out a stretch between rebalances at a time, over which the
    exposures are fixed, as matrix products over its days.
    """
    # BN-416: day by day with pandas this took 5 s at 6,000 names; the sums
    # now run in another order, so the parts differ in the last places.
    found = factors.factor_returns()
    days = lagged.index.intersection(found.index)
    rebalances = pd.DatetimeIndex(factors.dates)
    stretches: dict[int, list[pd.Timestamp]] = {}

    for day in days:
        at = as_of_position(rebalances, day - pd.Timedelta(days=1))

        if at is not None:
            stretches.setdefault(at, []).append(day)

    names = list(lagged.columns)
    tables = []

    for at, stretch in sorted(stretches.items()):
        exposures = factors.exposures(rebalances[at]).reindex(names).fillna(0.0)
        loadings = exposures[factors.names].to_numpy(dtype=float)
        held = lagged.loc[stretch, names].to_numpy(dtype=float)
        earned = name_returns.reindex(index=stretch, columns=names).fillna(0.0).to_numpy(
            dtype=float)
        market = found.loc[stretch, MARKET_FACTOR].to_numpy(dtype=float)
        paid = found.loc[stretch, factors.names].to_numpy(dtype=float)
        own = earned - (market[:, None] + paid @ loadings.T)
        table = pd.DataFrame(held @ loadings * paid, index=pd.DatetimeIndex(stretch),
                             columns=factors.names)
        table.insert(0, MARKET_FACTOR, held.sum(axis=1) * market)
        table[SPECIFIC] = (held * own).sum(axis=1)
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    return pd.concat(tables)


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
