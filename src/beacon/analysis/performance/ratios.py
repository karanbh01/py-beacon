# src/beacon/analysis/performance/ratios.py
"""
Risk-adjusted ratios beyond the Sharpe ratio: return over the downside
(Sortino, Kappa 3, upside potential, Omega), over drawdowns (Calmar, Martin,
Sterling, Burke) and over losing months (gain to pain), and the probability
that a Sharpe ratio is real (probabilistic and deflated), with the Sharpe
ratio corrected for serial correlation (Lo).

Excess returns are over the daily rate that compounds to the risk-free rate,
as the Sharpe ratio's are. A ratio over a risk that is not there (no day
below the target, no drawdown) is None.
"""
# BN-375. The forms chosen with Karan: Sortino over all days with the
# risk-free rate as its target, Calmar over the window measured, Omega at the
# risk-free rate, gain to pain on months, the probabilistic Sharpe ratio
# against 0 unless told otherwise.
import math
from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import formulas

# How many of the deepest drawdowns Sterling averages.
STERLING_DRAWDOWNS = 3

# The serial correlations Lo's correction counts, in days.
LO_LAGS = 10

# The Euler-Mascheroni constant, for the largest Sharpe ratio expected by luck.
_EULER = 0.5772156649015329


def excess_returns(returns: pd.Series,
                   periods_per_year: int,
                   risk_free_rate: float) -> pd.Series:
    """*returns* less the daily rate that compounds to *risk_free_rate*."""
    return returns - formulas.daily_rate(risk_free_rate, periods_per_year)


def downside_deviation(excess: pd.Series) -> float:
    """The root mean square of the shortfalls below 0, over every day."""
    return float(np.sqrt(np.mean(np.minimum(excess.to_numpy(dtype=float), 0.0) ** 2)))


def sortino(excess: pd.Series,
            periods_per_year: int) -> float | None:
    """The mean excess return over the downside deviation, annualised."""
    downside = downside_deviation(excess) if len(excess) else 0.0

    if not downside > formulas.NEGLIGIBLE_SPREAD:
        return None

    return float(excess.mean() / downside * np.sqrt(periods_per_year))


def kappa_3(excess: pd.Series) -> float | None:
    """The mean daily excess return over the cube root of the third lower
    moment: Sortino with the shortfalls cubed, which weighs the larger ones
    more. Daily: a cube root does not annualise by a square root."""
    if excess.empty:
        return None

    moment = float(np.mean(np.maximum(-excess.to_numpy(dtype=float), 0.0) ** 3))

    if not moment ** (1 / 3) > formulas.NEGLIGIBLE_SPREAD:
        return None

    return float(excess.mean() / moment ** (1 / 3))


def upside_potential(excess: pd.Series) -> float | None:
    """The mean gain above 0 over the downside deviation."""
    downside = downside_deviation(excess) if len(excess) else 0.0

    if not downside > formulas.NEGLIGIBLE_SPREAD:
        return None

    return float(np.mean(np.maximum(excess.to_numpy(dtype=float), 0.0)) / downside)


def omega(excess: pd.Series) -> float | None:
    """The sum of the gains above 0 over the sum of the shortfalls below it."""
    values = excess.to_numpy(dtype=float)
    losses = float(np.sum(np.maximum(-values, 0.0)))

    return float(np.sum(np.maximum(values, 0.0)) / losses) if losses > 0 else None


def gain_to_pain(monthly: pd.Series) -> float | None:
    """The sum of the monthly returns over the size of the losing months'
    sum."""
    pain = float(monthly[monthly < 0].sum())

    return float(monthly.sum() / abs(pain)) if pain < 0 else None


def calmar(annualised: float,
           deepest: float) -> float | None:
    """The annualised return over the size of the deepest drawdown."""
    return float(annualised / abs(deepest)) if deepest < 0 else None


def martin(excess: pd.Series,
           ulcer: float,
           periods_per_year: int) -> float | None:
    """The annualised mean excess return over the ulcer index."""
    return float(excess.mean() * periods_per_year / ulcer) if ulcer > 0 else None


def sterling(annualised: float,
             depths: Sequence[float]) -> float | None:
    """The annualised return over the mean size of the deepest drawdowns
    (`STERLING_DRAWDOWNS` of them)."""
    deepest = sorted(depths)[:STERLING_DRAWDOWNS]

    return float(annualised / abs(float(np.mean(deepest)))) if deepest else None


def burke(annualised: float,
          depths: Sequence[float]) -> float | None:
    """The annualised return over the root of the sum of every drawdown's
    depth squared."""
    total = float(np.sqrt(np.sum(np.square(depths)))) if depths else 0.0

    return float(annualised / total) if total > 0 else None


def moments(excess: pd.Series) -> tuple[float, float]:
    """The skew and kurtosis (not excess kurtosis) of the daily excess
    returns, from their population moments."""
    values = excess.to_numpy(dtype=float)
    centred = values - values.mean()
    spread = float(np.sqrt(np.mean(centred ** 2)))

    if not spread > formulas.NEGLIGIBLE_SPREAD:
        return 0.0, 3.0

    return (float(np.mean(centred ** 3) / spread ** 3), float(np.mean(centred ** 4) / spread ** 4))


def probabilistic_sharpe(excess: pd.Series,
                         periods_per_year: int,
                         benchmark: float = 0.0) -> float | None:
    """The probability that the true Sharpe ratio is above *benchmark* (a
    year), allowing for the skew and fat tails of the daily returns (Bailey
    and Lopez de Prado). None with fewer than three returns or no spread."""
    if len(excess) < 3:
        return None

    spread = float(excess.std())

    if not spread > formulas.NEGLIGIBLE_SPREAD:
        return None

    daily = float(excess.mean()) / spread
    skew, kurtosis = moments(excess)
    noise = 1.0 - skew * daily + (kurtosis - 1.0) / 4.0 * daily ** 2

    if not noise > 0:
        return None

    target = benchmark / math.sqrt(periods_per_year)

    return NormalDist().cdf((daily - target) * math.sqrt(len(excess) - 1) / math.sqrt(noise))


def expected_best_sharpe(trials: int,
                         variance: float) -> float:
    """The largest Sharpe ratio (in the units of *variance*'s) expected from
    *trials* independent strategies with no skill, whose Sharpe ratios vary
    by *variance* (Bailey and Lopez de Prado)."""
    normal = NormalDist()

    return math.sqrt(variance) * ((1.0 - _EULER) * normal.inv_cdf(1.0 - 1.0 / trials)
                                  + _EULER * normal.inv_cdf(1.0 - 1.0 / (trials * math.e)))


def deflated_sharpe(excess: pd.Series,
                    periods_per_year: int,
                    trials: int | None,
                    trial_variance: float | None) -> float | None:
    """The probabilistic Sharpe ratio against the best Sharpe ratio *trials*
    strategies would show by luck, their annual Sharpe ratios varying by
    *trial_variance*. None unless both are given and there were at least two
    trials."""
    if trials is None or trial_variance is None or trials < 2 or trial_variance <= 0:
        return None

    return probabilistic_sharpe(excess, periods_per_year,
                                expected_best_sharpe(trials, trial_variance))


def lo_sharpe(excess: pd.Series,
              periods_per_year: int,
              lags: int = LO_LAGS) -> float | None:
    """The Sharpe ratio a year corrected for the serial correlation of the
    daily returns (Lo, 2002), which annualising by the square root of the
    days assumes away. Correlations up to *lags* days are counted."""
    spread = float(excess.std()) if len(excess) > 1 else 0.0

    if not spread > formulas.NEGLIGIBLE_SPREAD or len(excess) <= lags + 1:
        return None

    daily = float(excess.mean()) / spread
    count = periods_per_year
    linked = sum((count - lag) * float(excess.autocorr(lag)) for lag in range(1, lags + 1))
    scale = count + 2.0 * linked

    return float(daily * count / math.sqrt(scale)) if scale > 0 else None
