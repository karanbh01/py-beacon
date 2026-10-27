# src/beacon/analysis/performance/distribution.py
"""
The shape of a run's daily returns: their spread below a target and below
their mean, their skew and tails, how each day's return relates to the days
before, a histogram to draw, and, by resampling, how far its annualised
return and Sharpe ratio could have come out otherwise.
"""
# BN-376.
from dataclasses import dataclass
from itertools import pairwise
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import formulas

# How many serial correlations are given, in days.
SERIAL_LAGS = 5

# The histogram's bins unless asked otherwise.
BINS = 50

# The resampling unless asked otherwise: how many runs, the share of them the
# interval holds, and the seed that makes them the same each time.
SAMPLES, CONFIDENCE, SEED = 1000, 0.95, 0


@dataclass(frozen=True)
class Histogram:
    """The daily returns counted into bins, with the normal distribution of
    the same mean and spread to compare them with.

    Attributes:
        edges: The bins' edges, one more than the bins.
        counts: The returns in each bin.
        normal: The returns a normal distribution would put in each bin.
    """
    edges: list[float]
    counts: list[int]
    normal: list[float]


@dataclass(frozen=True)
class Interval:
    """A figure and the range the resampled runs put it in.

    Attributes:
        value: The figure on the run's own returns.
        low: The lower end of the interval.
        high: The upper end.
        confidence: The share of resampled runs between them.
    """
    value: float
    low: float
    high: float
    confidence: float


def downside_deviation(returns: pd.Series,
                       periods_per_year: int,
                       risk_free_rate: float = 0.0) -> float:
    """The root mean square of the shortfalls below the daily risk-free rate,
    every day counted, annualised: the Sortino ratio's risk."""
    excess = returns - formulas.daily_rate(risk_free_rate, periods_per_year)
    values = np.minimum(excess.to_numpy(dtype=float), 0.0)

    return float(np.sqrt(np.mean(values ** 2) * periods_per_year)) if len(values) else 0.0


def semi_variance(returns: pd.Series,
                  periods_per_year: int) -> float:
    """The mean square of the shortfalls below the mean return, every day
    counted, annualised."""
    values = returns.to_numpy(dtype=float)

    if not len(values):
        return 0.0

    return float(np.mean(np.minimum(values - values.mean(), 0.0) ** 2) * periods_per_year)


def skew(returns: pd.Series) -> float | None:
    """The sample skew, bias-corrected as a spreadsheet's is; None with fewer
    than three returns or no spread."""
    if len(returns) < 3 or not float(returns.std()) > formulas.NEGLIGIBLE_SPREAD:
        return None

    return float(returns.skew())


def excess_kurtosis(returns: pd.Series) -> float | None:
    """The sample kurtosis over a normal distribution's, bias-corrected as a
    spreadsheet's is; None with fewer than four returns or no spread."""
    if len(returns) < 4 or not float(returns.std()) > formulas.NEGLIGIBLE_SPREAD:
        return None

    return float(returns.kurt())


def tail_ratio(returns: pd.Series) -> float | None:
    """The 95th percentile of the returns over the size of the 5th: how much
    larger the right tail is than the left."""
    if returns.empty:
        return None

    low = abs(float(returns.quantile(0.05)))

    return float(abs(float(returns.quantile(0.95))) / low) if low > 0 else None


def serial_correlation(returns: pd.Series,
                       lags: int = SERIAL_LAGS) -> dict[int, float | None]:
    """Each day's return's correlation with the return *lag* days before,
    for lags 1 to *lags*."""
    found: dict[int, float | None] = {}

    for lag in range(1, lags + 1):
        value = returns.autocorr(lag) if len(returns) > lag + 1 else float("nan")
        found[lag] = None if pd.isna(value) else float(value)

    return found


def histogram(returns: pd.Series,
              bins: int = BINS) -> Histogram:
    """The daily returns counted into *bins* bins of equal width."""
    values = returns.to_numpy(dtype=float)

    if not len(values):
        return Histogram(edges=[], counts=[], normal=[])

    counts, edges = np.histogram(values, bins=bins)
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    normal = ([len(values) * (_normal_cdf(high, values.mean(), spread)
                              - _normal_cdf(low, values.mean(), spread))
               for low, high in pairwise(edges)]
              if spread > 0 else [0.0] * bins)

    return Histogram(edges=[float(edge) for edge in edges],
                     counts=[int(count) for count in counts],
                     normal=[float(value) for value in normal])


def bootstrap(returns: pd.Series,
              periods_per_year: int,
              risk_free_rate: float = 0.0,
              samples: int = SAMPLES,
              confidence: float = CONFIDENCE,
              block: int = 1,
              seed: int = SEED) -> dict[str, Interval] | None:
    """How far the annualised return and the Sharpe ratio could have come out
    otherwise: each over *samples* runs of the same length made by drawing
    the run's own days again, in blocks of *block* days to keep any
    dependence between neighbouring days, with the interval holding
    *confidence* of them. The same *seed* draws the same runs. None with
    fewer than two blocks' worth of returns."""
    values = returns.to_numpy(dtype=float)
    count = len(values)

    if count < 2 * block or block < 1:
        return None

    random = np.random.default_rng(seed)
    starts = random.integers(0, count - block + 1, size=(samples, -(-count // block)))
    picks = (starts[:, :, None] + np.arange(block)).reshape(samples, -1)[:, :count]
    drawn = values[picks]
    growth = np.prod(1.0 + drawn, axis=1)
    annualised = growth ** (periods_per_year / count) - 1.0
    excess = drawn - formulas.daily_rate(risk_free_rate, periods_per_year)
    spread = excess.std(axis=1, ddof=1)
    sharpe = np.where(spread > formulas.NEGLIGIBLE_SPREAD,
                      excess.mean(axis=1) / np.where(spread > 0, spread, 1.0)
                      * np.sqrt(periods_per_year), 0.0)
    tail = (1.0 - confidence) / 2.0

    def interval(own: float,
                 resampled: np.ndarray) -> Interval:
        low, high = np.quantile(resampled, [tail, 1.0 - tail])
        return Interval(value=own, low=float(low), high=float(high), confidence=confidence)

    total = formulas.compounded_return(returns)

    return {"annualised_return": interval(formulas.annualised_return(total, count,
                                                                     periods_per_year),
                                          annualised),
            "sharpe_ratio": interval(formulas.sharpe_ratio(returns, periods_per_year,
                                                           risk_free_rate), sharpe)}


def _normal_cdf(value: float,
                mean: float,
                spread: float) -> float:
    """The normal distribution's cumulative probability."""
    return NormalDist(mean, spread).cdf(value)
