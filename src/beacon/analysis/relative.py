# src/beacon/analysis/relative.py
"""
Performance of one level series relative to another.

`BacktestResult.get_tracking_error` compares a portfolio against the index it
was built to track, which is the right question for a replication mandate. It
is not the only question: a portfolio is also measured against benchmarks it
was never trying to replicate. That comparison needs a general pair of level
series rather than a target `IndexResult`, which is what lives here.

Alignment is the part that matters. Two series rarely cover exactly the same
dates — a benchmark may start earlier, or trade on a holiday the portfolio does
not — and silently comparing mismatched observations produces numbers that look
fine and mean nothing.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252

# Two aligned observations give one return, which is not enough for a standard
# deviation. Three gives two returns: the minimum for a tracking error that is
# not simply zero.
MINIMUM_ALIGNED_OBSERVATIONS = 3

# A return series whose standard deviation is below this has no variation worth
# speaking of — 1e-12 is a hundred-millionth of a basis point per period. The
# threshold is absolute because returns are dimensionless fractions.
#
# It has to be a threshold rather than a check against exactly zero. A perfectly
# smooth geometric series has constant returns in exact arithmetic but a
# standard deviation of ~1e-17 in floating point, and beta computed from that is
# a ratio of one rounding error to another: it varies wildly with the level the
# series happens to be quoted at, which is nonsense for a scale-invariant
# quantity.
NEGLIGIBLE_RETURN_STD = 1e-12


@dataclass(frozen=True)
class RelativeMetrics:
    """How one level series performed against another.

    Attributes:
        observations: Aligned dates the comparison used, which may be fewer
            than either input carried.
        start: First aligned date, ISO 8601.
        end: Last aligned date, ISO 8601.
        total_return: Portfolio return over the aligned window.
        benchmark_return: Benchmark return over the same window.
        excess_return: Portfolio minus benchmark. Also the tracking
            difference — one name for the arithmetic difference of two total
            returns, kept under both because each is idiomatic in a different
            context.
        tracking_error: Annualised standard deviation of the per-period return
            differences.
        correlation: Correlation of the two return series.
        beta: Sensitivity of portfolio returns to benchmark returns —
            covariance over benchmark variance.
    """
    observations: int
    start: str
    end: str
    total_return: float
    benchmark_return: float
    excess_return: float
    tracking_error: float
    correlation: float
    beta: float


def align_on_common_window(portfolio: pd.Series,
                           benchmark: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Restrict two level series to the dates they share.

    Args:
        portfolio: Level series of the portfolio.
        benchmark: Level series of the benchmark.

    Returns:
        tuple: The two series over their shared dates, in ascending order.

    Raises:
        CalculationError: If either series is empty, or the two share fewer
            than MINIMUM_ALIGNED_OBSERVATIONS dates. Comparing series that
            barely overlap would return a number rather than an error, and
            nothing downstream would reveal that it rests on two data points.
    """
    if portfolio.empty or benchmark.empty:
        raise CalculationError(
            "BenchmarkAlignment",
            "cannot compare against a benchmark when either series is empty.")

    common = portfolio.index.intersection(benchmark.index).sort_values()

    if len(common) < MINIMUM_ALIGNED_OBSERVATIONS:
        raise CalculationError(
            "BenchmarkAlignment",
            f"the portfolio and benchmark share only {len(common)} date(s); at "
            f"least {MINIMUM_ALIGNED_OBSERVATIONS} are needed. The portfolio "
            f"covers {_describe_span(portfolio)} and the benchmark covers "
            f"{_describe_span(benchmark)}.")

    dropped = len(portfolio) - len(common)
    if dropped > 0:
        logger.info(
            f"Aligned to {len(common)} common dates, dropping {dropped} "
            f"portfolio observation(s) the benchmark does not cover.")

    return portfolio.loc[common], benchmark.loc[common]


def relative_metrics(portfolio: pd.Series,
                     benchmark: pd.Series,
                     periods_per_year: int = PERIODS_PER_YEAR) -> RelativeMetrics:
    """Compare a portfolio level series against a benchmark's.

    Args:
        portfolio: Level series of the portfolio. Need not be rebased —
            returns are scale-invariant, so only the shape matters.
        benchmark: Level series of the benchmark.
        periods_per_year: Annualisation factor for the tracking error.

    Returns:
        RelativeMetrics: The comparison, computed over the shared window only.

    Raises:
        CalculationError: If the series cannot be aligned, or a level of zero
            makes a return undefined.
    """
    aligned_portfolio, aligned_benchmark = align_on_common_window(portfolio, benchmark)

    portfolio_returns = aligned_portfolio.pct_change().dropna()
    benchmark_returns = aligned_benchmark.pct_change().dropna()

    total = _total_return(aligned_portfolio, "portfolio")
    benchmark_total = _total_return(aligned_benchmark, "benchmark")

    differences = portfolio_returns - benchmark_returns
    tracking_error = float(differences.std(ddof=1) * np.sqrt(periods_per_year))

    return RelativeMetrics(
        observations=len(aligned_portfolio),
        start=aligned_portfolio.index[0].isoformat(),
        end=aligned_portfolio.index[-1].isoformat(),
        total_return=total,
        benchmark_return=benchmark_total,
        excess_return=total - benchmark_total,
        tracking_error=tracking_error,
        correlation=_correlation(portfolio_returns, benchmark_returns),
        beta=_beta(portfolio_returns, benchmark_returns))


def _total_return(levels: pd.Series,
                  label: str) -> float:
    """Total return of a level series over its full span."""
    first = float(levels.iloc[0])

    if first == 0.0:
        raise CalculationError(
            "BenchmarkComparison",
            f"the {label} series starts at zero, so its return is undefined.")

    return float(levels.iloc[-1]) / first - 1.0


def _varies(returns: pd.Series) -> bool:
    """Whether a return series has variation beyond floating-point noise."""
    if len(returns) < 2:
        return False

    return float(returns.std(ddof=1)) > NEGLIGIBLE_RETURN_STD


def _correlation(portfolio_returns: pd.Series,
                 benchmark_returns: pd.Series) -> float:
    """Correlation of two return series, 0.0 when either barely varies.

    A series with no variation has nothing to correlate, which pandas reports
    as NaN. Zero is the honest summary — no relationship is measurable — and it
    keeps NaN out of a JSON payload.
    """
    if not _varies(portfolio_returns) or not _varies(benchmark_returns):
        return 0.0

    value = portfolio_returns.corr(benchmark_returns)

    return 0.0 if pd.isna(value) else float(value)


def _beta(portfolio_returns: pd.Series,
          benchmark_returns: pd.Series) -> float:
    """Sensitivity of portfolio returns to benchmark returns.

    Zero when the benchmark barely varies: with no benchmark movement there is
    nothing for the portfolio to be sensitive to, and the ratio degenerates
    into one rounding error divided by another.
    """
    if not _varies(benchmark_returns):
        return 0.0

    covariance = float(portfolio_returns.cov(benchmark_returns))

    return covariance / float(benchmark_returns.var(ddof=1))


def _describe_span(series: pd.Series) -> str:
    """Describe a series' date span for an error message."""
    return f"{series.index[0].date()} to {series.index[-1].date()}"
