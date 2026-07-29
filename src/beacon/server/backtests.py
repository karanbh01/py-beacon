# src/beacon/server/backtests.py
"""
Backtest job body and result assembly.

Everything reported here derives from a single canonical series — the
portfolio NAV, rebased to 100 — so the payload is internally consistent by
construction rather than by coincidence. A client that recomputes drawdown
from the level series, or compounds the annual returns, must land back on the
numbers the server sent; if those were computed independently they would drift
apart at the last decimal and nobody would know which to trust.
"""
import pandas as pd

from ..backtest.engine import BacktestEngine
from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator
from ..index.result import IndexResult
from .definitions import build_index_definition
from .jobs import JobBody, ProgressReporter
from .schemas import (
    BacktestMetrics,
    BacktestRequest,
    BacktestRunResult,
    IndexDocument,
    SeriesPayload,
)

# Every series is rebased to this so the portfolio and its benchmark start
# together and can be read off the same axis.
BASE_LEVEL = 100.0


def _rebase(series: pd.Series) -> pd.Series:
    """Rebase a series to BASE_LEVEL at its first observation."""
    if series.empty:
        return series

    first = series.iloc[0]
    if first == 0:
        return series

    return series / first * BASE_LEVEL


def _drawdown(level: pd.Series) -> pd.Series:
    """Drawdown from the running peak, derived from *level* itself."""
    if level.empty:
        return level

    return level / level.cummax() - 1.0


def annual_returns(level: pd.Series) -> dict[str, float]:
    """Calendar-year returns that compound exactly to the total.

    Each year runs from the previous year's closing level to its own, so the
    product of (1 + r) telescopes to ``last / first - 1``. Defining them any
    other way — from the first observation *within* each year, say — leaves a
    gap over each year boundary and the compounded total no longer matches.

    Args:
        level: The level series, indexed by date.

    Returns:
        dict: Year (as a string) -> return for that year.
    """
    if level.empty:
        return {}

    closes = level.groupby(level.index.year).last()
    returns: dict[str, float] = {}
    previous = level.iloc[0]

    for year, close in closes.items():
        returns[str(year)] = float(close / previous - 1.0)
        previous = close

    return returns


def _metrics(result: BacktestResult) -> BacktestMetrics:
    """Headline metrics, read from the library's own summary."""
    summary = result.summary()

    def value(key: str) -> float:
        raw = summary.get(key)
        return 0.0 if raw is None else float(raw)

    return BacktestMetrics(total_return=value("total_return"),
                           annualised_return=value("annualised_return"),
                           volatility=value("volatility"),
                           sharpe_ratio=value("sharpe_ratio"),
                           max_drawdown=value("max_drawdown"),
                           tracking_error=summary.get("tracking_error"),
                           tracking_difference=summary.get("tracking_difference"))


def assemble_result(result: BacktestResult,
                    index_result: IndexResult) -> BacktestRunResult:
    """Build the wire payload from a completed backtest.

    Args:
        result: The finished backtest.
        index_result: The index it tracked, used as the benchmark.

    Returns:
        BacktestRunResult: Level, returns, drawdown, annual returns, benchmark
        and metrics, all derived from the same NAV series.
    """
    level = _rebase(result.portfolio_nav)
    returns = level.pct_change().dropna()

    return BacktestRunResult(
        level=SeriesPayload.from_series(level),
        returns=SeriesPayload.from_series(returns),
        drawdown=SeriesPayload.from_series(_drawdown(level)),
        annual_returns=annual_returns(level),
        benchmark_level=SeriesPayload.from_series(_rebase(index_result.index_levels)),
        metrics=_metrics(result))


def build_backtest_job(document: IndexDocument,
                       fetcher: DataFetcher,
                       request: BacktestRequest) -> JobBody:
    """Build the job body that runs one backtest.

    Returned as a closure rather than run inline: the caller submits it to the
    job registry, which owns scheduling and progress publication.

    Args:
        document: The index definition to calculate and then track.
        fetcher: Data source for both the index and the simulation.
        request: Period, capital and cost settings for this run.

    Returns:
        JobBody: A coroutine function taking a progress reporter.
    """
    async def run(report: ProgressReporter) -> dict[str, object]:
        definition = build_index_definition(document)

        # Both ends are resolved up front. IndexCalculator.run() requires an
        # explicit end_date, so an omitted one becomes the last date the data
        # actually covers rather than an error the client never asked for.
        start = request.start or str(definition.base_date.date())
        end = request.end or str(fetcher.date_range[1].date())

        await report(0.05, "Calculating the index.")
        index_result = IndexCalculator(definition, fetcher).run(
            start_date=start, end_date=end)

        await report(0.5, "Simulating the tracking portfolio.")

        engine = BacktestEngine(start_date=start,
                                end_date=end,
                                initial_capital=request.initial_capital,
                                data_provider=fetcher,
                                target_index_result=index_result,
                                transaction_cost_bps=request.transaction_cost_bps)
        backtest = engine.run()

        await report(0.9, "Assembling results.")
        payload = assemble_result(backtest, index_result)

        await report(1.0, "Complete.")

        return payload.model_dump()

    return run
