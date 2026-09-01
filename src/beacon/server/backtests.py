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

from ..analysis.relative import align_on_common_window, relative_metrics
from ..backtest.engine import BacktestEngine
from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator
from ..index.result import IndexResult
from .benchmarks import resolve_benchmark
from .definitions import build_index_definition
from .jobs import JobBody, ProgressReporter
from .schemas import (
    BacktestMetrics,
    BacktestRequest,
    BacktestResultSummary,
    BacktestRunResult,
    BenchmarkRef,
    IndexDocument,
    RebalanceSnapshot,
    RelativeMetricsPayload,
    SeriesPayload,
)
from .store import DocumentStore

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
                    index_result: IndexResult,
                    benchmark: RelativeMetricsPayload | None = None,
                    cap: float | None = None) -> BacktestRunResult:
    """Build the wire payload from a completed backtest.

    Args:
        result: The finished backtest.
        index_result: The index it tracked, reported alongside as the
            replication reference.
        benchmark: Optional comparison against an external benchmark.

    Returns:
        BacktestRunResult: Level, returns, drawdown, annual returns, the
        tracked index and metrics, all derived from the same NAV series.
    """
    level = _rebase(result.trading_nav)
    returns = level.pct_change().dropna()

    return BacktestRunResult(
        level=SeriesPayload.from_series(level),
        returns=SeriesPayload.from_series(returns),
        drawdown=SeriesPayload.from_series(_drawdown(level)),
        annual_returns=annual_returns(level),
        index_level=SeriesPayload.from_series(_rebase(index_result.index_levels)),
        metrics=_metrics(result),
        benchmark=benchmark,
        rebalances=rebalance_snapshots(index_result, cap),
        total_costs=_total_costs(result),
        initial_capital=result.portfolio.initial_capital)


def rebalance_snapshots(index_result: IndexResult,
                        cap: float | None = None) -> list[RebalanceSnapshot]:
    """Composition at each rebalance, in date order.

    Carries the uncapped weights alongside the applied ones. On an uncapped
    index the two are identical and the duplication costs a little space; on a
    capped one the difference is the only record of what the cap did, and it
    cannot be recovered from the applied weights afterwards.

    The *cap itself* comes from the definition rather than from the cap report,
    because the calculator only files a report on dates where the cap actually
    bound. "A 20% cap applies and nothing reached it" and "no cap applies" are
    different statements about a methodology, and a client asking what the
    rules are should get the same answer on both dates.

    Args:
        index_result: The calculated index.
        cap: The definition's maximum constituent weight, if it has one.
    """
    snapshots = []

    for date in sorted(index_result.weight_snapshots):
        weights = index_result.weight_snapshots[date]
        report = index_result.cap_reports.get(date)

        announced = index_result.announcement_dates.get(date)

        snapshots.append(RebalanceSnapshot(
            date=date.strftime("%Y-%m-%d"),
            announced=announced.strftime("%Y-%m-%d") if announced else None,
            weights=dict(weights),
            uncapped_weights=dict(report.uncapped_weights) if report and
            report.uncapped_weights else dict(weights),
            capped=sorted(report.capped) if report else [],
            cap=cap,
            redistributed=report.redistributed if report else 0.0))

    return snapshots


def _total_costs(result: BacktestResult) -> float:
    """Transaction costs paid across the run."""
    return float(sum(transaction.transaction_cost
                     for transaction in result.portfolio.transactions))


def compare_against_benchmark(nav: pd.Series,
                              reference: BenchmarkRef,
                              fetcher: DataFetcher,
                              index_store: DocumentStore,
                              start: str,
                              end: str) -> RelativeMetricsPayload:
    """Resolve a benchmark and measure the portfolio against it.

    The benchmark series is rebased on the *aligned* window rather than its own
    full history, so both lines start at 100 on the same date and can be read
    off one axis. Rebasing before alignment would leave the benchmark starting
    somewhere other than 100 once trimmed.

    Args:
        nav: Portfolio NAV series.
        reference: What to compare against.
        fetcher: Data source.
        index_store: Where stored index definitions live.
        start: Window start, YYYY-MM-DD.
        end: Window end, YYYY-MM-DD.

    Returns:
        RelativeMetricsPayload: The comparison and the rebased benchmark.
    """
    levels = resolve_benchmark(reference, fetcher, index_store, start, end)
    metrics = relative_metrics(nav, levels)

    _, aligned_benchmark = align_on_common_window(nav, levels)

    return RelativeMetricsPayload(
        reference=reference,
        observations=metrics.observations,
        start=metrics.start,
        end=metrics.end,
        total_return=metrics.total_return,
        benchmark_return=metrics.benchmark_return,
        excess_return=metrics.excess_return,
        tracking_error=metrics.tracking_error,
        correlation=metrics.correlation,
        beta=metrics.beta,
        level=SeriesPayload.from_series(_rebase(aligned_benchmark)))


def build_backtest_job(document: IndexDocument,
                       fetcher: DataFetcher,
                       request: BacktestRequest,
                       index_store: DocumentStore,
                       record_store: DocumentStore | None = None) -> JobBody:
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
                                index_result=index_result,
                                transaction_cost_bps=request.transaction_cost_bps)
        backtest = engine.run()

        comparison = None
        if request.benchmark is not None:
            await report(0.8, f"Comparing against benchmark '{request.benchmark.id}'.")
            comparison = compare_against_benchmark(
                backtest.trading_nav, request.benchmark, fetcher, index_store,
                start, end)

        await report(0.9, "Assembling results.")
        payload = assemble_result(backtest, index_result, comparison,
                                  cap=definition.max_constituent_weight)

        # The record is captured here or never: the library BacktestResult
        # exists only inside this job, and the run payload the job returns is
        # a derived view of it, not the books (BN-158). Latest run wins,
        # matching latest_result semantics.
        if record_store is not None:
            record = BacktestResultSummary.from_result(backtest)
            record_store.write(document.id, record.model_dump(mode="json"))

        await report(1.0, "Complete.")

        return payload.model_dump()

    return run
