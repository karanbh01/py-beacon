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
from ..backtest.main import Backtest
from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher
from ..index.constructor import IndexDefinition
from ..index.derived import AnyIndexDefinition, OptimisedIndexDefinition
from ..index.result import IndexResult
from .benchmarks import resolve_benchmark
from .definitions import build_definition
from .jobs import JobBody, ProgressReporter
from .schemas import (
    BacktestMetrics,
    BacktestRequest,
    BacktestResultSummary,
    BacktestRunResult,
    BenchmarkRef,
    IndexDocument,
    RelativeMetricsPayload,
    SeriesPayload,
    rebalance_snapshots,
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


def target_cap(definition: AnyIndexDefinition) -> float | None:
    """The cap that applies to the run's `index.target` book, if any.

    Read off the built definition rather than the document because an optimised
    document has no pipeline of its own to read one from, while its target book
    is its *parent's* calculation and the parent's cap is the one that shaped
    those weights. A chain whose immediate parent is itself derived has no cap
    at that level, and says so.

    Args:
        definition: The definition the run calculated.
    """
    source = (definition.source
              if isinstance(definition, OptimisedIndexDefinition)
              else definition)

    return (source.max_constituent_weight
            if isinstance(source, IndexDefinition) else None)


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
        # Either face (BN-168): a rule pipeline, or a derivation whose source
        # is resolved through the store — recursively for a chain. The rest of
        # the job is identical, which is the point: an optimised index is
        # backtested through the same endpoint as any other.
        definition = build_definition(document, index_store)

        # Both ends are resolved up front. The calculation requires an
        # explicit end date, so an omitted one becomes the last date the data
        # actually covers rather than an error the client never asked for.
        start = request.start or str(definition.base_date.date())
        end = request.end or str(fetcher.date_range[1].date())

        # One call covers both stages since BN-161: the front door
        # fingerprints the calculation, reuses a cached index result when the
        # store allows it, and hands the schedule to the engine. The two
        # stage messages collapse into one; the reporter and the later
        # milestones stay.
        await report(0.05, "Calculating the index and simulating the "
                           "tracking portfolio.")

        backtest = Backtest(initial_capital=request.initial_capital,
                            transaction_cost_bps=request.transaction_cost_bps,
                            data_provider=fetcher).run(definition,
                                                       start=start,
                                                       end=end)

        # The calculation the run tracked, still needed for the payload's
        # index level and rebalance snapshots. The *tracked* book, not the
        # target one: on an optimised index those differ, and the replication
        # reference a client charts against the NAV has to be the book the
        # engine actually traded toward. A definition-driven run always
        # carries one (BN-164).
        book = backtest.index.tracked
        assert book is not None and book.source is not None
        index_result = book.source

        comparison = None
        if request.benchmark is not None:
            await report(0.8, f"Comparing against benchmark '{request.benchmark.id}'.")
            comparison = compare_against_benchmark(
                backtest.trading_nav, request.benchmark, fetcher, index_store,
                start, end)

        await report(0.9, "Assembling results.")
        # From the document rather than the definition: an optimised index has
        # no weight cap of its own — the solved weights are what its
        # constraints made them — so the field is simply absent there.
        cap = (document.pipeline.weighting.max_weight
               if document.pipeline is not None else None)
        payload = assemble_result(backtest, index_result, comparison, cap=cap)

        # The record is captured here or never: the library BacktestResult
        # exists only inside this job, and the run payload the job returns is
        # a derived view of it, not the books (BN-158). Latest run wins,
        # matching latest_result semantics.
        # A different cap from the payload's: that one stamps the *tracked*
        # book's snapshots, this one the record's *target* book, which on an
        # optimised run is the parent's calculation under the parent's cap.
        if record_store is not None:
            record = BacktestResultSummary.from_result(backtest,
                                                       cap=target_cap(definition))
            record_store.write(document.id, record.model_dump(mode="json"))

        await report(1.0, "Complete.")

        return payload.model_dump()

    return run
