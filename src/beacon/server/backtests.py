# src/beacon/server/backtests.py
"""
Backtest job body and result assembly.

Everything reported here derives from a single canonical series (the
portfolio NAV from its initial capital), so the payload is internally
consistent by
construction rather than by coincidence. A client that recomputes drawdown
from the level series, or compounds the annual returns, must land back on the
numbers the server sent; if those were computed independently they would drift
apart at the last decimal and nobody would know which to trust.
"""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..analysis.data import RunData
from ..analysis.performance import formulas
from ..analysis.performance.levels import annual_returns, daily_returns, rebased
from ..analysis.relative import align_on_common_window, relative_metrics
from ..analysis.request import keep_analysis_work
from ..backtest.main import Backtest
from ..backtest.result import BacktestResult
from ..backtest.share_classes import ShareClass
from ..data.fetcher import DataFetcher
from ..index.constructor import IndexDefinition
from ..index.derived import AnyIndexDefinition, OptimisedIndexDefinition
from ..index.result import IndexResult
from ..portfolio.cash_flows import FEE
from .analysis import build_request, keep_analytics, with_tables
from .analysis_schemas import AnalyticsBlock
from .backtest_settings import ACTIVE, INDEX_TRACKING, BuiltSettings
from .backtest_settings import build as build_settings
from .benchmarks import resolve_benchmark_book
from .comparisons import Comparators, comparison_payloads
from .definitions import build_definition
from .derivative_records import derivative_payloads
from .errors import FindingsError
from .holdings import HOLDINGS_KEY, holdings_document
from .job_threads import Report, run_in_thread
from .jobs import JobBody, ProgressReporter
from .record_tables import slim_record
from .schemas import (
    ActiveStepPayload,
    BacktestRequest,
    BacktestResultSummary,
    BacktestRunResult,
    BenchmarkRef,
    ComparisonsPayload,
    IndexDocument,
    MarketPayload,
    RebalanceStepPayload,
    RelativeMetricsPayload,
    ReplicationPayload,
    SeriesPayload,
    metrics_from,
    price_gap_payloads,
    rebalance_pricing_payloads,
    rebalance_snapshots,
    unfilled_payloads,
)
from .share_class_results import flow_payloads, share_class_payloads
from .store import DocumentStore
from .strategy_schemas import StrategyDocument


@dataclass(frozen=True)
class ClassContext:
    """What measuring a run's share classes needs: the classes, for their
    hedge ratios, the index's currency and the data the rates come from."""
    # BN-314.
    share_classes: Sequence[ShareClass]
    index_currency: str
    fetcher: DataFetcher


def _drawdown(level: pd.Series) -> pd.Series:
    """Drawdown from the running peak, derived from *level* itself."""
    return formulas.drawdown(level)


def _performance_from_start(result: BacktestResult) -> pd.Series:
    """The performance series with its opening point: with flows a unit at
    its launch price on the eve, otherwise the NAV from the capital."""
    if not result.flows:
        return _nav_from_capital(result)

    eve = result.portfolio.inception
    per_unit = result.nav_per_unit

    if eve is None or per_unit.empty:
        return per_unit

    return pd.concat([pd.Series({eve: result.launch_price}), per_unit])


def _nav_from_capital(result: BacktestResult) -> pd.Series:
    """The NAV with its opening row: the initial capital on the eve of the
    first day, then one value per simulated day. The metrics start from the
    same point, so every series here agrees with them."""
    nav = result.portfolio.nav

    return nav if len(nav) > len(result.trading_nav) else result.trading_nav


def assemble_result(result: BacktestResult,
                    index_result: IndexResult,
                    benchmark: RelativeMetricsPayload | None = None,
                    cap: float | None = None,
                    settings: dict[str, Any] | None = None,
                    classes: ClassContext | None = None,
                    comparators: Comparators | None = None) -> BacktestRunResult:
    """Build the wire payload from a completed backtest.

    Args:
        result: The finished backtest.
        index_result: The index it tracked, reported alongside as the
            replication reference.
        benchmark: Optional comparison against an external benchmark.
        classes: What measuring its share classes needs; None without them.
        comparators: The index and benchmark it is compared against; None
            leaves the comparisons empty.

    Returns:
        BacktestRunResult: Level, returns, drawdown, annual returns, the
        tracked index and metrics, all derived from the same NAV series.
    """
    # `level` is the run's own days, rebased to 100 on the first: the views
    # read its dates as the run's window. The other series start from the
    # initial capital (BN-246), as the metrics do, so the first day's return
    # carries the cost of the opening trades.
    # With flows the performance is a unit's, so a flow is not drawn as a gain
    # (BN-267); without, it is the NAV, as it always was.
    level = rebased(result.performance_levels())
    from_capital = rebased(_performance_from_start(result))
    opening = len(from_capital) - len(level)
    returns = daily_returns(from_capital)

    return BacktestRunResult(
        level=SeriesPayload.from_series(level),
        currency=result.currency,
        returns=SeriesPayload.from_series(returns),
        drawdown=SeriesPayload.from_series(_drawdown(from_capital).iloc[opening:]),
        annual_returns=annual_returns(from_capital),
        index_level=SeriesPayload.from_series(rebased(index_result.index_levels)),
        metrics=metrics_from(result.summary()),
        benchmark=benchmark,
        rebalances=rebalance_snapshots(index_result, cap),
        total_costs=result.total_costs,
        initial_capital=result.portfolio.initial_capital,
        price_gaps=price_gap_payloads(result),
        rebalance_pricing=rebalance_pricing_payloads(result),
        settings=settings,
        **strategy_and_vehicle_payloads(result),
        **derivative_payloads(result, level.index),
        comparisons=(comparison_payloads(result, index_result, comparators)
                     if comparators is not None else ComparisonsPayload()),
        **(share_class_payloads(result, classes.share_classes, level.index,
                                index_result.index_levels, classes.index_currency,
                                classes.fetcher)
           if classes is not None else {}))


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


def compare_against_benchmark(nav: pd.Series,
                              reference: BenchmarkRef,
                              levels: pd.Series) -> RelativeMetricsPayload:
    """Measure the portfolio against a resolved benchmark.

    The benchmark series is rebased on the *aligned* window rather than its own
    full history, so both lines start at 100 on the same date and can be read
    off one axis. Rebasing before alignment would leave the benchmark starting
    somewhere other than 100 once trimmed.

    Args:
        nav: Portfolio NAV series.
        reference: What it is compared against.
        levels: The benchmark's levels, resolved.

    Returns:
        RelativeMetricsPayload: The comparison and the rebased benchmark.
    """
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
        level=SeriesPayload.from_series(rebased(aligned_benchmark)))


# Where the backtest's own progress sits in the job's (BN-342).
RUN_START = 0.05
RUN_SPAN = 0.75


def build_backtest_job(strategy: StrategyDocument,
                       document: IndexDocument,
                       fetcher: DataFetcher,
                       request: BacktestRequest,
                       index_store: DocumentStore,
                       record_store: DocumentStore | None = None,
                       built: BuiltSettings | None = None) -> JobBody:
    """Build the job body that runs one backtest of a stored strategy.

    Returned as a closure rather than run inline: the caller submits it to the
    job registry, which owns scheduling and progress publication.

    Args:
        strategy: The stored strategy run. Its id and name are stamped on the
            run's settings, and its record is filed under its id.
        document: The index it holds, to calculate and then track.
        fetcher: Data source for both the index and the simulation.
        request: Period, capital and cost settings for this run, with the
            strategy's own strategy and implementation.
        built: Its strategy, implementation, flows, vehicle and modelling
            assumptions, already built; built here from *request* when None.

    Returns:
        JobBody: A coroutine function taking a progress reporter.
    """
    if built is None:
        built, findings = build_settings(request, fetcher)

        if built is None:
            raise FindingsError("backtest request", "its settings could not be built",
                                findings)

    settings = built
    # BN-398: refused here, before the run, when the analysis cannot be given.
    analysis = build_request(request.analysis) if request.analysis is not None else None

    def work(report: Report) -> dict[str, object]:
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
        # BN-365: a benchmark is resolved first and given to the run, so a
        # stored index's holdings are kept with it as well as its levels.
        benchmark = None
        if request.benchmark is not None:
            report(0.03, f"Calculating benchmark '{request.benchmark.id}'.")
            benchmark = resolve_benchmark_book(request.benchmark, fetcher, index_store,
                                               start, end)

        report(0.05, _stage(settings))

        backtest = Backtest(initial_capital=settings.initial_capital,
                            transaction_cost_bps=request.transaction_cost_bps,
                            currency=settings.currency,
                            modelling_assumptions=settings.modelling_assumptions,
                            dividends=settings.dividends,
                            reinvest_into=settings.reinvest_into,
                            implementation=settings.implementation,
                            flows=settings.flows or None,
                            share_classes=settings.share_classes or None,
                            vehicle=settings.vehicle,
                            benchmark=benchmark,
                            data_provider=fetcher).run(
            settings.strategy(definition),
            start=start,
            end=end,
            analysis=analysis,
            # BN-342: the run's progress, by stage and then by trading day,
            # inside the job's own span from here to the benchmark at 0.8.
            progress=lambda fraction, message: report(RUN_START + RUN_SPAN * fraction, message))
        # BN-313: which strategy ran, so a run is never served as another's.
        resolved = {"strategy_id": strategy.id, "strategy_name": strategy.name,
                    "index_id": document.id,
                    **settings.resolved, "start": start, "end": end,
                    "modelling_assumptions": (
            backtest.modelling_assumptions.as_dict()
            if backtest.modelling_assumptions is not None
            else settings.resolved["modelling_assumptions"])}

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
        benchmark_levels = None
        if request.benchmark is not None:
            report(0.8, f"Comparing against benchmark '{request.benchmark.id}'.")
            # Resolved once, before the run, for this comparison and the
            # run's comparisons.
            benchmark_levels = (benchmark.index_levels if isinstance(benchmark, IndexResult)
                                else benchmark)
            comparison = compare_against_benchmark(backtest.trading_nav, request.benchmark,
                                                   benchmark_levels)

        report(0.9, "Assembling results.")
        # From the document rather than the definition: an optimised index has
        # no weight cap of its own — the solved weights are what its
        # constraints made them — so the field is simply absent there.
        cap = (document.pipeline.weighting.max_weight
               if document.pipeline is not None else None)
        payload = assemble_result(backtest, index_result, comparison, cap=cap,
                                  settings=resolved,
                                  classes=ClassContext(settings.share_classes,
                                                       document.currency, fetcher),
                                  comparators=Comparators(
                                      index_id=document.id,
                                      benchmark=request.benchmark,
                                      benchmark_levels=benchmark_levels))

        # The record is captured here or never: the library BacktestResult
        # exists only inside this job, and the run payload the job returns is
        # a derived view of it, not the books (BN-158). Latest run wins,
        # matching latest_result semantics.
        # A different cap from the payload's: that one stamps the *tracked*
        # book's snapshots, this one the record's *target* book, which on an
        # optimised run is the parent's calculation under the parent's cap.
        if record_store is not None:
            # A report first, so a run cancelled by now stops before it files
            # its record (BN-333).
            report(0.95, "Saving the record.")
            # The weights are left to the run's tables (BN-404), so they are
            # not turned into lists only to be dropped.
            record = BacktestResultSummary.from_result(backtest,
                                                       cap=target_cap(definition),
                                                       settings=resolved,
                                                       weights=False)
            # BN-326: what each rebalance decided, for the holdings workflow,
            # kept in the record under a key its model does not read, so the
            # record is served as before and goes when it goes.
            # BN-361: the tables too large for the JSON, beside it, written
            # before the record so a record never sits beside an older run's.
            # BN-369: as RunData keeps it, so its analysis can be rerun.
            # BN-404: the record's large parts as tables there too.
            run = RunData.from_result(backtest)
            folder = record_store.panels_directory(strategy.id)
            run.write(folder)
            whole = {**record.model_dump(mode="json"),
                     HOLDINGS_KEY: holdings_document(backtest, index_result)}
            record_store.write(strategy.id, slim_record(whole, run, folder))

        # BN-398: the analysis in the payload, and beside the run's tables,
        # read by strategy.
        if backtest.analysis_block is not None:
            block = with_tables(backtest.analysis_block, strategy.id, backtest.run_data.tables)
            payload.analytics = AnalyticsBlock.model_validate(block)

            if record_store is not None:
                kept_in = record_store.panels_directory(strategy.id)
                keep_analytics(kept_in, block)
                # BN-399: the factor work too, so a rerun does not do it again.
                keep_analysis_work(backtest, kept_in)

        report(1.0, "Complete.")

        return payload.model_dump()

    # BN-333: the work runs in a worker thread, so the event loop answers
    # requests, /health among them, while it does.
    async def run(report: ProgressReporter) -> dict[str, object]:
        return await run_in_thread(report, work)

    return run


def strategy_and_vehicle_payloads(result: BacktestResult) -> dict[str, Any]:
    """What a strategy, flows, a vehicle and the implementation's stages add
    to the run payload; each empty or null when unused."""
    payloads: dict[str, Any] = {
        "flows": flow_payloads(result.flows),
        "fees_paid": -sum(flow.amount for flow in result.portfolio.cash_flows
                          if flow.kind == FEE),
        "replication": [ReplicationPayload(date=str(step.date.date()),
                                           holdings=step.holdings,
                                           tracking_error=step.tracking_error,
                                           unmeasured=list(step.unmeasured),
                                           weights=step.weights)
                        for step in result.replication],
        "active": [ActiveStepPayload(date=str(step.date.date()),
                                     tracking_error=step.tracking_error,
                                     active_share=step.active_share,
                                     weights=step.weights,
                                     risk_contributions=step.risk_contributions,
                                     exposures=step.exposures)
                   for step in result.active],
        "rebalance_steps": [RebalanceStepPayload(date=str(step.date.date()),
                                                 removed=step.removed,
                                                 capped=step.capped,
                                                 cash_weight=step.cash_weight)
                            for step in result.rebalance_steps],
        "unfilled": unfilled_payloads(result),
    }

    if result.flows:
        payloads["nav_per_unit"] = SeriesPayload.from_series(result.nav_per_unit)
        payloads["units"] = SeriesPayload.from_series(result.units_outstanding)

    if not result.market.empty:
        market = result.market
        payloads["market"] = MarketPayload(
            **{column: SeriesPayload.from_series(market[column])
               for column in ("market_price", "premium", "spread", "bid", "ask")})

    return payloads


def _stage(settings: BuiltSettings) -> str:
    """The job's message while the run is calculated and simulated."""
    holding = {INDEX_TRACKING: "replicating the index",
               ACTIVE: "building the active strategy"}.get(settings.strategy_type,
                                                         "tracking the index")
    vehicle = (f" in a {settings.vehicle.name}" if settings.vehicle is not None
               else "")

    return f"Calculating the index, {holding} and simulating it{vehicle}."
