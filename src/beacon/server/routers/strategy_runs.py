# src/beacon/server/routers/strategy_runs.py
"""
Backtesting a stored strategy, and reading its runs.

A backtest calculates the strategy's index and then simulates the portfolio day
by day, which is far too slow to hold an HTTP connection open for. The
submission endpoint therefore returns a job id; the client polls
`GET /jobs/{id}` or listens on `/ws`.

The read endpoints serve the panes of Beacon View (overview, weights,
attribution, a single name, and a comparison across strategies), and each
answers from the strategy's **most recent successful run** rather than
recalculating anything, so two panes read a moment apart describe the same
run. Every read is a 404 until the strategy has been backtested.
"""
# BN-313 moved these from `/beacon/{index_id}/...`: runs are kept per stored
# strategy, so two strategies on one index each keep their own.
from typing import Annotated, Any, Literal

import pandas as pd

from ..._optional import require
from ...data.fetcher import DataFetcher
from ...exceptions import ConfigurationError
from ..active_data import current_data, require_data
from ..backtest_settings import GENERIC, BuiltSettings
from ..backtest_settings import build as build_settings
from ..backtests import build_backtest_job
from ..documents import load_document, read_collection, validated
from ..errors import FindingsError
from ..holding_reference import ReferenceFieldsQuery, holding_reference
from ..holdings import build_holdings
from ..holdings_schemas import HoldingsView
from ..jobs import JobRegistry
from ..market_caps import (
    FREE_FLOAT_MARKET_CAP,
    MARKET_CAP,
    market_caps,
    universe_weights,
    uses_free_float,
)
from ..periods import build_periods
from ..schemas import (
    AssetView,
    AttributionView,
    BacktestJobStatus,
    BacktestRecordCollection,
    BacktestRecordRow,
    BacktestRequest,
    BacktestResultSummary,
    BacktestRunResult,
    BacktestValidation,
    CompareView,
    Identifier,
    IndexDocument,
    OverviewView,
    PeriodsView,
    ReferenceWindow,
    StrategyBacktestRequest,
    TolerantCollection,
    WeightsView,
)
from ..store import DocumentStore
from ..strategies import read_strategy
from ..strategy_runs import backtest_kind, index_weights, require_run
from ..strategy_schemas import StrategyDocument
from ..views import (
    build_asset_view,
    build_attribution,
    build_compare,
    build_overview,
)
from ..weights import build_weights

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Query, Request, status  # noqa: E402

BenchmarkQuery = Annotated[
    str | None,
    Query(description="Index id to measure tracking error against, calculated "
                      "on its own. Requires risk=true.")]
RiskQuery = Annotated[
    bool,
    Query(description="Decompose the portfolio's volatility across its "
                      "constituents. Off by default: estimating a covariance "
                      "over every name is the pane's whole cost.")]
AsOfQuery = Annotated[
    str | None,
    Query(description="Date to report at, YYYY-MM-DD. Defaults to the latest "
                      "rebalance.")]
StartQuery = Annotated[str | None,
                       Query(description="Inclusive start date, YYYY-MM-DD.")]
EndQuery = Annotated[str | None,
                     Query(description="Inclusive end date, YYYY-MM-DD.")]
# Each id non-empty and in the identifier pattern, and two or more, all in the
# spec so a client and the fuzzer read the same rule (BN-131, BN-250).
IdsQuery = Annotated[list[Identifier],
                     Query(min_length=2,
                           description="Strategy ids to compare, two or more.")]

FrequencyQuery = Annotated[Literal["annual", "quarterly", "monthly"],
                           Query(description="The calendar periods: years, quarters or "
                                             "months.")]

# The pointer a record that cannot be served carries, absent or unreadable.
RECORD_SOURCE = "run POST /strategies/{strategy_id}/backtest first"


def _strategy(request: Request,
              strategy_id: Identifier) -> StrategyDocument:
    """A stored strategy, or not-found."""
    store: DocumentStore = request.app.state.strategy_store

    return load_document(store, strategy_id, read_strategy, f"strategy '{strategy_id}'")


def _index_of(request: Request,
              strategy: StrategyDocument) -> IndexDocument:
    """The index a strategy holds, or not-found naming it."""
    return _index_document(request, strategy.index_id)


def _index_document(request: Request,
                    index_id: str) -> IndexDocument:
    """A stored index definition, or not-found."""
    store: DocumentStore = request.app.state.index_store

    return load_document(store, index_id, validated(IndexDocument), f"index '{index_id}'")


def _latest_run(request: Request,
                strategy_id: str) -> dict[str, Any]:
    """A strategy's latest successful run, or not-found."""
    registry: JobRegistry = request.app.state.jobs

    return require_run(registry, strategy_id)


def _record_row(strategy_id: str,
                document: dict[str, Any]) -> BacktestRecordRow:
    """The listing row for one stored record.

    The whole record is validated to produce the row (BN-174), so the listing
    never offers a row `/record` cannot open. A record without the strategy
    stamp was saved before runs were kept per strategy and is skipped.
    """
    record = BacktestResultSummary.model_validate(document)
    settings = record.settings or {}
    vehicle = settings.get("vehicle") or {}
    stamp = [settings.get(key) for key in ("strategy_id", "strategy_name", "index_id")]

    if not all(isinstance(value, str) for value in stamp):
        raise ConfigurationError(f"backtest record '{strategy_id}'",
                                 "it does not say which strategy it is of")

    return BacktestRecordRow(strategy_id=str(stamp[0]),
                             strategy_name=str(stamp[1]),
                             index_id=str(stamp[2]),
                             run_at=record.run_at,
                             strategy=(settings.get("strategy") or {}).get("type"),
                             vehicle=(vehicle.get("settings") or {}).get("name"),
                             vehicle_preset=(vehicle.get("preset")
                                             if vehicle.get("preset") != GENERIC else None),
                             currency=record.currency)


def _with_reference(record: BacktestResultSummary,
                    fetcher: DataFetcher | None,
                    reference_fields: str | None) -> BacktestResultSummary:
    """A record whose rows carry their names' reference columns: each
    rebalance and order as of its date, and each weights panel as the windows
    over its span."""
    # BN-325.
    books = [book for book in (record.portfolio, record.index.target, record.index.optimised,
                               record.benchmark) if book is not None]
    snapshots = [snapshot for book in books for snapshot in getattr(book, "rebalances", [])]
    names = {name for book in books for name in book.weights.columns}
    names |= {name for snapshot in snapshots for name in snapshot.weights}
    names |= {order.asset_id for order in record.unfilled}
    names |= {gap.asset_id for gap in record.price_gaps}
    reference = holding_reference(fetcher, reference_fields, names)

    for book in books:
        windows = (reference.windows_of(book.weights.columns,
                                        str(book.weights.index[0])[:10],
                                        str(book.weights.index[-1])[:10])
                   if book.weights.index else None)

        if windows is not None:
            book.reference_windows = {name: [ReferenceWindow.model_validate(window)
                                             for window in spans]
                                      for name, spans in windows.items()}

    for snapshot in snapshots:
        snapshot.reference = reference.map_of(snapshot.weights, snapshot.date)

    for order in record.unfilled:
        order.reference = reference.of(order.asset_id, order.date)

    for gap in record.price_gaps:
        gap.reference = reference.of(gap.asset_id, gap.date)

    return record


def _universe_weights(index: IndexDocument,
                      fetcher: DataFetcher,
                      names: list[str],
                      day: str) -> dict[str, float | None]:
    """Each name's share of the universe's market cap on *day*, on the
    index's basis (BN-329)."""
    caps = market_caps(fetcher, names, {MARKET_CAP, FREE_FLOAT_MARKET_CAP}, pd.Timestamp(day),
                       index.currency)
    weighting = index.pipeline.weighting if index.pipeline is not None else None

    return universe_weights(caps, names,
                            uses_free_float(weighting.scheme if weighting else None,
                                            weighting.params if weighting else None))


def _benchmark_weights(request: Request,
                       run: dict[str, Any],
                       fetcher: DataFetcher,
                       date: str | None) -> dict[str, float] | None:
    """The run's benchmark index's weights in force on *date*; None for a run
    without one, or with a benchmark that is a series rather than an index."""
    benchmark = (run.get("settings") or {}).get("benchmark") or {}

    if benchmark.get("kind") != "index":
        return None

    store: DocumentStore = request.app.state.index_store
    _, weights = index_weights(_index_document(request, benchmark["id"]), store, fetcher, date)

    return weights


def _request(strategy: StrategyDocument,
             body: StrategyBacktestRequest | None) -> BacktestRequest:
    """A backtest's settings: the body's, with the strategy's own strategy and
    implementation."""
    settings = body if body is not None else StrategyBacktestRequest()

    return BacktestRequest(**settings.model_dump(), strategy=strategy.strategy,
                           implementation=strategy.implementation)


def _built_settings(settings: BacktestRequest,
                    fetcher: DataFetcher) -> BuiltSettings:
    """A request's settings built, or 422 with every finding."""
    built, findings = build_settings(settings, fetcher)

    if built is None:
        raise FindingsError("backtest request", "its settings could not be built",
                            findings)

    return built


def _data_fetcher(request: Request) -> DataFetcher:
    """The loaded data, or 409 NO_DATA_LOADED (see `active_data`)."""
    return require_data(request, "a backtest cannot be run")


def build_strategy_runs_router() -> APIRouter:
    """Build the router for backtesting stored strategies and reading their
    runs, under /strategies.

    Mounted before the strategies router, so `/strategies/compare` and
    `/strategies/runs` are not read as strategy ids.

    Returns:
        APIRouter: Router carrying the backtest submission, its validation,
        and the result views.
    """
    router = APIRouter(prefix="/strategies", tags=["backtests"])

    # async, and it must stay that way: FastAPI runs a sync endpoint in a
    # worker thread, where there is no running event loop for the registry's
    # asyncio.create_task to attach to.
    @router.post("/{strategy_id}/backtest",
                 response_model=BacktestJobStatus,
                 status_code=status.HTTP_202_ACCEPTED)
    async def submit_backtest(request: Request,
                              strategy_id: Identifier,
                              body: StrategyBacktestRequest | None = None
                              ) -> BacktestJobStatus:
        """Backtest a stored strategy, as a job.

        The body is everything the strategy does not hold: the period,
        capital, costs, currency, dividends, benchmark, modelling
        assumptions, flows and vehicle. The strategy, its index and its
        implementation are the stored strategy's.
        """
        # Everything is resolved before the job is submitted, so an unknown
        # strategy, an index that has gone or an unconfigured server fails at
        # once rather than as a job a moment later.
        strategy = _strategy(request, strategy_id)
        document = _index_of(request, strategy)
        fetcher = _data_fetcher(request)
        settings = _request(strategy, body)
        built = _built_settings(settings, fetcher)

        registry: JobRegistry = request.app.state.jobs
        store: DocumentStore = request.app.state.index_store
        records: DocumentStore = request.app.state.backtest_record_store
        holdings: DocumentStore = request.app.state.holdings_store
        job = registry.submit(
            backtest_kind(strategy_id),
            build_backtest_job(strategy, document, fetcher, settings, store,
                               record_store=records, built=built, holdings_store=holdings))

        return BacktestJobStatus(**job.snapshot())

    @router.post("/{strategy_id}/backtest/validate", response_model=BacktestValidation)
    def validate_backtest(request: Request,
                          strategy_id: Identifier,
                          body: StrategyBacktestRequest | None = None) -> BacktestValidation:
        """What a backtest of the strategy would run, without running it.

        The same body `POST /strategies/{strategy_id}/backtest` takes.
        Answers the fully resolved settings or every finding, each at its
        field's path.
        """
        strategy = _strategy(request, strategy_id)
        _index_of(request, strategy)
        built, findings = build_settings(_request(strategy, body), _data_fetcher(request))

        return BacktestValidation(valid=built is not None,
                                  settings=built.resolved if built is not None else None,
                                  findings=findings)

    @router.get("/compare", response_model=CompareView)
    def compare(request: Request,
                ids: IdsQuery) -> CompareView:
        """Several strategies' latest runs on one axis, over the dates they
        all cover."""
        # Every strategy is resolved first, so an unknown id fails as a 404
        # naming it rather than as a comparison of fewer than were asked for.
        for strategy_id in ids:
            _strategy(request, strategy_id)

        return build_compare({strategy_id: _latest_run(request, strategy_id)
                              for strategy_id in ids})

    @router.get("/runs", response_model=BacktestRecordCollection)
    def runs(request: Request) -> BacktestRecordCollection:
        """Every strategy's latest saved run, newest first.

        One row per strategy: the record store keeps the latest run only,
        and `/strategies/{strategy_id}/record` serves its books. A record
        that cannot be read is skipped and counted rather than failing the
        listing.
        """
        records: DocumentStore = request.app.state.backtest_record_store
        rows, skipped = read_collection(records, _record_row, "backtest record")

        return BacktestRecordCollection(
            backtests=sorted(rows, key=lambda row: row.run_at or "", reverse=True),
            **TolerantCollection.skips(skipped))

    @router.get("/{strategy_id}/overview", response_model=OverviewView)
    def overview(request: Request,
                 strategy_id: Identifier) -> OverviewView:
        strategy = _strategy(request, strategy_id)

        return build_overview(strategy_id, strategy.index_id, strategy.name,
                              _latest_run(request, strategy_id))

    # BN-320: the run payload by strategy, so a client need not hold the job's id.
    @router.get("/{strategy_id}/result", response_model=BacktestRunResult)
    def result(request: Request,
               strategy_id: Identifier,
               reference_fields: ReferenceFieldsQuery = None) -> BacktestRunResult:
        """The strategy's latest successful run: the payload its job answered
        with, every series and record in it, kept across a restart. Each
        rebalance, unfilled order and price gap carries the reference columns
        of its names as they stood on its date.

        Raises:
            DataNotFoundError: Until the strategy has been backtested.
        """
        _strategy(request, strategy_id)
        payload = BacktestRunResult.model_validate(_latest_run(request, strategy_id))
        names = {name for snapshot in payload.rebalances for name in snapshot.weights}
        names |= {order.asset_id for order in payload.unfilled}
        names |= {gap.asset_id for gap in payload.price_gaps}
        # BN-325: a saved run reads without data, and then carries no columns.
        reference = holding_reference(current_data(request), reference_fields, names)

        for snapshot in payload.rebalances:
            snapshot.reference = reference.map_of(snapshot.weights, snapshot.date)

        for order in payload.unfilled:
            order.reference = reference.of(order.asset_id, order.date)

        for gap in payload.price_gaps:
            gap.reference = reference.of(gap.asset_id, gap.date)

        return payload

    # BN-324.
    @router.get("/{strategy_id}/periods", response_model=PeriodsView)
    def periods(request: Request,
                strategy_id: Identifier,
                frequency: FrequencyQuery = "annual") -> PeriodsView:
        """The latest run's figures in each calendar year, quarter or month,
        against its index and its benchmark: each period from the previous
        period's close, the first from the run's start, so the periods
        compound to the whole run."""
        strategy = _strategy(request, strategy_id)

        return build_periods(strategy_id, strategy.index_id,
                             _latest_run(request, strategy_id), frequency)

    # BN-326.
    @router.get("/{strategy_id}/holdings", response_model=HoldingsView)
    def holdings(request: Request,
                 strategy_id: Identifier,
                 date: AsOfQuery = None,
                 reference_fields: ReferenceFieldsQuery = None) -> HoldingsView:
        """The strategy's holdings workflow at a date, one row per name: the
        universe narrowed to the index, the index's cap, the benchmark's
        weight, the strategy's choice, each implementation stage, the weight
        traded to at the rebalance in force, and as of the date its drift,
        held weight and units. The run's last day when no date is given."""
        strategy = _strategy(request, strategy_id)
        index = _index_of(request, strategy)
        run = _latest_run(request, strategy_id)
        fetcher = _data_fetcher(request)
        store: DocumentStore = request.app.state.holdings_store
        saved = store.read(strategy_id)
        names = set((saved or {}).get("units") or {}) | {
            name for snapshot in run.get("rebalances") or [] for name in snapshot["weights"]}
        selections = ((saved or {}).get("selections") or {}).values()
        names |= {name for selection in selections for name in selection["universe"]}

        return build_holdings(strategy_id, index, run, saved, fetcher, date,
                              holding_reference(fetcher, reference_fields, names),
                              lambda rebalance: _benchmark_weights(request, run, fetcher,
                                                                   rebalance),
                              lambda names, day: _universe_weights(index, fetcher, names, day))

    @router.get("/{strategy_id}/record", response_model=BacktestResultSummary)
    def backtest_record(request: Request,
                        strategy_id: Identifier,
                        reference_fields: ReferenceFieldsQuery = None) -> BacktestResultSummary:
        """The latest run's books, nested: the record, not the derived view.

        `BacktestJobStatus.result` carries the run payload (rebased level,
        returns, drawdown), which is what a chart wants. This is the other
        half: the portfolio book with its day-zero NAV, bounded positions and
        weights with true totals, and the comparator books, null when the run
        had none.

        Raises:
            DataNotFoundError: If the strategy has never been backtested
                successfully. A record that cannot be read answers the same,
                as the listing skips it.
        """
        _strategy(request, strategy_id)
        records: DocumentStore = request.app.state.backtest_record_store
        record = load_document(records,
                               strategy_id,
                               validated(BacktestResultSummary),
                               f"a backtest record for strategy '{strategy_id}'",
                               source=RECORD_SOURCE)

        return _with_reference(record, current_data(request), reference_fields)

    @router.get("/{strategy_id}/weights", response_model=WeightsView)
    def weights(request: Request,
                strategy_id: Identifier,
                asof: AsOfQuery = None,
                risk: RiskQuery = False,
                benchmark: BenchmarkQuery = None,
                reference_fields: ReferenceFieldsQuery = None) -> WeightsView:
        strategy = _strategy(request, strategy_id)
        document = _index_of(request, strategy)
        run = _latest_run(request, strategy_id)
        fetcher = _data_fetcher(request)

        # The benchmark is an index, calculated on its own up to the same
        # date: what that index held then, with no backtest of it needed.
        benchmark_weights = None
        if benchmark and risk:
            store: DocumentStore = request.app.state.index_store
            _, benchmark_weights = index_weights(_index_document(request, benchmark),
                                                 store, fetcher, asof)

        view = build_weights(strategy_id,
                             strategy.index_id,
                             run,
                             asof,
                             fetcher,
                             with_risk=risk,
                             benchmark=benchmark_weights,
                             benchmark_id=benchmark if benchmark_weights else None,
                             currency=document.currency)
        # BN-325: each row's columns as they stood on the date asked about.
        reference = holding_reference(fetcher, reference_fields,
                                      [row.identifier for row in view.rows])

        for row in view.rows:
            row.reference = reference.of(row.identifier, view.as_of)

        return view

    @router.get("/{strategy_id}/attribution", response_model=AttributionView)
    def attribution(request: Request,
                    strategy_id: Identifier,
                    start: StartQuery = None,
                    end: EndQuery = None,
                    reference_fields: ReferenceFieldsQuery = None) -> AttributionView:
        strategy = _strategy(request, strategy_id)
        document = _index_of(request, strategy)
        fetcher = _data_fetcher(request)
        view = build_attribution(strategy_id,
                                 strategy.index_id,
                                 _latest_run(request, strategy_id),
                                 fetcher,
                                 start,
                                 end,
                                 document.currency)
        # BN-325: as the names stood at the window's end.
        reference = holding_reference(fetcher, reference_fields,
                                      [row.asset_id for row in view.contributions])

        for row in view.contributions:
            row.reference = reference.of(row.asset_id, view.end)

        return view

    @router.get("/{strategy_id}/assets/{identifier}", response_model=AssetView)
    def asset(request: Request,
              strategy_id: Identifier,
              identifier: str) -> AssetView:
        strategy = _strategy(request, strategy_id)
        document = _index_of(request, strategy)

        return build_asset_view(strategy_id,
                                strategy.index_id,
                                identifier,
                                _latest_run(request, strategy_id),
                                _data_fetcher(request),
                                document.currency)

    return router
