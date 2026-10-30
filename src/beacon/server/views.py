# src/beacon/server/views.py
"""
Reading a completed run: overview, weights, attribution, per-asset, compare.

A backtest is a job because it is slow. These are the panes that read what the
job produced, and they must be fast: a client switching tabs should not be
waiting on a recalculation. So they derive everything from the **stored run**
rather than recomputing the index.

The stored run carries its level and metrics, plus two things that cannot be
recovered from a NAV series:

* **rebalance snapshots**: the weights at each rebalance, both as applied and
  as they would have been uncapped. Everything about composition comes from
  these: the weights pane reads one, attribution drifts them forward day by day,
  the per-asset pane reads a name's history across them, and cap drag needs the
  uncapped set to compare against.
* **costs and starting capital**: two scalars, from which the cost drag falls
  out.

Daily weights are not stored. `drifted_weights()` reconstructs them from the
snapshots and the prices.

`attribute()` uses Carino linking, so the contributions sum to the compounded
total return rather than approximately to it. The endpoint reports the residual
regardless: it should sit at machine epsilon, and one that does not means an
assumption has broken somewhere upstream, which is worth surfacing rather than
rounding away.
"""
# Deriving from the stored run is what BN-91's result persistence was for.
# The level and metrics were already in the payload; the rebalance snapshots,
# costs and starting capital were added so these panes could be answered from
# it. Daily weights are deliberately left out: storing a weight per name per
# day would multiply the payload by the number of trading days to save an
# inexpensive calculation.
import logging
from typing import Any

import pandas as pd

from ..analysis.attribution.schedule import attribute_schedule
from ..analysis.performance.levels import compared
from ..analysis.relative import relative_metrics
from ..data.fetcher import DataFetcher
from ..data.prices import currencies_of
from ..exceptions import DataNotFoundError
from .runs import snapshots_from, weight_map
from .schemas import (
    AssetView,
    AttributionView,
    BacktestMetrics,
    CompareEntry,
    CompareView,
    ContributionPayload,
    OverviewView,
    RebalanceSnapshot,
    SeriesPayload,
)
from .weights import concentration_of, prices_for

logger = logging.getLogger(__name__)

def build_overview(strategy_id: str,
                   index_id: str,
                   name: str,
                   run: dict[str, Any]) -> OverviewView:
    """Headline view of a completed run."""
    level = SeriesPayload.model_validate(run["level"])
    snapshots = snapshots_from(run)
    latest = snapshots[-1]

    # The level index is stored as full ISO-8601 timestamps; the span is
    # published as YYYY-MM-DD (BN-187). These are dates rather than moments —
    # the time component is always midnight — and `CompareView` next door
    # already emits plain dates, so a client reading both had two parsing
    # paths for one concept.
    return OverviewView(
        strategy_id=strategy_id,
        index_id=index_id,
        name=name,
        start=str(level.index[0])[:10] if level.index else "",
        end=str(level.index[-1])[:10] if level.index else "",
        observations=len(level.index),
        rebalances=len(snapshots),
        last_rebalance=latest.date,
        metrics=BacktestMetrics.model_validate(run["metrics"]),
        concentration=concentration_of(latest.weights),
        level=level)


def build_attribution(strategy_id: str,
                      index_id: str,
                      run: dict[str, Any],
                      fetcher: DataFetcher,
                      start: str | None,
                      end: str | None,
                      currency: str | None = None) -> AttributionView:
    """Per-constituent contributions over a window, with the two drags.

    Prices are converted into *currency*, the index's, so each contribution
    includes its name's exchange-rate return and the contributions add up to
    the index's return.
    """
    snapshots = snapshots_from(run)

    # Prices are fetched over the whole history, not over the requested window.
    # Weights drift forward from each rebalance, so a rebalance that predates
    # the window still sets the weights inside it — fetching only the window
    # leaves those snapshots pointing at dates the price frame does not have.
    # The window is applied afterwards, by selecting periods.
    prices = _constituent_prices(fetcher, snapshots, None, None, currency)

    # Defaults to the run's own span, not the whole price history. The index
    # was only calculated over the window it was run for; drifting the last
    # rebalance forward past that would attribute returns for a period the
    # index does not cover, which reads as performance rather than as
    # extrapolation.
    span = _run_span(run)
    # BN-401: worked out by the library.
    result = attribute_schedule(weight_map(snapshots),
                                weight_map(snapshots, uncapped=True),
                                prices,
                                start or span[0],
                                end or span[1],
                                total_costs=float(run.get("total_costs") or 0.0),
                                initial_capital=float(run.get("initial_capital") or 0.0))

    # `start` is resolved — the shift drops the window's first date, so it is
    # a trading day later than what was asked for — and the request itself was
    # published nowhere. The pair beside it carries the window as given, or as
    # defaulted, which is what a client labelling a range needs (BN-187).
    return AttributionView(
        strategy_id=strategy_id,
        index_id=index_id,
        requested_start=start or span[0],
        requested_end=end or span[1],
        start=result.start,
        end=result.end,
        periods=result.periods,
        total_return=result.total_return,
        contributions=[ContributionPayload(asset_id=item.asset_id,
                                           contribution=item.contribution,
                                           average_weight=item.average_weight,
                                           total_return=item.total_return)
                       for item in result.contributions],
        residual=result.residual,
        reconciles=result.reconciles(),
        cap_drag=result.cap_drag,
        cost_drag=result.cost_drag)


def _run_span(run: dict[str, Any]) -> tuple[str | None, str | None]:
    """The first and last date the run actually covers."""
    level = SeriesPayload.model_validate(run["level"])
    if not level.index:
        return None, None

    return str(level.index[0])[:10], str(level.index[-1])[:10]


def _constituent_prices(fetcher: DataFetcher,
                        snapshots: list[RebalanceSnapshot],
                        start: str | None,
                        end: str | None,
                        currency: str | None = None) -> pd.DataFrame:
    """Prices for everything the index has ever held, over the window, in
    *currency*."""
    # BN-253: in each name's own currency until then, so attribution over a
    # multi-currency index missed every exchange-rate return.
    identifiers = sorted({name for snapshot in snapshots
                          for name in snapshot.weights})

    prices = prices_for(fetcher, identifiers, start, end, currency)
    if prices.empty:
        raise DataNotFoundError("prices for any constituent of this index",
                                source="MarketData")

    return prices


def build_asset_view(strategy_id: str,
                     index_id: str,
                     identifier: str,
                     run: dict[str, Any],
                     fetcher: DataFetcher,
                     currency: str | None = None) -> AssetView:
    """One constituent: its weight history and how it fared against the index.

    The comparison with the index is made in *currency*, the index's; the
    price series returned stays in the name's own currency, as a price chart
    should.
    """
    snapshots = snapshots_from(run)

    history = {snapshot.date: snapshot.weights[identifier]
               for snapshot in snapshots if identifier in snapshot.weights}
    if not history:
        raise DataNotFoundError(f"'{identifier}' in this index",
                                source="rebalance snapshots")

    # Keyed off the applied history, so the two series always cover the same
    # dates and a drilldown can plot them against each other without aligning
    # anything. Falls back to the applied weight where a run predates uncapped
    # weights being stored, since a missing raw figure means "no cap applied
    # here" more often than it means "unknown".
    raw_history = {
        snapshot.date: snapshot.uncapped_weights.get(identifier,
                                                     snapshot.weights[identifier])
        for snapshot in snapshots if identifier in snapshot.weights}

    frame = fetcher.fetch_market_data(identifier)
    if frame.empty or "CLOSE" not in frame.columns:
        raise DataNotFoundError(f"prices for '{identifier}'", source="MarketData")

    level = SeriesPayload.model_validate(run["level"])
    index_level = pd.Series(level.data,
                            index=pd.to_datetime(level.index)).astype(float)

    # Compared in the index's money: a name's return in its own currency
    # against an index's in another is not a like-for-like excess (BN-253).
    comparable = (fetcher.fetch_prices([identifier], currency=currency)
                  .get(identifier, frame["CLOSE"]))
    metrics = relative_metrics(comparable, index_level)

    return AssetView(
        strategy_id=strategy_id,
        index_id=index_id,
        identifier=identifier,
        weight_history=history,
        raw_weight_history=raw_history,
        rebalances_held=len(history),
        total_return=metrics.total_return,
        index_return=metrics.benchmark_return,
        excess_return=metrics.excess_return,
        tracking_error=metrics.tracking_error,
        correlation=metrics.correlation,
        beta=metrics.beta,
        observations=metrics.observations,
        price=SeriesPayload.from_series(frame["CLOSE"]),
        currency=currency.upper() if currency else None,
        price_currency=currencies_of(fetcher, [identifier], "")[identifier] or None)


def build_compare(runs: dict[str, dict[str, Any]]) -> CompareView:
    """Several strategies' latest runs on one axis, over the window they all
    share.

    Aligned rather than concatenated: two indices with different start dates
    would otherwise be compared over different periods, and the one with the
    shorter history would look better or worse for no reason but its span.
    Every level is rebased to 100 on the first shared date, so the lines start
    together and the comparison is of shape rather than of scale.
    """
    # BN-401: aligned and rebased by the library.
    found = compared({strategy_id: _level_series(run) for strategy_id, run in runs.items()})
    window = found.window

    return CompareView(strategy_ids=list(runs),
                       start=str(window[0].date()),
                       end=str(window[-1].date()),
                       observations=len(window),
                       entries=[CompareEntry(strategy_id=strategy_id,
                                             total_return=found.total_returns[strategy_id],
                                             level=SeriesPayload.from_series(level))
                                for strategy_id, level in found.levels.items()])


def _level_series(run: dict[str, Any]) -> pd.Series:
    """The run's level as a date-indexed Series."""
    payload = SeriesPayload.model_validate(run["level"])

    return pd.Series(payload.data,
                     index=pd.to_datetime(payload.index)).astype(float)
