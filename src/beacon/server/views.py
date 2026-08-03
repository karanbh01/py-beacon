# src/beacon/server/views.py
"""
Reading a completed run: overview, weights, attribution, per-asset, compare.

A backtest is a job because it is slow. These are the panes that read what the
job produced, and they must be fast — a client switching tabs should not be
waiting on a recalculation. So they derive everything from the **stored run**
rather than recomputing the index, which is what BN-91's result persistence was
for.

## What the run has to carry, and why

Level and metrics were already in the payload. Two things were not, and neither
can be recovered from a NAV series:

* **rebalance snapshots** — the weights at each rebalance, both as applied and
  as they would have been uncapped. Everything about composition comes from
  these: the weights pane reads one, attribution drifts them forward day by day,
  the per-asset pane reads a name's history across them, and cap drag needs the
  uncapped set to compare against.
* **costs and starting capital** — two scalars, from which the cost drag falls
  out.

Daily weights are deliberately *not* stored. `drifted_weights()` reconstructs
them from the snapshots and the prices, and storing a weight per name per day
would multiply the payload by the number of trading days to save an
inexpensive calculation.

## Attribution reconciles, and that is the point

`attribute()` uses Carino linking, so the contributions sum to the compounded
total return rather than approximately to it. The endpoint reports the residual
regardless: it should sit at machine epsilon, and one that does not means an
assumption has broken somewhere upstream — which is worth surfacing rather than
rounding away.
"""
import logging
from typing import Any

import pandas as pd

from ..analysis.attribution import attribute, cap_drag, cost_drag, drifted_weights
from ..analysis.relative import relative_metrics
from ..data.fetcher import DataFetcher
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

def build_overview(index_id: str,
                   name: str,
                   run: dict[str, Any]) -> OverviewView:
    """Headline view of a completed run."""
    level = SeriesPayload.model_validate(run["level"])
    snapshots = snapshots_from(run)
    latest = snapshots[-1]

    return OverviewView(
        index_id=index_id,
        name=name,
        start=str(level.index[0]) if level.index else "",
        end=str(level.index[-1]) if level.index else "",
        observations=len(level.index),
        rebalances=len(snapshots),
        last_rebalance=latest.date,
        metrics=BacktestMetrics.model_validate(run["metrics"]),
        concentration=concentration_of(latest.weights),
        level=level)


def build_attribution(index_id: str,
                      run: dict[str, Any],
                      fetcher: DataFetcher,
                      start: str | None,
                      end: str | None) -> AttributionView:
    """Per-constituent contributions over a window, with the two drags."""
    snapshots = snapshots_from(run)
    capped = weight_map(snapshots)
    uncapped = weight_map(snapshots, uncapped=True)

    # Prices are fetched over the whole history, not over the requested window.
    # Weights drift forward from each rebalance, so a rebalance that predates
    # the window still sets the weights inside it — fetching only the window
    # leaves those snapshots pointing at dates the price frame does not have.
    # The window is applied afterwards, by selecting periods.
    prices = _constituent_prices(fetcher, snapshots, None, None)

    weights = drifted_weights(capped, prices)
    asset_returns = prices.pct_change().reindex(weights.index)
    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

    # Defaults to the run's own span, not the whole price history. The index
    # was only calculated over the window it was run for; drifting the last
    # rebalance forward past that would attribute returns for a period the
    # index does not cover, which reads as performance rather than as
    # extrapolation.
    span = _run_span(run)
    window = _window_of(weights.index, start or span[0], end or span[1])
    weights = weights.loc[window]
    asset_returns = asset_returns.loc[window]
    period_returns = period_returns.loc[window]

    result = attribute(period_returns,
                       weights,
                       asset_returns,
                       cap_drag=_cap_drag(capped, uncapped, prices.loc[window]),
                       cost_drag=_cost_drag(run))

    return AttributionView(
        index_id=index_id,
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


def _window_of(index: pd.Index,
               start: str | None,
               end: str | None) -> pd.Index:
    """The requested slice of a date index.

    Raises:
        DataNotFoundError: If the window contains no dates. Attributing over an
            empty period would report a total return of zero, which reads as a
            flat index rather than as a question that could not be answered.
    """
    selected = pd.DatetimeIndex(index)

    if start is not None:
        selected = selected[selected >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected <= pd.Timestamp(end)]

    if selected.empty:
        raise DataNotFoundError(
            f"any index dates between {start or 'the start'} and "
            f"{end or 'the end'}",
            source="the run covers a different period")

    return selected


def _cap_drag(capped: dict[pd.Timestamp, dict[str, float]],
              uncapped: dict[pd.Timestamp, dict[str, float]],
              prices: pd.DataFrame) -> float | None:
    """What the cap cost, or None when nothing was capped.

    Reporting 0.0 for an uncapped index would be a different claim — that
    capping happened and made no difference — so an index with no cap says
    nothing rather than zero.
    """
    if capped == uncapped:
        return None

    return cap_drag(capped, uncapped, prices)


def _cost_drag(run: dict[str, Any]) -> float | None:
    """The direct effect of transaction costs, or None at zero cost."""
    costs = float(run.get("total_costs") or 0.0)
    capital = float(run.get("initial_capital") or 0.0)

    if costs <= 0.0 or capital <= 0.0:
        return None

    return cost_drag(costs, capital)


def _constituent_prices(fetcher: DataFetcher,
                        snapshots: list[RebalanceSnapshot],
                        start: str | None,
                        end: str | None) -> pd.DataFrame:
    """Prices for everything the index has ever held, over the window."""
    identifiers = sorted({name for snapshot in snapshots
                          for name in snapshot.weights})

    prices = prices_for(fetcher, identifiers, start, end)
    if prices.empty:
        raise DataNotFoundError("prices for any constituent of this index",
                                source="MarketData")

    return prices


def build_asset_view(index_id: str,
                     identifier: str,
                     run: dict[str, Any],
                     fetcher: DataFetcher) -> AssetView:
    """One constituent: its weight history and how it fared against the index."""
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

    metrics = relative_metrics(frame["CLOSE"], index_level)

    return AssetView(
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
        price=SeriesPayload.from_series(frame["CLOSE"]))


def build_compare(runs: dict[str, dict[str, Any]]) -> CompareView:
    """Several indices on one axis, over the window they all share.

    Aligned rather than concatenated: two indices with different start dates
    would otherwise be compared over different periods, and the one with the
    shorter history would look better or worse for no reason but its span.
    Every level is rebased to 100 on the first shared date, so the lines start
    together and the comparison is of shape rather than of scale.
    """
    levels = {index_id: _level_series(run) for index_id, run in runs.items()}
    window = _common_window(levels)

    entries = []
    for index_id, series in levels.items():
        clipped = series.loc[window]
        rebased = clipped / clipped.iloc[0] * 100.0

        entries.append(CompareEntry(
            index_id=index_id,
            total_return=float(rebased.iloc[-1] / rebased.iloc[0] - 1.0),
            level=SeriesPayload.from_series(rebased)))

    return CompareView(index_ids=list(runs),
                       start=str(window[0].date()),
                       end=str(window[-1].date()),
                       observations=len(window),
                       entries=entries)


def _level_series(run: dict[str, Any]) -> pd.Series:
    """The run's level as a date-indexed Series."""
    payload = SeriesPayload.model_validate(run["level"])

    return pd.Series(payload.data,
                     index=pd.to_datetime(payload.index)).astype(float)


def _common_window(levels: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Dates every series covers.

    Built by intersecting rather than by taking the widest start and narrowest
    end, so a gap in the middle of one series drops that date for everyone
    instead of leaving a hole only one line has.
    """
    shared: pd.DatetimeIndex | None = None

    for series in levels.values():
        index = pd.DatetimeIndex(series.index)
        shared = index if shared is None else shared.intersection(index)

    if shared is None or shared.empty:
        raise DataNotFoundError(
            "a window these indices share",
            source="their level series do not overlap on any date")

    return shared.sort_values()

