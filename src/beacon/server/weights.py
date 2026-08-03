# src/beacon/server/weights.py
"""
The weights pane: composition at a date, per constituent.

Split out of `views.py`, which was at the point where adding per-row detail
would have taken it well past the size this codebase keeps modules to. What
lives here is one question — *what does the index hold, and how did it get
there* — and everything the table needs to answer it.

## Two weights per row, not one

`raw_weight` is what the weighting scheme produced; `weight` is what survived
the cap. The pair is what makes a capped index legible: without the raw figure
a reader sees several names sitting at exactly 20% and cannot tell whether the
cap was binding hard on one and barely on another. The library has kept
`uncapped_weights` on every snapshot since capping was written, precisely so
this counterfactual stays available — this endpoint surfaces it rather than
computing anything new.

Their sums are the check worth knowing: raw weights sum to 1 and applied
weights sum to 1, and the weight moved between them is `cap_redistributed`.

## Drift is per name and in aggregate, from one walk

The aggregate `DriftPayload` and every row's `delta_since_rebalance` come from
the same held-weight vector. Computing them separately would let the total stop
matching the rows it is a total of, which is the kind of disagreement nobody
notices until a client displays both.

Drift is measured against the *targets the last rebalance set*, not against the
previous rebalance. An equal-weighted index resets to 1/n every time, so
comparing consecutive rebalances would report zero drift forever; the question
worth answering is how far prices have moved the index since it was last reset.

## `shares_outstanding` is the company's, not the index's

An index fact sheet's "shares" column is usually shares *held per index unit*,
which needs a divisor and a notional this endpoint has neither of. What the
data layer holds is the company's shares outstanding, so that is what is
served — under a name that cannot be mistaken for the other figure. A field
called `shares` would have been read as index shares by at least one person.
"""
import logging
from typing import Any

import pandas as pd

from ..analysis.attribution import drifted_weights
from ..analysis.concentration import concentration, drift_from_target, top_n_weight
from ..data.fetcher import DataFetcher
from ..risk.contribution import RiskContributions, risk_contributions
from ..risk.model import estimate_risk_model
from .runs import snapshot_at, snapshots_from
from .schemas import (
    ConcentrationPayload,
    ConstituentRow,
    DriftPayload,
    RebalanceSnapshot,
    RiskPayload,
    WeightsView,
)

logger = logging.getLogger(__name__)

# Sizes reported by the weights pane. Both are conventional concentration
# cutoffs and cheap to compute, so reporting both saves a round trip.
TOP_N = (5, 10)

# Below this many observations a covariance over hundreds of names is
# noise wearing a matrix's clothes. Reporting nothing beats reporting a
# decomposition of an estimate that means nothing.
MINIMUM_OBSERVATIONS = 60


def concentration_of(weights: dict[str, float]) -> ConcentrationPayload:
    """Concentration measures for one weight vector."""
    metrics = concentration(weights)

    return ConcentrationPayload(
        herfindahl=metrics.herfindahl_index,
        effective_assets=metrics.effective_assets,
        top_weights={str(size): top_n_weight(weights, size) for size in TOP_N},
        largest=metrics.largest_weight,
        constituents=metrics.assets)


def build_weights(index_id: str,
                  run: dict[str, Any],
                  as_of: str | None,
                  fetcher: DataFetcher,
                  with_risk: bool = False) -> WeightsView:
    """Composition at a date, with per-constituent rows, drift and cap flags.

    Args:
        index_id: The index being read.
        run: The stored run, for context the snapshot does not carry.
        as_of: Date asked about; None means the latest rebalance.
        fetcher: Data source, for prices and shares outstanding.
        with_risk: Decompose the index's volatility across its constituents.
            Off by default because estimating a covariance over every name is
            the pane's whole cost.

    Returns:
        WeightsView: The pane's whole payload.
    """
    snapshot = snapshot_at(snapshots_from(run), as_of)
    held = _held_weights(snapshot, as_of, fetcher)

    contributions, window = (_risk_of(snapshot, run, fetcher)
                             if with_risk else (None, (None, None)))

    return WeightsView(
        index_id=index_id,
        as_of=as_of or snapshot.date,
        rebalance_date=snapshot.date,
        announced_date=snapshot.announced,
        weights=snapshot.weights,
        rows=build_rows(snapshot, held, as_of, fetcher, contributions),
        concentration=concentration_of(snapshot.weights),
        drift=_drift_payload(snapshot, held),
        capped=snapshot.capped,
        cap=snapshot.cap,
        cap_redistributed=snapshot.redistributed,
        risk=_risk_payload(contributions, window))


def _held_weights(snapshot: RebalanceSnapshot,
                  as_of: str | None,
                  fetcher: DataFetcher) -> dict[str, float] | None:
    """What the targets have drifted to by *as_of*.

    None when there is nothing to measure — *as_of* is the rebalance date
    itself, or no prices cover the window. Both the aggregate drift and the
    per-row deltas read this one result, so they cannot disagree.
    """
    if as_of is None or pd.Timestamp(as_of) <= pd.Timestamp(snapshot.date):
        return None

    prices = prices_for(fetcher, sorted(snapshot.weights), snapshot.date, as_of)
    if prices.empty:
        return None

    drifted = drifted_weights({pd.Timestamp(snapshot.date): snapshot.weights},
                              prices)
    if drifted.empty:
        return None

    return {str(name): float(value)
            for name, value in drifted.iloc[-1].items() if pd.notna(value)}


def _drift_payload(snapshot: RebalanceSnapshot,
                   held: dict[str, float] | None) -> DriftPayload | None:
    """The aggregate drift figures, from the held weights."""
    if held is None:
        return None

    metrics = drift_from_target(held, snapshot.weights)

    return DriftPayload(total_absolute=metrics.total_absolute,
                        maximum=metrics.max_absolute,
                        worst=metrics.max_absolute_asset_id or "",
                        turnover=metrics.turnover,
                        since=snapshot.date)


def _estimation_window(run: dict[str, Any]) -> tuple[str | None, str | None]:
    """The span the run covers, which the covariance is estimated over.

    The run's own window rather than a user-chosen one: a decomposition of
    *this* index's risk should be estimated over the period the index was
    calculated for, and letting the two diverge would put a number on the pane
    that describes a different history from the levels beside it.
    """
    level = run.get("level") or {}
    index = level.get("index") or []

    if not index:
        return None, None

    return str(index[0])[:10], str(index[-1])[:10]


def _risk_of(snapshot: RebalanceSnapshot,
             run: dict[str, Any],
             fetcher: DataFetcher
             ) -> tuple[RiskContributions | None, tuple[str | None, str | None]]:
    """Decompose the index's volatility across the rebalance's holdings."""
    window = _estimation_window(run)
    prices = prices_for(fetcher, sorted(snapshot.weights), *window)

    if prices.empty:
        logger.warning("No prices over %s to %s, so no risk decomposition.",
                       *window)

        return None, window

    returns = prices.pct_change().dropna(how="all")
    if len(returns) < MINIMUM_OBSERVATIONS:
        logger.warning(
            "Only %d return observation(s) over the run's window, fewer than "
            "the %d a covariance needs; no risk decomposition.",
            len(returns), MINIMUM_OBSERVATIONS)

        return None, window

    model = estimate_risk_model(returns)

    return risk_contributions(snapshot.weights, model.covariance), window


def _risk_payload(contributions: RiskContributions | None,
                  window: tuple[str | None, str | None]) -> RiskPayload | None:
    """The risk block, or None when nothing could be estimated."""
    if contributions is None:
        return None

    return RiskPayload(volatility=contributions.volatility,
                       covered_weight=contributions.covered_weight,
                       uncovered=list(contributions.uncovered),
                       window_start=window[0],
                       window_end=window[1])


def build_rows(snapshot: RebalanceSnapshot,
               held: dict[str, float] | None,
               as_of: str | None,
               fetcher: DataFetcher,
               contributions: RiskContributions | None = None
               ) -> list[ConstituentRow]:
    """One row per constituent, heaviest first.

    Args:
        snapshot: The rebalance in force.
        held: Drifted weights, or None when nothing has drifted yet.
        as_of: The date shares outstanding are read at.
        fetcher: Data source.

    Returns:
        list: Rows ordered by applied weight, descending. A weights table is
        read from the top, so the order that matters is the one the reader
        cares about rather than the order the store happened to hold.
    """
    capped = set(snapshot.capped)
    shares = _shares_for(fetcher, sorted(snapshot.weights),
                         as_of or snapshot.date)

    rows = [
        ConstituentRow(
            identifier=identifier,
            weight=weight,
            # Falls back to the applied weight rather than to zero: a run
            # stored before uncapped weights were carried has no raw figure,
            # and zero would render as "the cap took everything".
            raw_weight=snapshot.uncapped_weights.get(identifier, weight),
            capped=identifier in capped,
            shares_outstanding=shares.get(identifier),
            delta_since_rebalance=(None if held is None
                                   else held.get(identifier, 0.0) - weight),
            # Null rather than zero for an uncovered name: zero would read as
            # "contributes no risk", which is a claim, where null says the
            # model could not answer.
            risk_contribution=(contributions.contribution.get(identifier)
                               if contributions else None))
        for identifier, weight in snapshot.weights.items()]

    return sorted(rows, key=lambda row: row.weight, reverse=True)


def _shares_for(fetcher: DataFetcher,
                identifiers: list[str],
                date: str) -> dict[str, float]:
    """Shares outstanding per name on a date.

    A name the dataset has no share count for is simply absent, so its row
    reports null. Shares outstanding is optional market data — an index built
    on equal weights never needs it — and turning its absence into an error
    would make the whole pane fail for a column nothing depends on.
    """
    shares: dict[str, float] = {}

    for identifier in identifiers:
        value = fetcher.fetch_shares_outstanding(identifier, date)
        if value is not None:
            shares[identifier] = float(value)

    if not shares:
        logger.debug("No shares outstanding held for any of %d constituent(s).",
                     len(identifiers))

    return shares


def prices_for(fetcher: DataFetcher,
                identifiers: list[str],
                start: str | None,
                end: str | None) -> pd.DataFrame:
    """Close prices for a set of names over a window, names on the columns."""
    series: dict[str, pd.Series] = {}

    for identifier in identifiers:
        frame = fetcher.fetch_market_data(identifier, start, end)
        if not frame.empty and "CLOSE" in frame.columns:
            series[identifier] = frame["CLOSE"]

    if not series:
        return pd.DataFrame()

    return pd.DataFrame(series).sort_index()
