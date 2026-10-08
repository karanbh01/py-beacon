# src/beacon/strategy/tracking.py
"""
Index tracking: holding an index, in full or through a subset.

    Backtest(initial_capital=1e8).run(
        IndexTracking(my_index, replication=OptimisedReplication(holdings=50)),
        start="2023-01-03", end="2024-12-31")

A bare index definition passed to `run()` is tracked in full. `IndexTracking`
says how else to hold it. The index itself is still calculated and is what
the run is measured against; at each rebalance the replication turns the
index's weights into the weights the portfolio trades to.

- `FullReplication`: every constituent at its index weight.
- `OptimisedReplication`: a subset weighted by the optimiser to minimise
  tracking error.
- `SampledReplication`: the largest names in each sector and size cell, each
  cell at its index weight.

**Optimised.** At each rebalance the covariance of the constituents is
estimated from their trailing daily returns in the book's currency (a year
by default), shrunk toward constant correlation, and the optimiser finds the
long-only weights closest to the index in tracking error, within a holdings
limit and any other constraints. A name with too little history to measure
is held at its index weight, outside the optimisation.

**Sampled** (stratified). The constituents are split into cells by sector and
by size (terciles of market cap by default). Each cell is given holdings in
proportion to its index weight, at least one where the limit allows, holds
its largest names by index weight, and is scaled to its index weight. A
limit smaller than the number of cells keeps the heaviest cells, and the
others' weight is spread across them.
"""
# BN-269, phase 7 of decisions/0006 (first half; active strategies are
# BN-280). Decided with Karan on 2026-10-08: rolling shrunk covariance by
# default, stratified sampling by sector and size.
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from ..data.fetcher import DataFetcher
from ..expressions import data
from ..expressions.resolve import value_of
from ..index.derived import AnyIndexDefinition
from ..optimise.constraints import Cardinality, Constraint, FullInvestment, PositionBounds
from ..optimise.solver import minimise_tracking_error
from ..risk.model import CONSTANT_CORRELATION, estimate_risk_model

logger = logging.getLogger(__name__)

# Trading days a year of covariance is estimated over, and the fewest a name
# needs to be measured at all.
DEFAULT_LOOKBACK_DAYS = 252
MINIMUM_OBSERVATIONS = 126

# What a name with no classification is grouped under.
UNCLASSIFIED = "Unclassified"


@dataclass(frozen=True)
class ReplicationContext:
    """What a replication may read at a rebalance.

    Attributes:
        fetcher: The run's data.
        currency: The book's currency, which returns are measured in.
    """
    fetcher: DataFetcher
    currency: str


@dataclass(frozen=True)
class ReplicationStep:
    """What a replication did at one rebalance.

    Attributes:
        date: The rebalance date.
        weights: The weights the portfolio traded to.
        holdings: How many names they hold.
        tracking_error: The ex-ante annualised tracking error to the index,
            for an optimised replication; None when it was not estimated.
        unmeasured: Names held at their index weight because their history
            was too short to measure.
    """
    date: pd.Timestamp
    weights: dict[str, float]
    holdings: int
    tracking_error: float | None = None
    unmeasured: tuple[str, ...] = field(default_factory=tuple)


class Replication(ABC):
    """How an index is held: from its weights at a rebalance to the
    portfolio's."""

    @property
    def name(self) -> str:
        """How the run's record names this replication."""
        return type(self).__name__

    @abstractmethod
    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: ReplicationContext) -> ReplicationStep:
        """The portfolio's weights for the index's *target* on *date*."""


class FullReplication(Replication):
    """Every constituent at its index weight."""

    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: ReplicationContext) -> ReplicationStep:
        held = {name: weight for name, weight in target.items() if weight > 0}

        return ReplicationStep(date=date, weights=held, holdings=len(held),
                               tracking_error=0.0)

    def __repr__(self) -> str:
        return "FullReplication()"


class OptimisedReplication(Replication):
    """A subset weighted to minimise tracking error to the index.

    Args:
        holdings: The most names to hold, or None for no limit.
        constraints: Further optimiser constraints, such as `GroupBounds`.
            Long-only bounds and full investment are always applied.
        lookback_days: Trading days of returns the covariance is estimated
            over.
        minimum_observations: The fewest returns a name needs to be
            measured; a name with fewer is held at its index weight.

    Raises:
        ValueError: If *holdings* is below 1 or the windows are too short.
    """

    def __init__(self,
                 holdings: int | None = None,
                 constraints: Sequence[Constraint] = (),
                 lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                 minimum_observations: int = MINIMUM_OBSERVATIONS):
        if holdings is not None and holdings < 1:
            raise ValueError(f"holdings must be at least 1, got {holdings!r}.")

        if minimum_observations < 2 or lookback_days < minimum_observations:
            raise ValueError("lookback_days must be at least minimum_observations, "
                             "and that at least 2.")

        self.holdings = holdings
        self.constraints: tuple[Constraint, ...] = tuple(constraints)
        self.lookback_days = lookback_days
        self.minimum_observations = minimum_observations

    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: ReplicationContext) -> ReplicationStep:
        target = {name: weight for name, weight in target.items() if weight > 0}
        returns = _trailing_returns(list(target), date, context, self.lookback_days)
        measured = [name for name in target
                    if name in returns and returns[name].count() >= self.minimum_observations]
        unmeasured = {name: target[name] for name in target if name not in measured}

        if len(measured) < 2:
            logger.warning("[%s] Too few names with enough history to optimise; "
                           "the index is held in full.", date.date())

            return FullReplication().replicate(target, date, context)

        panel = returns[measured].dropna()
        risk = estimate_risk_model(panel, target=CONSTANT_CORRELATION)
        solved = minimise_tracking_error({name: target[name] for name in measured},
                                         constraints=self._constraints(target, measured,
                                                                       unmeasured),
                                         risk_model=risk)
        weights = {**{name: float(weight) for name, weight in solved.weights.items()
                      if weight > 1e-9},
                   **unmeasured}

        return ReplicationStep(date=date, weights=weights, holdings=len(weights),
                               tracking_error=solved.tracking_error(),
                               unmeasured=tuple(sorted(unmeasured)))

    def _constraints(self,
                     target: dict[str, float],
                     measured: list[str],
                     unmeasured: dict[str, float]) -> list[Constraint]:
        """Long-only, invested as the measured names are in the index, within
        what the holdings limit leaves after the unmeasured names."""
        rules: list[Constraint] = [
            FullInvestment(sum(target[name] for name in measured)),
            PositionBounds(minimum=0.0, maximum=1.0),
            *self.constraints]

        if self.holdings is not None:
            rules.append(Cardinality(max(self.holdings - len(unmeasured), 1)))

        return rules

    def __repr__(self) -> str:
        return (f"OptimisedReplication(holdings={self.holdings!r}, "
                f"lookback_days={self.lookback_days!r})")


class SampledReplication(Replication):
    """The largest names in each sector and size cell, each cell at its
    index weight.

    Args:
        holdings: How many names to hold.
        size_buckets: How many size groups (by market cap) each sector is
            split into: 3 for terciles.
        scheme: The classification the sectors are read from.

    Raises:
        ValueError: If *holdings* or *size_buckets* is below 1.
    """

    def __init__(self,
                 holdings: int,
                 size_buckets: int = 3,
                 scheme: str = "SECTOR"):
        if holdings < 1 or size_buckets < 1:
            raise ValueError("SampledReplication needs at least one holding "
                             "and one size bucket.")

        self.holdings = holdings
        self.size_buckets = size_buckets
        self.scheme = scheme

    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: ReplicationContext) -> ReplicationStep:
        target = {name: weight for name, weight in target.items() if weight > 0}

        if self.holdings >= len(target):
            return FullReplication().replicate(target, date, context)

        cells = self._cells(target, date, context)
        quotas = _quotas({cell: sum(target[name] for name in names)
                          for cell, names in cells.items()},
                         {cell: len(names) for cell, names in cells.items()},
                         self.holdings)
        weights: dict[str, float] = {}

        for cell, count in quotas.items():
            chosen = sorted(cells[cell], key=lambda name: target[name],
                            reverse=True)[:count]
            weights.update(_scaled({name: target[name] for name in chosen},
                                   sum(target[name] for name in cells[cell])))

        weights = _scaled(weights, sum(target.values()))

        return ReplicationStep(date=date, weights=weights, holdings=len(weights))

    def _cells(self,
               target: dict[str, float],
               date: pd.Timestamp,
               context: ReplicationContext) -> dict[tuple[str, int], list[str]]:
        """The constituents by sector and size bucket."""
        names = list(target)
        sectors = context.fetcher.fetch_classifications(names, date, self.scheme)
        caps = {name: value_of(data.market.market_cap, name, date, context.fetcher,
                               currency=context.currency) for name in names}
        ranked = sorted(names, key=lambda name: (caps[name] is not None,
                                                 caps[name] or 0.0, target[name]))
        buckets = min(self.size_buckets, len(names))
        cells: dict[tuple[str, int], list[str]] = {}

        for position, name in enumerate(ranked):
            bucket = position * buckets // len(ranked)
            sector = sectors.get(name) or UNCLASSIFIED
            cells.setdefault((sector, bucket), []).append(name)

        return cells

    def __repr__(self) -> str:
        return (f"SampledReplication(holdings={self.holdings!r}, "
                f"size_buckets={self.size_buckets!r}, scheme={self.scheme!r})")


class IndexTracking:
    """A strategy that holds an index.

    Args:
        index: The index to track.
        replication: How it is held; in full by default.
    """

    def __init__(self,
                 index: AnyIndexDefinition,
                 replication: Replication | None = None):
        self.index = index
        self.replication: Replication = (replication if replication is not None
                                         else FullReplication())

    def replicated(self,
                   snapshots: dict[pd.Timestamp, dict[str, float]],
                   context: ReplicationContext) -> list[ReplicationStep]:
        """The replication at each of the index's rebalances, in date order."""
        return [self.replication.replicate(dict(weights), date, context)
                for date, weights in sorted(snapshots.items())]

    def __repr__(self) -> str:
        return (f"IndexTracking(index={self.index.index_id!r}, "
                f"replication={self.replication!r})")


def _trailing_returns(names: list[str],
                      date: pd.Timestamp,
                      context: ReplicationContext,
                      lookback_days: int) -> pd.DataFrame:
    """Daily returns in the book's currency over the trading days before and
    including *date*, one column per name."""
    start = date - pd.Timedelta(days=int(lookback_days * 1.6) + 10)
    prices = context.fetcher.fetch_prices(names, start.strftime("%Y-%m-%d"),
                                          date.strftime("%Y-%m-%d"),
                                          currency=context.currency)

    if prices.empty:
        return pd.DataFrame()

    # A missed bar carries the last close rather than ending a name's history.
    return prices.ffill().pct_change(fill_method=None).iloc[1:].tail(lookback_days)


def _quotas(weights: dict[tuple[str, int], float],
            sizes: dict[tuple[str, int], int],
            holdings: int) -> dict[tuple[str, int], int]:
    """How many names each cell holds: its share of *holdings* by weight,
    at least one and no more than it has, rounded by largest remainder. A
    limit below the number of cells keeps the heaviest."""
    kept = sorted(weights, key=lambda cell: weights[cell], reverse=True)[:holdings]
    total = sum(weights[cell] for cell in kept)
    share = {cell: holdings * weights[cell] / total if total > 0 else 1.0
             for cell in kept}
    quotas = {cell: min(max(1, math.floor(share[cell])), sizes[cell]) for cell in kept}
    spare = holdings - sum(quotas.values())

    # The floor of one can over-allocate: take back from the furthest above.
    while spare < 0:
        cell = max((cell for cell in kept if quotas[cell] > 1),
                   key=lambda cell: quotas[cell] - share[cell])
        quotas[cell] -= 1
        spare += 1

    while spare > 0:
        room = [cell for cell in kept if quotas[cell] < sizes[cell]]

        if not room:
            break

        cell = max(room, key=lambda cell: share[cell] - quotas[cell])
        quotas[cell] += 1
        spare -= 1

    return quotas


def _scaled(weights: dict[str, float],
            total: float) -> dict[str, float]:
    """*weights* scaled to add up to *total*."""
    current = sum(weights.values())

    if current <= 0.0:
        return weights

    return {name: weight * total / current for name, weight in weights.items()}
