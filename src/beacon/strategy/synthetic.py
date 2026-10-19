# src/beacon/strategy/synthetic.py
"""
Synthetic replication: a substitute basket held, its return swapped for the
index's.

    IndexTracking(my_index,
                  replication=SyntheticReplication(basket=other_index,
                                                   spread_bps=15.0))

The fund buys a basket of securities that need not be the index's, and holds
a total return swap with a counterparty: on the swap's notional it pays the
basket's return and the spread, and is paid the index's return. What the
basket earns goes out as fast as it comes in, so the fund earns the index's
return less the spread.

The basket still shapes the result:

- **Resets.** Between resets the swap is worth the index's performance less
  the basket's. A basket unlike the index moves that value further, so the
  swap reaches its exposure limit sooner and is reset more often, and each
  reset trades the basket.
- **Counterparty exposure.** The swap's value owed to the fund is what the
  fund would lose if the counterparty failed. It is capped: the swap is reset
  when it reaches *exposure_limit* of the fund's net assets (10% under UCITS).
- **Dividend tax.** The fund receives the basket's dividends net of its own
  withholding and passes on only what it received, while the swap pays the
  index's return as the index measures it (price, total or net total return).
- **Costs.** The basket is bought and sold with the run's costs, and the
  spread accrues on the notional every calendar day (ACT/360).

**The basket is required.** It is an index definition, calculated over the
run and held at its own weights, or a fixed set of weights. Real swap-based
funds hold whatever liquid basket the counterparty proposes within their
collateral rules, rarely the index itself.

**Resets.** The swap is reset (its value paid in cash, and the notional set
back to the fund's net assets) when the fund's exposure reaches the limit, at
each of the index's rebalances, and on any day money flows in or out. A reset
on a rebalance settles before the rebalance trades, so the basket is bought
or sold once.
"""
# BN-270, phase 8 of decisions/0006. Decided with Karan on 2026-10-11: the
# basket is required (not defaulted to the index), and the swap resets at the
# exposure limit and at each rebalance.
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from ..catalogue import FRACTION, REPLICATION, Display, register
from ..index.derived import AnyIndexDefinition
from ..index.result import IndexResult
from .base import StrategyContext
from .tracking import Replication, ReplicationStep

# The most of a UCITS fund's net assets its exposure to one OTC derivative
# counterparty may reach when the counterparty is a credit institution.
UCITS_EXPOSURE_LIMIT = 0.10


@dataclass(frozen=True)
class SwapTerms:
    """The total return swap a synthetic fund holds.

    Attributes:
        spread_bps: The spread the fund pays on the notional, in basis points
            a year, accrued over calendar days (ACT/360).
        exposure_limit: The swap's value owed to the fund, as a share of its
            net assets, at which the swap is reset.
    """
    spread_bps: float = 0.0
    exposure_limit: float = UCITS_EXPOSURE_LIMIT


@register(REPLICATION, "Synthetic", fields={
    "basket": Display("Substitute basket", help="The basket the fund holds, as weights by "
                      "identifier; the swap pays the index's return for its return."),
    "spread_bps": Display("Swap spread"),
    "exposure_limit": Display("Counterparty exposure limit", unit=FRACTION,
                              minimum=0.0, maximum=1.0),
})
class SyntheticReplication(Replication):
    """A substitute basket held, its return swapped for the index's.

    Args:
        basket: What the fund holds: an index definition, calculated over
            the run and held at its weights, or fixed weights by identifier,
            which are scaled to add up to one.
        spread_bps: The spread the fund pays on the swap, in basis points a
            year.
        exposure_limit: The swap's value owed to the fund, as a share of its
            net assets, at which the swap is reset. 0.10 (UCITS) by default.

    Raises:
        ValueError: If the basket is empty or has a weight that is not
            positive, the spread is not a finite number, or the limit is not
            above 0 and at most 1.
    """

    # The shapes the editor renders these in (BN-330).
    PARAM_SCHEMAS: ClassVar[dict[str, str]] = {"basket": "IdentifierValues"}

    def __init__(self,
                 basket: AnyIndexDefinition | Mapping[str, float],
                 spread_bps: float = 0.0,
                 exposure_limit: float = UCITS_EXPOSURE_LIMIT):
        if not math.isfinite(spread_bps):
            raise ValueError(f"spread_bps must be a finite number, got {spread_bps!r}.")

        if not 0.0 < exposure_limit <= 1.0:
            raise ValueError(f"exposure_limit is a share of net assets above 0 "
                             f"and at most 1, got {exposure_limit!r}.")

        self.basket = basket
        self.terms = SwapTerms(spread_bps=spread_bps, exposure_limit=exposure_limit)
        self._weights = _fixed(basket) if isinstance(basket, Mapping) else None
        self._snapshots: dict[pd.Timestamp, dict[str, float]] | None = None

    @property
    def spread_bps(self) -> float:
        """The spread the fund pays on the swap, in basis points a year."""
        return self.terms.spread_bps

    @property
    def exposure_limit(self) -> float:
        """The share of net assets owed to the fund at which the swap resets."""
        return self.terms.exposure_limit

    @property
    def basket_index(self) -> AnyIndexDefinition | None:
        """The basket's index definition, or None for fixed weights."""
        return None if isinstance(self.basket, Mapping) else self.basket

    def with_basket(self,
                    calculated: IndexResult) -> "SyntheticReplication":
        """This replication with its basket index's calculation, whose weights
        it holds at each rebalance."""
        bound = SyntheticReplication(self.basket, self.terms.spread_bps,
                                     self.terms.exposure_limit)
        bound._snapshots = dict(calculated.weight_snapshots)

        return bound

    def replicate(self,
                  target: dict[str, float],
                  date: pd.Timestamp,
                  context: StrategyContext) -> ReplicationStep:
        weights = self._basket_on(date)

        return ReplicationStep(date=date, weights=weights, holdings=len(weights))

    def _basket_on(self,
                   date: pd.Timestamp) -> dict[str, float]:
        """The basket's weights in force on *date*."""
        if self._weights is not None:
            return dict(self._weights)

        if not self._snapshots:
            raise ValueError("the basket index has not been calculated; "
                             "Backtest.run calculates it before replicating.")

        earlier = [day for day in self._snapshots if day <= date]
        # A basket index based after the tracked one has no weights yet on
        # the first rebalance; its first snapshot is the closest it has.
        chosen = max(earlier) if earlier else min(self._snapshots)

        return {name: weight for name, weight in self._snapshots[chosen].items()
                if weight > 0}

    def __repr__(self) -> str:
        basket = (f"{len(self.basket)} names" if isinstance(self.basket, Mapping)
                  else self.basket.index_id)
        return (f"SyntheticReplication(basket={basket!r}, "
                f"spread_bps={self.terms.spread_bps!r}, "
                f"exposure_limit={self.terms.exposure_limit!r})")


def _fixed(weights: Mapping[str, float]) -> dict[str, float]:
    """Fixed basket weights, checked and scaled to add up to one."""
    if not weights:
        raise ValueError("the basket needs at least one name.")

    bad = [name for name, weight in weights.items()
           if not math.isfinite(weight) or weight <= 0.0]

    if bad:
        raise ValueError(f"basket weights must be positive: {', '.join(sorted(bad))}.")

    total = sum(weights.values())

    return {name: weight / total for name, weight in weights.items()}
