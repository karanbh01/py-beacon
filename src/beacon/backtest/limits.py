# src/beacon/backtest/limits.py
"""
A fund structure's diversification limits, applied as capacity limits.

    Vehicle(limits=[UcitsLimits()])

A structure's rules on how concentrated a fund may be are applied at each
rebalance with the capacity caps, so a fund near a limit holds less of a name
rather than breaking it. The weight a limit removes is redistributed as the
implementation says, until no limit is broken.

- **UCITS**: at most 10% in one issuer, and the holdings above 5% may add up
  to at most 40%. A fund replicating an index may instead hold up to 20% of
  one issuer, or 35% of the single largest.
- **1940 Act diversified**: for 75% of the fund, at most 5% in one issuer and
  at most 10% of an issuer's voting shares. Applied as: the holdings above 5%
  may add up to at most 25%, and no holding is more than 10% of the company's
  shares outstanding.

Each name is treated as its own issuer.
"""
# BN-268, phase 6 of decisions/0006.
from abc import ABC, abstractmethod

from .capacity import CapacityCap, OwnershipCap
from .redistribution import TOLERANCE, water_filled


class DiversificationLimit(ABC):
    """A limit on how concentrated a fund may be."""

    @property
    def name(self) -> str:
        """How the run's record names this limit."""
        return type(self).__name__

    def caps(self) -> list[CapacityCap]:
        """Per-name capacity caps the limit adds, such as a share of a
        company's shares outstanding."""
        return []

    @abstractmethod
    def limited(self,
                weights: dict[str, float],
                rule: str,
                capped: dict[str, float]) -> dict[str, float]:
        """*weights* within the limit, redistributed by *rule*. Records each
        name cut, and the weight it was cut to, in *capped*."""


class _LargeHoldingsLimit(DiversificationLimit):
    """At most *issuer_max* in one name, and the names above *large* adding
    up to at most *large_total*.

    When the large names add up to too much, the largest are kept as they
    are, as many as fit under the total, and the rest are cut to *large*.
    """

    def __init__(self,
                 issuer_max: float,
                 large: float,
                 large_total: float):
        self.issuer_max = issuer_max
        self.large = large
        self.large_total = large_total

    def limited(self,
                weights: dict[str, float],
                rule: str,
                capped: dict[str, float]) -> dict[str, float]:
        limits = dict.fromkeys(weights, self.issuer_max)

        # Each pass cuts at least one more name to the large threshold.
        for _ in range(len(weights) + 1):
            weights = water_filled(weights, limits, rule, capped)
            large = sorted((name for name, weight in weights.items()
                            if weight > self.large + TOLERANCE),
                           key=lambda name: weights[name], reverse=True)

            if sum(weights[name] for name in large) <= self.large_total + TOLERANCE:
                break

            kept = 0.0

            for name in large:
                if kept + weights[name] <= self.large_total + TOLERANCE:
                    kept += weights[name]
                else:
                    limits[name] = self.large

            # A name cut to the threshold must not climb back above it.
            for name in weights:
                if name not in large:
                    limits[name] = min(limits[name], self.large)

        return weights


class UcitsLimits(DiversificationLimit):
    """The UCITS diversification rules.

    Args:
        index_tracking: Apply the rules for a fund replicating an index (20%
            in one issuer, 35% in the largest) rather than the standard 5/10/40.
    """

    def __init__(self,
                 index_tracking: bool = True):
        self.index_tracking = index_tracking

    def limited(self,
                weights: dict[str, float],
                rule: str,
                capped: dict[str, float]) -> dict[str, float]:
        if not self.index_tracking:
            return _LargeHoldingsLimit(0.10, 0.05, 0.40).limited(weights, rule,
                                                                 capped)

        if not weights:
            return weights

        largest = max(weights, key=lambda name: weights[name])
        limits = {name: (0.35 if name == largest else 0.20) for name in weights}

        return water_filled(weights, limits, rule, capped)

    def __repr__(self) -> str:
        return f"UcitsLimits(index_tracking={self.index_tracking!r})"


class Act1940Limits(DiversificationLimit):
    """The US Investment Company Act's test for a diversified fund."""

    def caps(self) -> list[CapacityCap]:
        return [OwnershipCap(0.10, free_float=False)]

    def limited(self,
                weights: dict[str, float],
                rule: str,
                capped: dict[str, float]) -> dict[str, float]:
        return _LargeHoldingsLimit(1.0, 0.05, 0.25).limited(weights, rule, capped)

    def __repr__(self) -> str:
        return "Act1940Limits()"
