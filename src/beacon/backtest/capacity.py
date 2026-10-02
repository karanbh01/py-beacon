# src/beacon/backtest/capacity.py
"""
Capacity: how much of each name a backtest may hold at its size.

    Implementation(caps=[OwnershipCap(0.05), LiquidityCap(days=5)],
                   minimum_position=MinimumPosition(value=50_000))

A cap limits a position; it does not exclude the name. Each cap gives the
largest value a position may have on a rebalance date, and the weight that
leaves at the book's size then is the name's cap. A name over its cap is cut
to it, and the excess is spread across the names under their caps in
proportion to their weights, repeated until none is over. As the fund grows,
a position shrinks smoothly rather than jumping to zero at a threshold.

Under ``redistribution="cash"`` the excess stays in cash instead, and so does
the weight of a position too small to keep.

A name a cap cannot value (no free float, no volume) is not limited by it.
"""
# BN-265, phase 3 of decisions/0006. Fund structures' diversification limits
# (UCITS 5/10/40, the 1940 Act's 75/5/10) arrive here with BN-268.
from abc import ABC, abstractmethod

import pandas as pd

from ..expressions import data
from ..expressions.resolve import value_of
from .screens import DEFAULT_LIQUIDITY_DAYS, ScreenContext, average_traded


class CapacityCap(ABC):
    """The largest value a position may have on a date."""

    @property
    def name(self) -> str:
        """How the run's record names this cap."""
        return type(self).__name__

    @abstractmethod
    def max_value(self,
                  asset_id: str,
                  date: pd.Timestamp,
                  book_value: float,
                  context: ScreenContext) -> float | None:
        """The largest position value in the book's currency, or None when
        this cap does not limit the name."""


class OwnershipCap(CapacityCap):
    """At most a share of a name's free-float market cap.

    Args:
        max_share: The share, as a decimal: 0.05 holds at most 5% of the
            free float.

    Raises:
        ValueError: If *max_share* is not above 0 and at most 1.
    """

    def __init__(self,
                 max_share: float):
        if not 0.0 < max_share <= 1.0:
            raise ValueError(f"OwnershipCap's max_share is a share of the free "
                             f"float above 0 and at most 1, got {max_share!r}.")

        self.max_share = max_share

    def max_value(self,
                  asset_id: str,
                  date: pd.Timestamp,
                  book_value: float,
                  context: ScreenContext) -> float | None:
        cap = value_of(data.market.free_float_market_cap, asset_id, date,
                       context.fetcher, currency=context.currency)

        return None if cap is None else float(cap) * self.max_share


class LiquidityCap(CapacityCap):
    """At most what could be sold over a number of days.

    The position's value is limited to *days* times the name's average daily
    traded value times *participation*, the share of each day's trading the
    fund could be.

    Args:
        days: Days to liquidate.
        participation: The share of a day's traded value, as a decimal.
        lookback_days: Trading days in the average.

    Raises:
        ValueError: If *days* or *participation* is not positive.
    """

    def __init__(self,
                 days: float,
                 participation: float = 0.2,
                 lookback_days: int = DEFAULT_LIQUIDITY_DAYS):
        if days <= 0 or not 0.0 < participation <= 1.0:
            raise ValueError("LiquidityCap needs days above 0 and a "
                             "participation above 0 and at most 1.")

        self.days = days
        self.participation = participation
        self.lookback_days = lookback_days

    def max_value(self,
                  asset_id: str,
                  date: pd.Timestamp,
                  book_value: float,
                  context: ScreenContext) -> float | None:
        traded = average_traded(asset_id, date, context, self.lookback_days)

        return None if traded is None else traded * self.days * self.participation


class WeightCap(CapacityCap):
    """At most a share of the book, whatever the name.

    Args:
        max_weight: The share, as a decimal.

    Raises:
        ValueError: If *max_weight* is not above 0 and at most 1.
    """

    def __init__(self,
                 max_weight: float):
        if not 0.0 < max_weight <= 1.0:
            raise ValueError(f"WeightCap's max_weight is above 0 and at most "
                             f"1, got {max_weight!r}.")

        self.max_weight = max_weight

    def max_value(self,
                  asset_id: str,
                  date: pd.Timestamp,
                  book_value: float,
                  context: ScreenContext) -> float | None:
        return self.max_weight * book_value


class MinimumPosition:
    """Positions too small to keep, by value or by weight.

    A position below either is dropped and its weight redistributed.

    Args:
        value: The smallest position value, in the book's currency.
        weight: The smallest weight, as a decimal.

    Raises:
        ValueError: If neither is given.
    """

    def __init__(self,
                 value: float | None = None,
                 weight: float | None = None):
        if value is None and weight is None:
            raise ValueError("MinimumPosition needs a value, a weight or both.")

        self.value = value
        self.weight = weight

    def too_small(self,
                  weight: float,
                  book_value: float) -> bool:
        """Whether a position of *weight* in a book of *book_value* is."""
        if self.weight is not None and weight < self.weight:
            return True

        return self.value is not None and weight * book_value < self.value
