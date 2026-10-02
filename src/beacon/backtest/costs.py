# src/beacon/backtest/costs.py
"""
What trading costs beyond the fixed basis points, and how fast it can happen.

`MarketImpact` charges a trade for its size against what the name trades in a
day, by the square-root law. `ExecutionLimit` caps how much of an order can
trade in one day, by participation in the day's volume or by spreading it over
a number of days. Both are set on a backtest's `Implementation`; see
`beacon.backtest.execution` for how a rebalance uses them.
"""
# BN-266, phase 4 of decisions/0006.
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from .screens import (
    DEFAULT_LIQUIDITY_DAYS,
    ScreenContext,
    average_traded,
    converted_closes,
)


class MarketImpact:
    """The cost of a trade's size, by the square-root law.

    Args:
        coefficient: Scales the impact; around 1 is typical.
        lookback_days: Trading days in the volatility and the average traded
            value.

    Raises:
        ValueError: If *coefficient* is negative.
    """

    def __init__(self,
                 coefficient: float = 1.0,
                 lookback_days: int = DEFAULT_LIQUIDITY_DAYS):
        if coefficient < 0:
            raise ValueError(f"MarketImpact's coefficient cannot be negative, "
                             f"got {coefficient!r}.")

        self.coefficient = coefficient
        self.lookback_days = lookback_days

    def rate(self,
             asset_id: str,
             trade_value: float,
             date: pd.Timestamp,
             context: ScreenContext) -> float:
        """The impact of trading *trade_value* of *asset_id*, as a fraction
        of the value. 0 when the name's volatility or traded value is
        unknown."""
        traded = average_traded(asset_id, date, context, self.lookback_days)
        start = date - pd.Timedelta(days=int(self.lookback_days * 1.6) + 10)
        closes = converted_closes(asset_id, start, date, context)
        returns = closes.pct_change().dropna().tail(self.lookback_days)

        if traded is None or traded <= 0.0 or len(returns) < 2:
            return 0.0

        volatility = float(returns.std())

        return float(self.coefficient * volatility
                     * math.sqrt(max(trade_value, 0.0) / traded))


class ExecutionLimit:
    """How much of an order can trade in one day.

    Under a participation limit, a name with a volume of 0 on the day trades
    nothing. A blank volume is replaced by the last one reported, if no older
    than the run's `volume_backfill_days`, and otherwise by the name's
    average daily volume over *lookback_days*. A name with no volume at all
    is not limited by participation, and the run logs a warning.

    Args:
        participation: At most this share of the day's volume, as a decimal.
        days: Spread each order evenly over this many trading days.
        lookback_days: Trading days in the average daily volume that stands
            in for a blank one.

    Raises:
        ValueError: If neither is given, or either is out of range.
    """

    def __init__(self,
                 participation: float | None = None,
                 days: int | None = None,
                 lookback_days: int = DEFAULT_LIQUIDITY_DAYS):
        if participation is None and days is None:
            raise ValueError("ExecutionLimit needs a participation, a number "
                             "of days or both.")

        if participation is not None and not 0.0 < participation <= 1.0:
            raise ValueError(f"participation is a share of the day's volume "
                             f"above 0 and at most 1, got {participation!r}.")

        if days is not None and days < 1:
            raise ValueError(f"days must be at least 1, got {days!r}.")

        self.participation = participation
        self.days = days
        self.lookback_days = lookback_days

    def volume(self,
               asset_id: str,
               date: pd.Timestamp,
               fetcher: DataFetcher,
               backfill_days: int) -> float | None:
        """The shares *asset_id* is taken to trade on *date*: the day's
        volume, else the last reported within *backfill_days* calendar days,
        else the average daily volume. None when it has no volume at all."""
        start = date - pd.Timedelta(days=int(self.lookback_days * 1.6) + 10)
        frame = fetcher.fetch_market_data(asset_id, start.strftime("%Y-%m-%d"),
                                          date.strftime("%Y-%m-%d"))

        if frame.empty or "VOLUME" not in frame.columns:
            return None

        reported = frame["VOLUME"].dropna()
        reported = reported[reported.index <= date]

        if reported.empty:
            return None

        if (date - reported.index[-1]).days <= backfill_days:
            return float(np.asarray(reported)[-1])

        return float(reported.tail(self.lookback_days).mean())

    def allowed(self,
                order: "WorkingOrder",
                volume: float | None) -> float:
        """How many shares of *order* may trade on a day the name trades
        *volume* shares; None when its volume is unknown."""
        allowed = order.remaining

        if self.days is not None:
            allowed = min(allowed, order.quantity / self.days)

        if self.participation is not None and volume is not None:
            allowed = min(allowed, volume * self.participation)

        return max(allowed, 0.0)


@dataclass
class WorkingOrder:
    """An order still being worked over the sessions after a rebalance."""
    date: pd.Timestamp
    asset_id: str
    side: str
    quantity: float
    filled: float = 0.0

    @property
    def remaining(self) -> float:
        return self.quantity - self.filled

