# src/beacon/backtest/etf.py
"""
An exchange-traded fund: creations in units, and a market price on an
exchange.

    Backtest(initial_capital=5e8, vehicle=ucits_etf(), flows=...)

**The primary market.** Authorised participants (APs) create and redeem
shares with the fund in creation units, at NAV. The run's flows are their net
demand: each day it is rounded toward zero to whole creation units, and what
is left over carries to the next day. A cash creation delivers money, which
the fund invests, and a variable creation fee charges the AP the cost; an
in-kind creation delivers the holdings, so the fund does not trade.

**The secondary market.** Everyone else trades shares on an exchange. After
each close the run quotes them, as `EtfMarket` describes:

- The **premium** (or discount, when negative) persists from day to day, is
  pushed by the day's creations and by the NAV's return, wanders with seeded
  noise, and is held inside the **arbitrage band**: the cost of creating
  above NAV and of redeeming below it, which is the AP fee plus the cost of
  one creation unit's basket (or the creation fee, for cash).
- The **spread** is a floor plus the basket's trading cost and the NAV's
  recent volatility, and never less than one price tick.

```text
market price = NAV per unit x (1 + premium)
ask = market price x (1 + spread / 2),  bid = market price x (1 - spread / 2)
```

The basket's cost comes from the run's cost model: the fixed cost plus market
impact for one creation unit's worth of every holding, at each holding's
weight. With market impact on, it is refreshed at each rebalance and at the
start of each week rather than every day, which would read every holding's
history daily.
"""
# BN-268, phase 6 of decisions/0006. See docs/concepts/fund-vehicles.md.
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .dealing import DilutionLevy, Pricing, SinglePricing
from .vehicle import Vehicle

# Shares in a creation unit, and a share's price at launch, when unset.
DEFAULT_CREATION_UNIT = 50_000
ETF_LAUNCH_PRICE = 100.0


class EtfMarket:
    """How an ETF's shares are quoted on the exchange.

    Args:
        spread_floor_bps: The narrowest spread a market maker quotes, in
            basis points.
        basket_sensitivity: How much of the basket's trading cost the spread
            passes on: 1 adds the whole cost.
        volatility_sensitivity: How much the spread widens per unit of the
            NAV's daily volatility: 0.05 adds 5 basis points at 1% a day.
        tick: The smallest price step, in the book's currency.
        persistence: How much of yesterday's premium remains today, from 0
            up to 1: 0.5 halves it each day.
        flow_sensitivity: The premium per unit of the day's net creations
            as a share of the fund: 0.1 turns 1% of creations into a 10
            basis point premium.
        return_sensitivity: The premium per unit of the NAV's daily return,
            so the price leads NAV on large moves: 0.05 turns a 4% fall into
            a 20 basis point discount.
        noise_bps: The standard deviation of the premium's daily wander, in
            basis points.
        volatility_days: Simulated days the volatility is measured over.
        seed: The random seed, so a run repeats exactly.

    Raises:
        ValueError: If a setting is out of range.
    """

    def __init__(self,
                 spread_floor_bps: float = 2.0,
                 basket_sensitivity: float = 1.0,
                 volatility_sensitivity: float = 0.05,
                 tick: float = 0.01,
                 persistence: float = 0.5,
                 flow_sensitivity: float = 0.1,
                 return_sensitivity: float = 0.05,
                 noise_bps: float = 2.0,
                 volatility_days: int = 21,
                 seed: int = 0):
        if not 0.0 <= persistence < 1.0:
            raise ValueError(f"persistence is from 0 up to 1, got "
                             f"{persistence!r}.")

        if min(spread_floor_bps, basket_sensitivity, volatility_sensitivity,
               tick, noise_bps) < 0:
            raise ValueError("EtfMarket's spread floor, sensitivities, tick and "
                             "noise cannot be negative.")

        if volatility_days < 2:
            raise ValueError(f"volatility_days must be at least 2, got "
                             f"{volatility_days!r}.")

        self.spread_floor_bps = spread_floor_bps
        self.basket_sensitivity = basket_sensitivity
        self.volatility_sensitivity = volatility_sensitivity
        self.tick = tick
        self.persistence = persistence
        self.flow_sensitivity = flow_sensitivity
        self.return_sensitivity = return_sensitivity
        self.noise_bps = noise_bps
        self.volatility_days = volatility_days
        self.seed = seed


@dataclass(frozen=True)
class Quote:
    """One day's quote for an ETF's shares.

    Attributes:
        date: The day.
        nav_per_unit: NAV per share at the close.
        premium: The market price over NAV, less one; negative for a
            discount.
        spread: The full bid-ask spread, as a share of the market price.
        market_price: The mid price on the exchange.
        bid: What a seller receives.
        ask: What a buyer pays.
        create_cost: The arbitrage band's upper edge, as a share of NAV.
        redeem_cost: Its lower edge, as a share of NAV.
    """
    date: pd.Timestamp
    nav_per_unit: float
    premium: float
    spread: float
    market_price: float
    bid: float
    ask: float
    create_cost: float
    redeem_cost: float


class EtfVehicle(Vehicle):
    """A vehicle whose shares are created in units and traded on an exchange.

    Args:
        creation_unit: Shares in a creation unit.
        in_kind: Create and redeem by delivering the holdings, so the fund
            does not trade, rather than in cash.
        ap_fee: What an AP pays per creation or redemption, in the book's
            currency. It covers the fund's administration, so it is not paid
            into the fund, but it is part of the cost of arbitrage.
        creation_fee_bps: For cash creations, the variable fee charged to the
            AP and paid into the fund, in basis points. None estimates it
            from the cost of trading the basket.
        market: How the shares are quoted; `EtfMarket()` when None.
        **settings: Any other `Vehicle` setting. The launch price defaults to
            100 a share.

    Raises:
        ValueError: If *creation_unit* is below 1 or *ap_fee* is negative.
    """

    def __init__(self,
                 creation_unit: int = DEFAULT_CREATION_UNIT,
                 in_kind: bool = False,
                 ap_fee: float = 500.0,
                 creation_fee_bps: float | None = None,
                 market: EtfMarket | None = None,
                 **settings: Any):
        if creation_unit < 1 or ap_fee < 0:
            raise ValueError("An ETF needs a creation unit of at least one "
                             "share and an AP fee that is not negative.")

        settings.setdefault("launch_price", ETF_LAUNCH_PRICE)
        settings.setdefault("name", "ETF")
        settings.setdefault("pricing", _creation_pricing(in_kind, creation_fee_bps))
        super().__init__(**settings)

        self.creation_unit = creation_unit
        self.in_kind = in_kind
        self.ap_fee = ap_fee
        self.creation_fee_bps = creation_fee_bps
        self.market = market if market is not None else EtfMarket()

    def creation_units(self,
                       amount: float,
                       nav_per_unit: float) -> float:
        """How many whole creation units *amount* deals, toward zero."""
        value = self.creation_unit * nav_per_unit

        return float(math.trunc(amount / value)) if value > 0 else 0.0

    def __repr__(self) -> str:
        return (f"EtfVehicle(name={self.name!r}, "
                f"creation_unit={self.creation_unit!r}, "
                f"in_kind={self.in_kind!r}, ap_fee={self.ap_fee!r})")


class MarketMaker:
    """Quotes an ETF's shares each day, as its `EtfMarket` says."""

    def __init__(self,
                 vehicle: EtfVehicle):
        self.vehicle = vehicle
        self.market = vehicle.market
        self._random = np.random.default_rng(self.market.seed)
        self._premium = 0.0
        self._returns: list[float] = []

    def quote(self,
              date: pd.Timestamp,
              nav_per_unit: float,
              previous: float,
              flow_share: float,
              basket_cost: float) -> Quote:
        """The day's quote.

        Args:
            date: The day.
            nav_per_unit: NAV per share at the close.
            previous: NAV per share at the previous close.
            flow_share: The day's net creations as a share of the fund.
            basket_cost: The cost of trading one creation unit's basket, as a
                share of it.
        """
        market = self.market
        nav_return = nav_per_unit / previous - 1.0 if previous > 0 else 0.0
        self._returns = [*self._returns, nav_return][-market.volatility_days:]
        volatility = float(np.std(self._returns, ddof=1)) if len(self._returns) > 1 else 0.0

        create, redeem = self._band(nav_per_unit, basket_cost)
        premium = (market.persistence * self._premium
                   + market.flow_sensitivity * flow_share
                   + market.return_sensitivity * nav_return
                   + market.noise_bps / 10_000.0 * float(self._random.normal()))
        self._premium = min(max(premium, -redeem), create)

        price = nav_per_unit * (1.0 + self._premium)
        spread = max(market.spread_floor_bps / 10_000.0
                     + market.basket_sensitivity * basket_cost
                     + market.volatility_sensitivity * volatility,
                     market.tick / price if price > 0 else 0.0)

        return Quote(date=date, nav_per_unit=nav_per_unit, premium=self._premium,
                     spread=spread, market_price=price,
                     bid=price * (1.0 - spread / 2.0),
                     ask=price * (1.0 + spread / 2.0),
                     create_cost=create, redeem_cost=redeem)

    def _band(self,
              nav_per_unit: float,
              basket_cost: float) -> tuple[float, float]:
        """The cost of creating and of redeeming, as shares of NAV."""
        vehicle = self.vehicle
        unit_value = vehicle.creation_unit * nav_per_unit
        ap = vehicle.ap_fee / unit_value if unit_value > 0 else 0.0

        # In kind, the AP trades the basket itself; in cash, it pays the fee.
        fee = vehicle.creation_fee_bps
        trading = (basket_cost if vehicle.in_kind or fee is None
                   else fee / 10_000.0)

        return ap + trading, ap + trading


def quotes_frame(quotes: list[Quote]) -> pd.DataFrame:
    """The quotes as a frame indexed by date."""
    if not quotes:
        return pd.DataFrame(columns=["nav_per_unit", "premium", "spread",
                                     "market_price", "bid", "ask",
                                     "create_cost", "redeem_cost"])

    frame = pd.DataFrame([quote.__dict__ for quote in quotes])

    return frame.set_index("date")


def _creation_pricing(in_kind: bool,
                      creation_fee_bps: float | None) -> Pricing:
    """What an AP pays into the fund when it creates or redeems."""
    if in_kind:
        return SinglePricing()

    return DilutionLevy(rate_bps=creation_fee_bps)
