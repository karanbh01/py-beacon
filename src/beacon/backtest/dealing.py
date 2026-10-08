# src/beacon/backtest/dealing.py
"""
How an open-ended fund prices the units investors buy and sell.

    Vehicle(pricing=SwingPricing(threshold=0.02))

When investors buy or sell, the fund trades to invest or raise the money, and
that costs money. Left alone, the cost comes out of the fund, so the investors
who stay pay for the ones who came or went: **dilution**. A pricing method
decides who pays. NAV per unit is calculated first; the method then sets the
price units are created or cancelled at, or adds a charge, and whatever the
dealing investor pays above NAV, or receives below it, stays in the fund.

- `SinglePricing`: everyone deals at NAV, and the fund bears the trading
  cost.
- `DualPricing`: buyers pay NAV plus an offer adjustment, sellers receive NAV
  less a bid adjustment.
- `SwingPricing`: the price swings up on a net inflow and down on a net
  outflow, when the flow is larger than a threshold.
- `DilutionLevy`: deals are at NAV, and a deal larger than a threshold pays
  a charge into the fund.

An adjustment, swing factor or levy rate left unset is **estimated** from the
run's cost model: the fixed cost plus the market impact of trading the whole
flow, as a share of it.

A run deals one net flow a day, so a dual price is charged to the day's net
buyers or net sellers rather than to each deal.
"""
# BN-268, phase 6 of decisions/0006. See docs/concepts/fund-vehicles.md.
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..catalogue import BPS, FRACTION, PRICING, Display, register

IN = "subscriptions"
OUT = "redemptions"
BOTH = "both"
LEVY_SIDES = (BOTH, IN, OUT)


@dataclass(frozen=True)
class Deal:
    """One day's flow, priced.

    Attributes:
        units: Units created (positive) or cancelled (negative).
        cash: Cash into the fund (positive) or out of it (negative).
        price: The price per unit the units were dealt at.
        adjustment: What the dealing investors paid into the fund, through
            the price or a levy, to cover the trading their flow causes.
    """
    units: float
    cash: float
    price: float
    adjustment: float


class Pricing(ABC):
    """How a fund prices its dealing."""

    @property
    def estimated(self) -> bool:
        """Whether the method needs the run's estimate of the flow's
        trading cost."""
        return False

    @abstractmethod
    def deal(self,
             amount: float,
             nav_per_unit: float,
             net_assets: float,
             cost_rate: float) -> Deal:
        """Price a flow.

        Args:
            amount: The flow at NAV, in the book's currency: positive for
                money paid in, negative for the value of units redeemed.
            nav_per_unit: Today's NAV per unit, before the flow.
            net_assets: The fund's net assets before the flow.
            cost_rate: The estimated cost of trading the flow, as a share of
                it, for a method that estimates.
        """


@register(PRICING, "Single pricing")
class SinglePricing(Pricing):
    """Everyone deals at NAV per unit."""

    def deal(self,
             amount: float,
             nav_per_unit: float,
             net_assets: float,
             cost_rate: float) -> Deal:
        return _adjusted(amount, nav_per_unit, 0.0)

    def __repr__(self) -> str:
        return "SinglePricing()"


@register(PRICING, "Dual pricing", fields={
    "offer_bps": Display("Offer adjustment", unit=BPS, minimum=0.0),
    "bid_bps": Display("Bid adjustment", unit=BPS, minimum=0.0),
})
class DualPricing(Pricing):
    """Buyers pay an offer price above NAV, sellers receive a bid price
    below it.

    Args:
        offer_bps: The offer price's premium over NAV, in basis points. None
            estimates it from the cost of buying the flow.
        bid_bps: The bid price's discount to NAV, in basis points. None
            estimates it from the cost of selling.

    Raises:
        ValueError: If either is negative.
    """

    def __init__(self,
                 offer_bps: float | None = None,
                 bid_bps: float | None = None):
        _not_negative(offer_bps=offer_bps, bid_bps=bid_bps)
        self.offer_bps = offer_bps
        self.bid_bps = bid_bps

    @property
    def estimated(self) -> bool:
        return self.offer_bps is None or self.bid_bps is None

    def deal(self,
             amount: float,
             nav_per_unit: float,
             net_assets: float,
             cost_rate: float) -> Deal:
        side = self.offer_bps if amount > 0 else self.bid_bps

        return _adjusted(amount, nav_per_unit, _rate(side, cost_rate))

    def __repr__(self) -> str:
        return f"DualPricing(offer_bps={self.offer_bps!r}, bid_bps={self.bid_bps!r})"


@register(PRICING, "Swing pricing", fields={
    "factor_bps": Display("Swing factor", unit=BPS, minimum=0.0),
    "threshold": Display("Swing above this flow", unit=FRACTION, minimum=0.0, maximum=1.0),
})
class SwingPricing(Pricing):
    """A single price that swings with the day's net flow.

    Args:
        factor_bps: How far the price swings, in basis points. None estimates
            it from the cost of trading the flow.
        threshold: The net flow, as a share of the fund, above which the
            price swings. 0 (the default) swings on every flow: full swing.

    Raises:
        ValueError: If either is negative.
    """

    def __init__(self,
                 factor_bps: float | None = None,
                 threshold: float = 0.0):
        _not_negative(factor_bps=factor_bps, threshold=threshold)
        self.factor_bps = factor_bps
        self.threshold = threshold

    @property
    def estimated(self) -> bool:
        return self.factor_bps is None

    def deal(self,
             amount: float,
             nav_per_unit: float,
             net_assets: float,
             cost_rate: float) -> Deal:
        if not _above(amount, net_assets, self.threshold):
            return _adjusted(amount, nav_per_unit, 0.0)

        return _adjusted(amount, nav_per_unit, _rate(self.factor_bps, cost_rate))

    def __repr__(self) -> str:
        return (f"SwingPricing(factor_bps={self.factor_bps!r}, "
                f"threshold={self.threshold!r})")


@register(PRICING, "Dilution levy", fields={
    "rate_bps": Display("Levy", unit=BPS, minimum=0.0),
    "threshold": Display("Charge above this deal", unit=FRACTION, minimum=0.0, maximum=1.0),
    "on": Display("Applies to", choices=LEVY_SIDES),
})
class DilutionLevy(Pricing):
    """A charge into the fund on deals larger than a threshold.

    Args:
        rate_bps: The levy, in basis points of the deal. None estimates it
            from the cost of trading the flow.
        threshold: The deal, as a share of the fund, above which it is
            charged. 0 (the default) charges every deal.
        on: Which deals: ``"both"`` (the default), ``"subscriptions"`` or
            ``"redemptions"``. A redemption fee is a levy on redemptions.

    Raises:
        ValueError: If *rate_bps* or *threshold* is negative, or *on* is not
            one of the three.
    """

    def __init__(self,
                 rate_bps: float | None = None,
                 threshold: float = 0.0,
                 on: str = BOTH):
        _not_negative(rate_bps=rate_bps, threshold=threshold)

        if on not in LEVY_SIDES:
            raise ValueError(f"Unknown side {on!r}. Supported: "
                             f"{', '.join(LEVY_SIDES)}.")

        self.rate_bps = rate_bps
        self.threshold = threshold
        self.on = on

    @property
    def estimated(self) -> bool:
        return self.rate_bps is None

    def deal(self,
             amount: float,
             nav_per_unit: float,
             net_assets: float,
             cost_rate: float) -> Deal:
        side = IN if amount > 0 else OUT
        charged = self.on in (BOTH, side) and _above(amount, net_assets,
                                                     self.threshold)
        levy = abs(amount) * _rate(self.rate_bps, cost_rate) if charged else 0.0

        if amount > 0:
            return Deal(units=(amount - levy) / nav_per_unit, cash=amount,
                        price=nav_per_unit, adjustment=levy)

        return Deal(units=amount / nav_per_unit, cash=amount + levy,
                    price=nav_per_unit, adjustment=levy)

    def __repr__(self) -> str:
        return (f"DilutionLevy(rate_bps={self.rate_bps!r}, "
                f"threshold={self.threshold!r}, on={self.on!r})")


def _adjusted(amount: float,
              nav_per_unit: float,
              rate: float) -> Deal:
    """A deal at NAV moved by *rate*: up for money in, down for money out."""
    if amount > 0:
        price = nav_per_unit * (1.0 + rate)
        units = amount / price

        return Deal(units=units, cash=amount, price=price,
                    adjustment=amount - units * nav_per_unit)

    price = nav_per_unit * (1.0 - rate)
    units = amount / nav_per_unit

    return Deal(units=units, cash=units * price, price=price,
                adjustment=-amount * rate)


def _rate(bps: float | None,
          estimate: float) -> float:
    """A rate from basis points, or the estimate when unset."""
    return estimate if bps is None else bps / 10_000.0


def _above(amount: float,
           net_assets: float,
           threshold: float) -> bool:
    """Whether a flow is larger than *threshold* of the fund."""
    if threshold <= 0.0:
        return amount != 0.0

    return net_assets > 0.0 and abs(amount) > threshold * net_assets


def _not_negative(**values: float | None) -> None:
    for name, value in values.items():
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative, got {value!r}.")
