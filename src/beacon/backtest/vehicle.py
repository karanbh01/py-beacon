# src/beacon/backtest/vehicle.py
"""
How a backtest's money is held: the vehicle.

    Backtest(initial_capital=1e8, vehicle=Vehicle(management_fee_bps=20))

A vehicle says nothing about what is held (the strategy) or what money
arrives (the flows). It carries what the fund itself charges, how it prices
the units investors deal in, the diversification limits its structure
imposes, and its unit price at launch. A preset (see
`beacon.backtest.presets`) fills these in for a real fund structure.

The management fee accrues every calendar day on the fund's net assets
(ACT/365): ``net assets x fee / 10,000 x days / 365``, owed as a liability
that the NAV is net of and paid from cash as soon as there is cash to pay it.
"""
# BN-267: the minimal vehicle phase 5 needs for the fee. Pricing and
# diversification limits are BN-268, with the presets.
from collections.abc import Iterable

from .dealing import Pricing, SinglePricing
from .limits import DiversificationLimit

# NAV per unit at launch when the run sets none.
DEFAULT_LAUNCH_PRICE = 1.0


class Vehicle:
    """How a fund holds its money.

    Args:
        management_fee_bps: The annual management fee in basis points, such
            as 20 for 0.20%. Accrued daily on net assets, ACT/365.
        launch_price: NAV per unit when the fund launches, which sets how
            many units the initial capital buys.
        pricing: How the fund prices dealing (see `beacon.backtest.dealing`).
            Single pricing by default.
        limits: The structure's diversification limits (see
            `beacon.backtest.limits`), applied at each rebalance with the
            capacity caps.
        name: What the vehicle is called, such as a preset's name.

    Raises:
        ValueError: If the fee is negative or the launch price not positive.
    """

    def __init__(self,
                 management_fee_bps: float = 0.0,
                 launch_price: float = DEFAULT_LAUNCH_PRICE,
                 pricing: Pricing | None = None,
                 limits: Iterable[DiversificationLimit] = (),
                 name: str = "Vehicle"):
        if management_fee_bps < 0:
            raise ValueError(f"management_fee_bps cannot be negative, got "
                             f"{management_fee_bps!r}.")

        if launch_price <= 0:
            raise ValueError(f"launch_price must be positive, got "
                             f"{launch_price!r}.")

        self.management_fee_bps = management_fee_bps
        self.launch_price = launch_price
        self.pricing: Pricing = pricing if pricing is not None else SinglePricing()
        self.limits: tuple[DiversificationLimit, ...] = tuple(limits)
        self.name = name

    def daily_fee(self,
                  net_assets: float,
                  days: int) -> float:
        """The fee *net_assets* accrue over *days* calendar days."""
        if days <= 0 or net_assets <= 0.0:
            return 0.0

        return net_assets * self.management_fee_bps / 10_000.0 * days / 365.0

    def __repr__(self) -> str:
        return (f"Vehicle(name={self.name!r}, "
                f"management_fee_bps={self.management_fee_bps!r}, "
                f"pricing={self.pricing!r}, limits={list(self.limits)!r})")
