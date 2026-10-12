# src/beacon/backtest/share_classes.py
"""
Share classes: several classes of one fund, sharing one pool of assets.

    Backtest(initial_capital={"Acc": 6e7, "Dist": 4e7},
             share_classes=[ShareClass("Acc"),
                            ShareClass("Dist", distribution="distributing")],
             flows={"Dist": PeriodicFlows(fraction=0.01)},
             vehicle=uk_oeic(management_fee_bps=50))

A fund's classes own slices of one book. Every class's money is traded in
that book, so capacity caps, market impact and execution limits see the whole
fund's size, and each class keeps its own units, net assets and NAV per unit.

**Each day** the book's gains, losses and income (price moves, dividends,
interest, trading costs) are split between the classes in proportion to their
net assets at the previous close, the usual method for a multi-class fund.
What belongs to one class is then applied to it alone:

- **its fee**, its own `management_fee_bps` (the vehicle's when None),
  accrued on its own net assets;
- **its flows**, dealt at its own NAV per unit, sized against its own net
  assets, and moved by the vehicle's pricing (a swing or a levy) as the
  fund's net flow sets it;
- **its distributions**: a distributing class's share of the dividends is
  paid out, while an accumulating class's stays in the book, invested as the
  run's dividend policy says.

`result.share_classes` holds each class's book. The result's own NAV and
metrics are the whole fund's.
"""
# BN-279 (#292). Decided with Karan on 2026-10-12: the engine models the
# classes (a Fund passes its own through), the pool is split by the previous
# close's net assets, and currency-hedged classes are a separate issue.
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..portfolio.base import Portfolio
from .flows import FlowContext, FlowRecord, Flows
from .vehicle import Vehicle

ACCUMULATING = "accumulating"
DISTRIBUTING = "distributing"
DISTRIBUTIONS = (ACCUMULATING, DISTRIBUTING)

# A class with fewer units than this has none: its NAV per unit is carried.
NO_UNITS = 1e-12


@dataclass(frozen=True)
class ShareClass:
    """One share class of a fund.

    Attributes:
        name: What the class is called, such as ``"Acc"`` or ``"I GBP Dist"``.
        management_fee_bps: The class's annual management fee in basis
            points. None uses the vehicle's.
        distribution: ``"accumulating"`` (the default) keeps its income in
            the fund, ``"distributing"`` pays it out.

    Raises:
        ValueError: If the distribution is not one of the two, or the fee is
            negative.
    """
    name: str
    management_fee_bps: float | None = None
    distribution: str = ACCUMULATING

    def __post_init__(self) -> None:
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"Unknown distribution {self.distribution!r}. "
                             f"Supported: {', '.join(DISTRIBUTIONS)}.")

        if self.management_fee_bps is not None and self.management_fee_bps < 0:
            raise ValueError(f"management_fee_bps cannot be negative, got "
                             f"{self.management_fee_bps!r}.")


@dataclass(frozen=True)
class ShareClassBook:
    """One share class's record of a run.

    Attributes:
        name: The class's name.
        distribution: ``"accumulating"`` or ``"distributing"``.
        management_fee_bps: The fee it was charged, in basis points a year.
        net_assets: Its net assets at the end of each day, from the eve of
            the first.
        units: Its units outstanding at the end of each day.
        nav_per_unit: Its NAV per unit at the end of each day.
        fees: The fee it accrued each day.
        distributions: What it paid out each day, in total.
        distributions_per_unit: What it paid out each day, per unit.
        flows: Its own flows as dealt.
    """
    name: str
    distribution: str
    management_fee_bps: float
    net_assets: pd.Series
    units: pd.Series
    nav_per_unit: pd.Series
    fees: pd.Series
    distributions: pd.Series
    distributions_per_unit: pd.Series
    flows: list[FlowRecord] = field(default_factory=list)

    def get_returns(self) -> pd.Series:
        """The class's daily returns per unit: its NAV per unit from one day
        to the next, with what it paid out added back, so a distributing
        class's returns are total returns. The first runs from the launch
        price to the first close."""
        per_unit = self.nav_per_unit
        paid = self.distributions_per_unit.reindex(per_unit.index, fill_value=0.0)

        return ((per_unit + paid) / per_unit.shift(1) - 1.0).iloc[1:]

    def total_return(self) -> float:
        """The class's total return over the run, distributions reinvested."""
        returns = self.get_returns()

        return float((1.0 + returns).prod() - 1.0) if len(returns) else 0.0


@dataclass
class _Ledger:
    """One class's running books."""
    share_class: ShareClass
    fee_bps: float
    capital: float
    flows: list[Flows]
    net: float = 0.0
    units: float = 0.0
    weight: float = 0.0
    requested: float = 0.0
    per_unit: dict[pd.Timestamp, float] = field(default_factory=dict)
    net_history: dict[pd.Timestamp, float] = field(default_factory=dict)
    units_history: dict[pd.Timestamp, float] = field(default_factory=dict)
    fees: dict[pd.Timestamp, float] = field(default_factory=dict)
    paid: dict[pd.Timestamp, float] = field(default_factory=dict)
    paid_per_unit: dict[pd.Timestamp, float] = field(default_factory=dict)
    records: list[FlowRecord] = field(default_factory=list)

    def unit_price(self) -> float:
        """NAV per unit now, or the last one while it has no units."""
        if self.units > NO_UNITS:
            return self.net / self.units

        return list(self.per_unit.values())[-1]


def class_terms(share_classes: Sequence[ShareClass] | None,
                initial_capital: float | Mapping[str, float],
                flows: Flows | list[Flows] | Mapping[str, Flows | list[Flows]] | None,
                vehicle: Vehicle | None) -> list[_Ledger]:
    """Each class's capital, fee and flows, checked; empty without classes.

    Raises:
        ValueError: If classes share a name, a capital or flows name a class
            there is not, or a single capital or flow list is given for more
            than one class (or a mapping without classes).
    """
    if not share_classes:
        if isinstance(initial_capital, Mapping) or isinstance(flows, Mapping):
            raise ValueError("Capital or flows by share class need share_classes.")

        return []

    names = [share_class.name for share_class in share_classes]

    if len(set(names)) != len(names):
        raise ValueError(f"Share class names must differ, got {names}.")

    capital = _by_class(initial_capital, names, "initial_capital")
    by_class = _by_class(flows, names, "flows") if flows is not None else {}
    fee = vehicle.management_fee_bps if vehicle is not None else 0.0

    return [_Ledger(share_class=share_class,
                    fee_bps=(share_class.management_fee_bps
                             if share_class.management_fee_bps is not None else fee),
                    capital=float(capital.get(share_class.name, 0.0)),
                    flows=_as_list(by_class.get(share_class.name)))
            for share_class in share_classes]


def _by_class(value: object,
              names: list[str],
              what: str) -> dict[str, Any]:
    """*value* by class name: a mapping as given, a single value for a fund
    of one class."""
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))

        if unknown:
            raise ValueError(f"{what} names no share class {', '.join(unknown)}. "
                             f"Classes: {', '.join(names)}.")

        return dict(value)

    if len(names) > 1:
        raise ValueError(f"With several share classes, give {what} by class name, "
                         f"such as {{{names[0]!r}: ...}}.")

    return {names[0]: value}


def _as_list(flows: object) -> list[Flows]:
    """Flows as a list."""
    if flows is None:
        return []

    return list(flows) if isinstance(flows, list | tuple) else [flows]  # type: ignore[list-item]


class ShareClassesMixin:
    """Keeps each share class's books beside the fund's.

    Before AccountingMixin and DividendsMixin in BacktestEngine's bases,
    whose declarations of these methods it provides; what it uses from them
    is annotated here, never defined, so it shadows nothing.
    """

    _ledgers: list[_Ledger]
    _net_value: Callable[[Portfolio], float]

    def _start_classes(self,
                       eve: pd.Timestamp,
                       launch_price: float) -> None:
        """Open each class's register on the eve of the run."""
        for ledger in self._ledgers:
            ledger.net = ledger.capital
            ledger.units = ledger.capital / launch_price
            ledger.per_unit = {eve: launch_price}
            ledger.net_history = {eve: ledger.net}
            ledger.units_history = {eve: ledger.units}
            ledger.fees, ledger.paid, ledger.paid_per_unit = {}, {}, {}
            ledger.records = []

    def _begin_class_day(self) -> None:
        """Fix the day's split: each class's share of the net assets at the
        previous close."""
        total = sum(max(ledger.net, 0.0) for ledger in self._ledgers)

        for ledger in self._ledgers:
            ledger.weight = max(ledger.net, 0.0) / total if total > 0.0 else 0.0

    def _allocate_classes(self,
                          portfolio: Portfolio) -> None:
        """Split what the book gained or lost since the last split between
        the classes, by the day's shares."""
        if not self._ledgers:
            return

        change = self._net_value(portfolio) - sum(ledger.net for ledger in self._ledgers)

        for ledger in self._ledgers:
            ledger.net += ledger.weight * change

    def _class_fees(self,
                    portfolio: Portfolio,
                    date: pd.Timestamp,
                    days: int) -> float:
        """Each class's fee on its own net assets over *days*; the total."""
        self._allocate_classes(portfolio)
        total = 0.0

        for ledger in self._ledgers:
            fee = (max(ledger.net, 0.0) * ledger.fee_bps / 10_000.0 * days / 365.0
                   if days > 0 else 0.0)
            ledger.net -= fee
            ledger.fees[date] = ledger.fees.get(date, 0.0) + fee
            total += fee

        return total

    def _class_distributions(self,
                             income: float,
                             date: pd.Timestamp) -> float:
        """The distributing classes' shares of *income*, taken from them;
        the total to pay out."""
        total = 0.0

        for ledger in self._ledgers:
            if ledger.share_class.distribution != DISTRIBUTING:
                continue

            paid = ledger.weight * income
            ledger.net -= paid
            ledger.paid[date] = ledger.paid.get(date, 0.0) + paid

            if ledger.units > NO_UNITS:
                ledger.paid_per_unit[date] = (ledger.paid_per_unit.get(date, 0.0)
                                              + paid / ledger.units)

            total += paid

        return total

    def _class_flow_amount(self,
                           previous: pd.Timestamp,
                           date: pd.Timestamp,
                           carried: bool) -> float:
        """Each class's flow for the day, sized against the class itself;
        the fund's net flow. With *carried* (an ETF, which deals in whole
        creation units and carries the rest), each class's undealt demand
        is kept, so what is dealt later is shared by who asked for it."""
        total = 0.0

        for ledger in self._ledgers:
            history = pd.Series({**ledger.per_unit, date: ledger.unit_price()})
            context = FlowContext(previous=previous, date=date, aum=ledger.net,
                                  nav_per_unit=history.iloc[1:])
            amount = sum(scenario.amount(context) for scenario in ledger.flows)

            # A class cannot redeem more than it holds.
            amount = -min(-amount, max(ledger.net, 0.0)) if amount < 0.0 else amount
            ledger.requested = (ledger.requested if carried else 0.0) + amount
            total += amount

        return total

    def _deal_classes(self,
                      dealt: float,
                      dealing_price: float,
                      fund_per_unit: float,
                      date: pd.Timestamp) -> None:
        """The fund's dealt flow shared between the classes that asked for
        it, each dealt at its own NAV per unit, moved as the fund's price was
        by the vehicle's pricing."""
        asked = sum(ledger.requested for ledger in self._ledgers)

        if asked == 0.0 or dealt == 0.0:
            return

        filled = dealt / asked
        moved = dealing_price / fund_per_unit if fund_per_unit > 0.0 else 1.0

        for ledger in self._ledgers:
            cash = ledger.requested * filled
            ledger.requested -= cash

            if cash == 0.0:
                continue

            nav = ledger.unit_price()
            price = nav * moved
            units = max(cash / price, -ledger.units) if math.isfinite(price) and price > 0 else 0.0
            value = units * nav
            ledger.units += units
            ledger.net += value
            ledger.records.append(FlowRecord(date=date, amount=cash, nav_per_unit=nav,
                                             units=units, dealing_price=price,
                                             adjustment=cash - value))

    def _close_classes(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Split the rest of the day's result, and record each class."""
        if not self._ledgers:
            return

        self._allocate_classes(portfolio)

        for ledger in self._ledgers:
            ledger.per_unit[date] = ledger.unit_price()
            ledger.net_history[date] = ledger.net
            ledger.units_history[date] = ledger.units

    def _class_books(self) -> dict[str, ShareClassBook]:
        """Each class's record of the run, by name."""
        return {ledger.share_class.name: ShareClassBook(
                    name=ledger.share_class.name,
                    distribution=ledger.share_class.distribution,
                    management_fee_bps=ledger.fee_bps,
                    net_assets=pd.Series(ledger.net_history, dtype=float),
                    units=pd.Series(ledger.units_history, dtype=float),
                    nav_per_unit=pd.Series(ledger.per_unit, dtype=float),
                    fees=pd.Series(ledger.fees, dtype=float),
                    distributions=pd.Series(ledger.paid, dtype=float),
                    distributions_per_unit=pd.Series(ledger.paid_per_unit, dtype=float),
                    flows=list(ledger.records))
                for ledger in self._ledgers}
