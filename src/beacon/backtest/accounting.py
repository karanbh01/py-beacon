# src/beacon/backtest/accounting.py
"""
A backtest's fund accounting: units, flows, the management fee and interest.

## Units

The initial capital buys units at the vehicle's launch price (1.0 without a
vehicle). A flow creates or cancels units at that day's NAV per unit, before
anything is traded for it, so a flow changes the fund's size and not its NAV
per unit; performance is measured per unit. Without flows the units never
change and every figure is what it would be without them.

## A day's flows

Flows are dealt after the day's prices are marked:

- **In**: the cash arrives and units are created. On a rebalance day the
  rebalance invests it; on any other day it is invested at once, as the
  implementation's `invest_flows` says, keeping its cash buffer topped up
  first. Under an execution limit the buying is worked over the following
  days like any other order.
- **Out**: paid from cash first, then by selling every holding pro rata, and
  units are cancelled. A redemption is paid the day it is dealt, so the
  sales that fund it are not held back by an execution limit. One larger
  than the fund is cut to the fund.

## Pricing

The vehicle's pricing method (see `beacon.backtest.dealing`) sets the price
units are dealt at, or adds a levy, so the dealing investors can pay for the
trading their flow causes. Under single pricing, the default, the fund bears
it. A method left to estimate its rate is given the cost of trading the whole
flow: the fixed cost plus market impact, across the names the flow would
buy or sell.

## Exchange-traded funds

With an `EtfVehicle` the flows are the APs' net demand, rounded toward zero
to whole creation units each day, with the rest carried to the next. An
in-kind creation or redemption moves the holdings themselves, pro rata and
without cost; a cash one is invested or raised like any flow, and its
creation fee is paid into the fund. After each close the shares are quoted
on the exchange (see `beacon.backtest.etf`).

## The management fee

The vehicle's fee accrues every calendar day on the fund's net assets
(ACT/365) and is owed as a liability: the NAV is net of it, a rebalance
invests only what is not owed, and it is paid from cash as soon as there is
cash to pay it.
"""
# BN-267, phase 5 of decisions/0006. Interest on cash (BN-276) moved here from
# engine.py, beside the other daily accruals.
import logging

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..portfolio.base import Portfolio, TradeInstruction
from ..portfolio.cash_flows import FEE, INTEREST, REDEMPTION, SUBSCRIPTION
from .costs import WorkingOrder
from .dealing import Deal, SinglePricing
from .etf import EtfVehicle, MarketMaker, Quote
from .flows import FlowContext, FlowRecord, Flows
from .implementation import HOLDINGS, Implementation, RebalanceStep
from .result import UnfilledOrder
from .vehicle import DEFAULT_LAUNCH_PRICE, Vehicle

logger = logging.getLogger(__name__)

# Below this a trade is not worth placing, as in execution.py.
MIN_TRADE_VALUE = 0.01

# How many rounds of selling a redemption may take to cover its costs.
SELL_ROUNDS = 4


class AccountingMixin:
    """Units, flows, fees and interest, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    initial_capital: float
    implementation: Implementation
    modelling_assumptions: ModellingAssumptions
    flows: list[Flows]
    vehicle: Vehicle | None
    transaction_cost_bps: float
    _rebalance_steps: list[RebalanceStep]
    _working: list[WorkingOrder]
    _fees_payable: float

    def _fetch_price(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float | None:
        """Provided by PricingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _instruction(self,
                     asset_id: str,
                     side: str,
                     quantity: float,
                     price: float,
                     date: pd.Timestamp) -> TradeInstruction:
        """Provided by ExecutionMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _affordable(self,
                    buys: list[TradeInstruction],
                    cash: float,
                    date: pd.Timestamp) -> list[TradeInstruction]:
        """Provided by ExecutionMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _trade_all(self,
                   portfolio: Portfolio,
                   trades: list[TradeInstruction],
                   date: pd.Timestamp) -> list[UnfilledOrder]:
        """Provided by ExecutionMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _cost_rate(self,
                   asset_id: str,
                   notional: float,
                   date: pd.Timestamp) -> float:
        """Provided by ExecutionMixin; declared so this one type-checks."""
        raise NotImplementedError

    # -- the books -----------------------------------------------------------

    @property
    def launch_price(self) -> float:
        """NAV per unit at launch."""
        return (self.vehicle.launch_price if self.vehicle is not None
                else DEFAULT_LAUNCH_PRICE)

    def _start_accounting(self,
                          eve: pd.Timestamp,
                          first_day: pd.Timestamp) -> None:
        """Open the unit register on the eve of the run."""
        self._units = self.initial_capital / self.launch_price
        self._fees_payable = 0.0
        self._flow_records: list[FlowRecord] = []
        self._units_history: dict[pd.Timestamp, float] = {eve: self._units}
        self._payable_history: dict[pd.Timestamp, float] = {eve: 0.0}
        self._per_unit: dict[pd.Timestamp, float] = {eve: self.launch_price}

        # BN-268: an ETF's carried demand, its quotes and its basket's cost.
        self._creation_carry = 0.0
        self._dealt_today = 0.0
        self._quotes: list[Quote] = []
        self._basket: tuple[tuple[int, int], float] | None = None
        self._market_maker = (MarketMaker(self.vehicle)
                              if isinstance(self.vehicle, EtfVehicle) else None)

        for scenario in self.flows:
            scenario.start(first_day)

    def _net_value(self,
                   portfolio: Portfolio) -> float:
        """The fund's net assets: what it holds, less the fee it owes."""
        return portfolio.get_total_value() - self._fees_payable

    def _nav_per_unit(self,
                      portfolio: Portfolio) -> float:
        """Today's NAV per unit, or the last one while no units exist."""
        if self._units > 0.0:
            return self._net_value(portfolio) / self._units

        return list(self._per_unit.values())[-1]

    def _close_books(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> None:
        """Pay what fee the cash allows and record the day's units."""
        self._settle_fees(portfolio, date)
        self._units_history[date] = self._units
        self._payable_history[date] = self._fees_payable
        self._per_unit[date] = self._nav_per_unit(portfolio)
        self._quote_market(portfolio, date)

    # -- accruals ------------------------------------------------------------

    def _accrue(self,
                portfolio: Portfolio,
                previous: pd.Timestamp,
                date: pd.Timestamp) -> None:
        """Interest on cash and the management fee, from *previous* to
        *date*, both over calendar days (ACT/365)."""
        days = (date - previous).days

        if days <= 0:
            return

        rate = self.modelling_assumptions.cash_rate or 0.0

        if rate != 0.0 and portfolio.cash_balance > 0.0:
            portfolio.receive_cash(portfolio.cash_balance * rate * days / 365.0,
                                   date, INTEREST)

        if self.vehicle is not None:
            self._fees_payable += self.vehicle.daily_fee(
                self._net_value(portfolio), days)

    def _settle_fees(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> None:
        """Pay as much of the fee owed as the cash covers."""
        paid = min(self._fees_payable, portfolio.cash_balance)

        if paid > 0.0:
            portfolio.receive_cash(-paid, date, FEE)
            self._fees_payable -= paid

    # -- flows ---------------------------------------------------------------

    def _deal_flows(self,
                    portfolio: Portfolio,
                    previous: pd.Timestamp,
                    date: pd.Timestamp) -> float:
        """Deal today's flows: cash in or out, units created or cancelled.

        Returns:
            float: The cash that arrived, for the day to invest; 0 for none
            or an outflow.
        """
        self._dealt_today = 0.0

        if not self.flows:
            return 0.0

        net = self._net_value(portfolio)
        per_unit = self._nav_per_unit(portfolio)
        history = pd.Series({**self._per_unit, date: per_unit})
        context = FlowContext(previous=previous, date=date, aum=net,
                              nav_per_unit=history.iloc[1:])
        amount = sum(scenario.amount(context) for scenario in self.flows)
        etf = self.vehicle if isinstance(self.vehicle, EtfVehicle) else None
        creations = None

        if etf is not None:
            amount, creations = self._in_creation_units(etf, amount, per_unit)

        if amount < 0.0:
            amount = -min(-amount, max(net, 0.0))

        if abs(amount) < MIN_TRADE_VALUE:
            return 0.0

        in_kind = etf is not None and etf.in_kind
        deal = self._dealt(portfolio, amount, per_unit, net, date)

        if deal.cash > 0.0:
            portfolio.receive_cash(deal.cash, date, SUBSCRIPTION)

            if in_kind:
                self._transfer_in(portfolio, deal.cash, date)
        elif deal.cash < 0.0:
            deal = _scaled(deal, self._redeem(portfolio, -deal.cash, date,
                                              free=in_kind) / -deal.cash)

        self._units = max(self._units + deal.units, 0.0)
        self._dealt_today = deal.cash
        self._flow_records.append(FlowRecord(
            date=date, amount=deal.cash, nav_per_unit=per_unit,
            units=deal.units, dealing_price=deal.price,
            adjustment=deal.adjustment, creation_units=creations))

        logger.debug("[%s] Flow of %.2f dealt at %.6f a unit.", date.date(),
                     deal.cash, deal.price)

        # Delivered in kind, an inflow is already invested.
        return 0.0 if in_kind else max(deal.cash, 0.0)

    def _in_creation_units(self,
                           etf: EtfVehicle,
                           amount: float,
                           per_unit: float) -> tuple[float, float]:
        """The demand an ETF can deal today in whole creation units, and how
        many; the rest carries to tomorrow."""
        wanted = amount + self._creation_carry
        units = etf.creation_units(wanted, per_unit)
        dealt = units * etf.creation_unit * per_unit
        self._creation_carry = wanted - dealt

        return dealt, units

    def _transfer_in(self,
                     portfolio: Portfolio,
                     amount: float,
                     date: pd.Timestamp) -> None:
        """Holdings worth *amount* delivered in kind: bought pro rata at the
        day's prices, at no cost."""
        weights = self._flow_weights(portfolio, date)
        total = sum(weights.values())

        if total <= 0.0:
            return

        for name, weight in weights.items():
            price = self._fetch_price(name, date)

            if price is not None and price > 0.0:
                quantity = amount * weight / total / price
                portfolio.apply(TradeInstruction(name, "BUY", quantity, price,
                                                 0.0), date)

    def _dealt(self,
               portfolio: Portfolio,
               amount: float,
               per_unit: float,
               net: float,
               date: pd.Timestamp) -> Deal:
        """Today's flow, priced by the vehicle's method."""
        pricing = (self.vehicle.pricing if self.vehicle is not None
                   else SinglePricing())
        cost = (self._flow_cost_rate(portfolio, amount, date)
                if pricing.estimated else 0.0)

        return pricing.deal(amount, per_unit, net, cost)

    def _flow_cost_rate(self,
                        portfolio: Portfolio,
                        amount: float,
                        date: pd.Timestamp) -> float:
        """The cost of trading a flow of *amount*, as a share of it: what an
        inflow would buy, or the holdings an outflow would sell, each at its
        share of the flow."""
        weights = (self._flow_weights(portfolio, date) if amount > 0.0
                   else self._held_values(portfolio, date))
        total = sum(weights.values())

        if total <= 0.0:
            return self._fixed_cost_rate()

        return sum(weight / total
                   * self._cost_rate(name, abs(amount) * weight / total, date)
                   for name, weight in weights.items())

    def _redeem(self,
                portfolio: Portfolio,
                owed: float,
                date: pd.Timestamp,
                free: bool = False) -> float:
        """Raise *owed* in cash, selling pro rata if the cash falls short,
        and pay it out; with *free*, the holdings leave in kind, at no cost.
        Returns what was paid."""
        for _ in range(SELL_ROUNDS):
            short = owed + self._fees_payable - portfolio.cash_balance

            if short < MIN_TRADE_VALUE or not self._sell_pro_rata(
                    portfolio, short, date, free):
                break

        paid = min(owed, max(portfolio.cash_balance - self._fees_payable, 0.0))

        if paid > 0.0:
            portfolio.receive_cash(-paid, date, REDEMPTION)

        return paid

    def _sell_pro_rata(self,
                       portfolio: Portfolio,
                       amount: float,
                       date: pd.Timestamp,
                       free: bool = False) -> bool:
        """Sell every holding in proportion to its value to raise about
        *amount*, at no cost when *free*. Returns whether anything was
        sold."""
        values = self._held_values(portfolio, date)
        held = sum(values.values())

        if held <= 0.0:
            return False

        # Grossed up for the fixed cost; impact is covered by another round.
        rate = 0.0 if free else self._fixed_cost_rate()
        share = min(amount / held / (1.0 - rate), 1.0)
        sold = False

        for name, value in values.items():
            holding = portfolio.holdings[name]
            price = value / holding.quantity
            quantity = holding.quantity * share

            if quantity * price >= MIN_TRADE_VALUE:
                portfolio.apply(TradeInstruction(name, "SELL", quantity, price, 0.0)
                                if free else self._instruction(name, "SELL",
                                                               quantity, price,
                                                               date), date)
                sold = True

        return sold

    def _invest_inflow(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Invest the cash that is free, keeping the buffer, as the
        implementation says: at once, or as working orders under an
        execution limit."""
        net = self._net_value(portfolio)
        free = (portfolio.cash_balance - self._fees_payable
                - self.implementation.cash_buffer * net)
        weights = self._flow_weights(portfolio, date)
        total = sum(weights.values())

        if free < MIN_TRADE_VALUE or total <= 0.0:
            return

        buys = []

        for name, weight in weights.items():
            price = self._fetch_price(name, date)

            if price is None or price <= 0.0:
                continue

            spend = free * weight / total
            rate = self._fixed_cost_rate()
            buys.append(self._instruction(name, "BUY",
                                          spend / (price * (1.0 + rate)),
                                          price, date))

        buys = self._affordable(buys, free, date)

        if self.implementation.execution is not None:
            self._working.extend(WorkingOrder(date, buy.asset_id, "BUY",
                                              buy.quantity) for buy in buys)
            return

        self._trade_all(portfolio, buys, date)

    def _flow_weights(self,
                      portfolio: Portfolio,
                      date: pd.Timestamp) -> dict[str, float]:
        """What an inflow buys, and in what proportion."""
        if self.implementation.invest_flows == HOLDINGS:
            return self._held_values(portfolio, date)

        if not self._rebalance_steps:
            return {}

        return {name: weight
                for name, weight in self._rebalance_steps[-1].weights.items()
                if weight > 0.0}

    def _held_values(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> dict[str, float]:
        """Each priced holding's value today."""
        return {name: holding.quantity * price
                for name, holding in portfolio.holdings.items()
                if holding.quantity > 0
                and (price := self._fetch_price(name, date)) is not None
                and price > 0}

    # -- an ETF's quotes ----------------------------------------------------

    def _quote_market(self,
                      portfolio: Portfolio,
                      date: pd.Timestamp) -> None:
        """Quote an ETF's shares at the close."""
        maker = self._market_maker

        if maker is None:
            return

        history = list(self._per_unit.values())
        net = self._net_value(portfolio)
        flow_share = self._dealt_today / net if net > 0.0 else 0.0

        self._quotes.append(maker.quote(date, history[-1], history[-2],
                                        flow_share,
                                        self._basket_cost(portfolio, date,
                                                          maker.vehicle)))

    def _basket_cost(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp,
                     etf: EtfVehicle) -> float:
        """The cost of trading one creation unit's basket, as a share of it.

        With market impact on, refreshed at each rebalance and each new week,
        since impact reads every holding's history.
        """
        week = (date.isocalendar().year, date.isocalendar().week)
        rebalanced = bool(self._rebalance_steps) and self._rebalance_steps[-1].date == date
        cached = self._basket

        if (self.implementation.impact is not None and cached is not None
                and cached[0] == week and not rebalanced):
            return cached[1]

        values = self._held_values(portfolio, date)
        total = sum(values.values())
        unit_value = etf.creation_unit * self._nav_per_unit(portfolio)
        cost = (self._fixed_cost_rate() if total <= 0.0
                else sum(value / total
                         * self._cost_rate(name, unit_value * value / total, date)
                         for name, value in values.items()))
        self._basket = (week, cost)

        return cost

    def _fixed_cost_rate(self) -> float:
        """The fixed cost as a fraction of a trade's value."""
        return self.transaction_cost_bps / 10_000.0


def _scaled(deal: Deal,
            share: float) -> Deal:
    """*deal* cut to *share* of itself: a redemption the fund could only
    partly pay."""
    if share >= 1.0:
        return deal

    return Deal(units=deal.units * share, cash=deal.cash * share,
                price=deal.price, adjustment=deal.adjustment * share)
