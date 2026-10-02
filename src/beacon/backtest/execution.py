# src/beacon/backtest/execution.py
"""
Trading: what a rebalance trades, what it costs, and how fast it can happen.

    Implementation(impact=MarketImpact(coefficient=1.0),
                   execution=ExecutionLimit(participation=0.1))

## Costs

Every trade pays the fixed cost in basis points of its value. With a
`MarketImpact` it also pays market impact, which grows with the trade's size
against what the name trades in a day (the square-root law):

    impact = coefficient x daily volatility x sqrt(trade value / average
             daily traded value)

as a fraction of the trade's value. Without impact a backtest's return does
not depend on its size; with it, the same trade costs a large fund more.

## Sizing buys

A rebalance sells first and buys with what is left. Buys are sized so that
they and their costs fit the cash the sells leave, so a rebalance fills in
full and the book ends fully invested, short of its target weights only by
the costs.

## Execution limits

With an `ExecutionLimit`, a name trades at most a share of the day's volume,
or an order is spread evenly over a number of days, or both. Whatever cannot
trade on the rebalance day is a working order, traded on the following
sessions under the same limit. A new rebalance replaces any order still
working. A replaced order, or one still working when the run ends, is
recorded in `result.unfilled` with the reason ``"execution limit"``.

A target name with no price on the rebalance day is recorded in
`result.unfilled` with the reason ``"no price"``.
"""
# BN-266, phase 4 of decisions/0006, and the sizing and unpriceable-target
# halves of BN-252. The trade generation moved here from engine.py.
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..portfolio.base import (
    CASH_TOLERANCE,
    Holding,
    Portfolio,
    TradeInstruction,
)
from .costs import ExecutionLimit, WorkingOrder
from .implementation import Implementation
from .result import (
    CASH_SHORT,
    EXECUTION_LIMIT,
    NO_PRICE,
    UnfilledOrder,
)
from .rules import BacktestModifier
from .screens import ScreenContext

logger = logging.getLogger(__name__)

# Below this a trade is not worth placing.
MIN_TRADE_VALUE = 0.01


class ExecutionMixin:
    """Trade generation, costs and execution, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    data_provider: DataFetcher
    currency: str
    implementation: Implementation
    modifiers: list[BacktestModifier]
    transaction_cost_bps: float
    _working: list[WorkingOrder]

    def _fetch_price(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float | None:
        """Provided by PricingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _record_rebalance_pricing(self,
                                  date: pd.Timestamp) -> None:
        """Provided by BacktestEngine; declared so this one type-checks."""
        raise NotImplementedError

    # -- costs ---------------------------------------------------------------

    def _trade_cost(self,
                    asset_id: str,
                    notional: float,
                    date: pd.Timestamp) -> float:
        """The fixed cost plus any market impact of trading *notional*."""
        rate = self.transaction_cost_bps / 10_000.0
        impact = self.implementation.impact

        if impact is not None and notional > 0.0:
            rate += impact.rate(asset_id, notional, date,
                                ScreenContext(self.data_provider, self.currency))

        return notional * rate

    def _instruction(self,
                     asset_id: str,
                     side: str,
                     quantity: float,
                     price: float,
                     date: pd.Timestamp) -> TradeInstruction:
        """A trade, costed."""
        return TradeInstruction(asset_id, side, quantity, price,
                                self._trade_cost(asset_id, quantity * price, date))

    # -- what to trade -------------------------------------------------------

    def _sell_instruction(self,
                          asset_id: str,
                          holding: Holding,
                          target_weights: dict[str, float],
                          current_value: float,
                          date: pd.Timestamp) -> TradeInstruction | None:
        """A sell of *asset_id* if it is out of the target or overweight."""
        price = self._fetch_price(asset_id, date)
        if price is None:
            return None

        target_w = target_weights.get(asset_id, 0.0)
        if target_w == 0:
            return self._instruction(asset_id, "SELL", holding.quantity, price,
                                     date)

        excess_value = holding.quantity * price - current_value * target_w
        if excess_value <= 1e-6 or excess_value / price <= 1e-9:
            return None

        return self._instruction(asset_id, "SELL", excess_value / price, price,
                                 date)

    def _generate_trades(self,
                         portfolio: Portfolio,
                         target_weights: dict[str, float],
                         date: pd.Timestamp
                         ) -> tuple[list[TradeInstruction], list[UnfilledOrder]]:
        """The trades that move *portfolio* to *target_weights*, sells first,
        and the target names that could not be priced.

        Buys are sized so they and their costs fit the cash the sells leave
        (BN-252): they used to be sized on the whole NAV with nothing set
        aside for costs, so the last buy of almost every costed rebalance ran
        short.
        """
        current_value = portfolio.get_total_value()
        if current_value <= 0:
            return [], []

        sells = [instruction
                 for asset_id, holding in portfolio.holdings.items()
                 if (instruction := self._sell_instruction(
                     asset_id, holding, target_weights, current_value, date))
                 is not None]

        buys: list[TradeInstruction] = []
        unpriced: list[UnfilledOrder] = []

        for asset_id, weight in target_weights.items():
            if weight <= 0:
                continue

            price = self._fetch_price(asset_id, date)

            if price is None or price <= 0:
                unpriced.append(UnfilledOrder(
                    date=date, asset_id=asset_id, requested_quantity=0.0,
                    filled_quantity=0.0, price=float("nan"),
                    shortfall_value=current_value * weight, reason=NO_PRICE))
                continue

            held = portfolio.holdings.get(asset_id)
            deficit = current_value * weight - (held.quantity * price if held else 0.0)

            if deficit > 1e-6:
                buys.append(self._instruction(asset_id, "BUY", deficit / price,
                                              price, date))

        cash = (portfolio.cash_balance
                + sum(sell.quantity * sell.price - sell.cost for sell in sells))

        return sells + self._affordable(buys, cash, date), unpriced

    def _affordable(self,
                    buys: list[TradeInstruction],
                    cash: float,
                    date: pd.Timestamp) -> list[TradeInstruction]:
        """*buys* scaled down together, when they and their costs need more
        than *cash*, so every one of them fills."""
        needed = sum(buy.quantity * buy.price + buy.cost for buy in buys)

        if needed <= cash or needed <= 0.0:
            return buys

        scale = max(cash, 0.0) / needed

        return [self._instruction(buy.asset_id, "BUY", buy.quantity * scale,
                                  buy.price, date) for buy in buys]

    # -- trading it ----------------------------------------------------------

    def _rebalance(self,
                   portfolio: Portfolio,
                   target_weights: dict[str, float],
                   date: pd.Timestamp) -> list[UnfilledOrder]:
        """Trade *portfolio* toward *target_weights*.

        Modifiers may veto the rebalance or adjust its trades. Orders still
        working from the last rebalance are replaced first.

        Returns:
            list of UnfilledOrder: What could not be traded as asked.
        """
        current_value = portfolio.get_total_value()
        if current_value <= 0:
            logger.warning(f"[{date}] Portfolio value is {current_value:.2f}. "
                           f"Skipping rebalance.")
            return []

        for modifier in self.modifiers:
            if modifier.should_skip_rebalance(date, portfolio, target_weights):
                logger.info(f"[{date}] Rebalance skipped by "
                            f"{modifier.__class__.__name__}.")
                return []

        logger.info(f"[{date}] Rebalancing to target weights: {target_weights}")

        # Recorded before the trades, and only for a rebalance that goes ahead:
        # a skipped one priced nothing (BN-183).
        self._record_rebalance_pricing(date)

        unfilled = self._abandon_working()
        trades, unpriced = self._generate_trades(portfolio, target_weights, date)

        for modifier in self.modifiers:
            trades = modifier.adjust_trades(trades, date, portfolio)

        if self.implementation.execution is None:
            unfilled.extend(self._trade_all(portfolio, trades, date))
        else:
            self._working = [WorkingOrder(date, trade.asset_id, trade.side,
                                          trade.quantity) for trade in trades]
            self._work_orders(portfolio, date)

        return unfilled + unpriced

    def _trade_all(self,
                   portfolio: Portfolio,
                   trades: list[TradeInstruction],
                   date: pd.Timestamp) -> list[UnfilledOrder]:
        """Execute every trade now: sells, then buys as cash allows."""
        unfilled: list[UnfilledOrder] = []

        for trade in trades:
            if trade.side == "SELL":
                portfolio.apply(trade, date)
                logger.debug(f"[{date}] Sold {trade.quantity:.4f} of "
                             f"{trade.asset_id}")
            elif trade.side == "BUY":
                shortfall = self._execute_buy(portfolio, trade, date)

                if shortfall is not None:
                    unfilled.append(shortfall)

        return unfilled

    def _work_orders(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> None:
        """Trade what the execution limit allows of each working order today,
        sells before buys, buys no further than the cash."""
        limit = self.implementation.execution

        if limit is None or not self._working:
            return

        for side in ("SELL", "BUY"):
            for order in self._working:
                if order.side == side:
                    self._work(portfolio, order, limit, date)

        self._working = [order for order in self._working
                         if order.remaining * _price_or_zero(
                             self._fetch_price(order.asset_id, date))
                         >= MIN_TRADE_VALUE]

    def _work(self,
              portfolio: Portfolio,
              order: WorkingOrder,
              limit: ExecutionLimit,
              date: pd.Timestamp) -> None:
        """Trade today's share of one working order."""
        price = self._fetch_price(order.asset_id, date)

        if price is None or price <= 0:
            return

        quantity = limit.allowed(order, date, self.data_provider)

        if order.side == "SELL":
            held = portfolio.holdings.get(order.asset_id)
            quantity = min(quantity, held.quantity if held else 0.0)
        else:
            quantity = _affordable_quantity(portfolio.cash_balance, price,
                                            quantity,
                                            self._cost_rate(order.asset_id,
                                                            quantity * price,
                                                            date))

        if quantity * price < MIN_TRADE_VALUE:
            return

        portfolio.apply(self._instruction(order.asset_id, order.side, quantity,
                                          price, date), date)
        order.filled += quantity

    def _cost_rate(self,
                   asset_id: str,
                   notional: float,
                   date: pd.Timestamp) -> float:
        """The cost of trading *notional*, as a fraction of it."""
        if notional <= 0.0:
            return self.transaction_cost_bps / 10_000.0

        return self._trade_cost(asset_id, notional, date) / notional

    def _abandon_working(self) -> list[UnfilledOrder]:
        """Record every order still working as unfilled, and drop them."""
        abandoned = [UnfilledOrder(date=order.date, asset_id=order.asset_id,
                                   requested_quantity=order.quantity,
                                   filled_quantity=order.filled,
                                   price=float("nan"),
                                   shortfall_value=float("nan"),
                                   reason=EXECUTION_LIMIT)
                     for order in self._working if order.remaining > 0]

        if abandoned:
            logger.info("%d order(s) still working were replaced or ended "
                        "unfinished.", len(abandoned))

        self._working = []

        return abandoned

    def _execute_buy(self,
                     portfolio: Portfolio,
                     trade: TradeInstruction,
                     date: pd.Timestamp) -> UnfilledOrder | None:
        """Buy as much of *trade* as the available cash supports.

        Buys are sized to fit the cash, so this sizes down only on floating
        point noise or when a modifier changed the trades. Dropping the whole
        order when cash falls a little short would distort the run far more
        than a slightly smaller position.

        Returns:
            UnfilledOrder or None: A record when the order could not be
            filled in full, otherwise None.
        """
        required = trade.quantity * trade.price + trade.cost

        if portfolio.cash_balance >= required * (1 - CASH_TOLERANCE):
            portfolio.apply(trade, date)
            logger.debug(f"[{date}] Bought {trade.quantity:.4f} of {trade.asset_id}")
            return None

        rate = trade.cost / (trade.quantity * trade.price) if trade.quantity else 0.0
        affordable = _affordable_quantity(portfolio.cash_balance, trade.price,
                                          trade.quantity, rate)

        if affordable * trade.price < MIN_TRADE_VALUE:
            logger.warning(f"[{date}] Cannot buy {trade.asset_id}: need "
                           f"{required:.2f}, have {portfolio.cash_balance:.2f}.")
            return UnfilledOrder(date=date, asset_id=trade.asset_id,
                                 requested_quantity=trade.quantity,
                                 filled_quantity=0.0, price=trade.price,
                                 shortfall_value=trade.quantity * trade.price,
                                 reason=CASH_SHORT)

        portfolio.apply(TradeInstruction(trade.asset_id, "BUY", affordable,
                                         trade.price,
                                         affordable * trade.price * rate), date)
        logger.warning(f"[{date}] Partially filled {trade.asset_id}: bought "
                       f"{affordable:.4f} of {trade.quantity:.4f} requested.")

        return UnfilledOrder(date=date, asset_id=trade.asset_id,
                             requested_quantity=trade.quantity,
                             filled_quantity=affordable, price=trade.price,
                             shortfall_value=(trade.quantity - affordable)
                             * trade.price,
                             reason=CASH_SHORT)


def _affordable_quantity(cash: float,
                         price: float,
                         quantity: float,
                         cost_rate: float) -> float:
    """At most *quantity*, no more than *cash* covers with its costs."""
    return max(min(cash / (price * (1.0 + cost_rate)), quantity), 0.0)


def _price_or_zero(price: float | None) -> float:
    return price if price is not None else 0.0
