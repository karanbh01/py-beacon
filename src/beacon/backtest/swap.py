# src/beacon/backtest/swap.py
"""
The total return swap a synthetic fund holds, valued beside its books.

The fund's portfolio holds the substitute basket; the swap is not a holding.
Its value is kept here, owed to the fund when positive and by it when
negative, and the fund's net assets are the portfolio's value, less the fee
owed, plus this. It is settled in cash at each reset.

Between resets, on a notional N set at the last reset:

    value = N x (index return) - N x (basket return) - spread accrued

The index return is the tracked index's level against its level at the reset.
The basket return is chained day by day from the basket's own moves: each day
the holdings carried in from the day before, at today's prices, plus the
dividends they paid today, over their value at yesterday's close. So money the
fund adds or takes out, its fee, its trading costs and its cash do not count
as the basket's return; only what the basket itself earned does. The spread
accrues on the notional over calendar days (ACT/360).

The swap opens at the first rebalance, when the basket is first bought, and is
reset:

- when its value owed to the fund reaches the exposure limit of net assets,
- at every rebalance, settled before the rebalance trades,
- on any day money flows in or out, settled after the flows are dealt.

A positive value settled is cash the fund then invests in the basket; a
negative one is paid from cash, selling the basket pro rata for what the cash
does not cover.
"""
# BN-270. The basket return is chained rather than read off the fund's total
# assets so that flows, fees and costs, which are the fund's and not the
# basket's, never pass through the swap.
import logging
from dataclasses import dataclass

import pandas as pd

from ..index.result import IndexResult
from ..portfolio import reasons
from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import DIVIDEND, SWAP
from ..strategy.synthetic import SwapTerms

logger = logging.getLogger(__name__)

# Why a swap was reset.
LIMIT = "limit"
REBALANCE = "rebalance"
FLOWS = "flows"

# The financing leg's day count: money-market ACT/360, as TotalReturnSwap uses.
SPREAD_DAY_COUNT = 360.0


@dataclass(frozen=True)
class SwapReset:
    """One reset of a synthetic fund's swap.

    Attributes:
        date: The day it was reset.
        settled: The swap's value paid in cash: received by the fund when
            positive, paid by it when negative. 0 when the swap opened.
        reason: Why: ``"limit"`` (its value owed to the fund reached the
            exposure limit), ``"rebalance"`` or ``"flows"``.
        notional: The notional it was reset to: the fund's net assets.
    """
    date: pd.Timestamp
    settled: float
    reason: str
    notional: float


class SwapMixin:
    """Values, settles and resets a synthetic fund's swap."""

    # The engine's own, which BacktestEngine and its other mixins declare.
    swap: SwapTerms | None
    index_result: IndexResult
    _fees_payable: float
    _swap_value: float

    # Last in BacktestEngine's bases, so these declarations never shadow the
    # AccountingMixin methods they stand for.
    def _net_value(self,
                   portfolio: Portfolio) -> float:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _sell_pro_rata(self,
                       portfolio: Portfolio,
                       amount: float,
                       date: pd.Timestamp,
                       free: bool = False) -> bool:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _start_swap(self) -> None:
        """Forget any earlier run's swap; nothing is open until the basket is
        first bought."""
        self._swap_value = 0.0
        self._swap_open = False
        self._swap_notional = 0.0
        self._swap_level = 1.0
        self._swap_basket = 1.0
        self._swap_spread = 0.0
        self._swap_base = 0.0
        self._swap_owed = 0.0
        self._swap_cursor = 0
        self._swap_carried: dict[str, float] = {}
        self._swap_history: dict[pd.Timestamp, float] = {}
        self._swap_resets: list[SwapReset] = []
        self._swap_levels = self.index_result.index_levels.sort_index()

    def _carry_swap_basket(self,
                           portfolio: Portfolio) -> None:
        """Note the basket carried into the day, before today's dividends
        are reinvested or anything is traded."""
        if self.swap is not None:
            self._swap_carried = {name: holding.quantity
                                  for name, holding in portfolio.holdings.items()
                                  if holding.quantity > 0}

    def _mark_swap(self,
                   portfolio: Portfolio,
                   previous: pd.Timestamp,
                   date: pd.Timestamp) -> None:
        """Value the open swap at today's prices."""
        if self.swap is None or not self._swap_open:
            return

        carried = sum(quantity * (portfolio.holdings[name].current_price or 0.0)
                      for name, quantity in self._swap_carried.items()
                      if name in portfolio.holdings)
        # Only the cash flows since yesterday's close are read: today's
        # dividends arrive before the mark, and rereading the whole list
        # every day would cost the square of the run's length.
        paid = sum(flow.amount for flow in portfolio.cash_flows[self._swap_cursor:]
                   if flow.kind == DIVIDEND and flow.date == date)

        if self._swap_base > 0.0:
            self._swap_basket *= (carried + paid) / self._swap_base

        days = max((date - previous).days, 0)
        self._swap_spread += (self._swap_notional * self.swap.spread_bps / 10_000.0
                              * days / SPREAD_DAY_COUNT)
        self._swap_value = (self._swap_owed
                            + self._swap_notional * (self._level(date) / self._swap_level - 1.0)
                            - self._swap_notional * (self._swap_basket - 1.0)
                            - self._swap_spread)

    def _swap_reset_reason(self,
                           portfolio: Portfolio,
                           rebalancing: bool,
                           flowed: bool) -> str | None:
        """Why the swap is reset today, or None when it is not."""
        if self.swap is None:
            return None

        if rebalancing:
            return REBALANCE

        if not self._swap_open:
            return None

        if flowed:
            return FLOWS

        if self._swap_value >= self.swap.exposure_limit * self._net_value(portfolio):
            return LIMIT

        return None

    def _settle_swap(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> float:
        """Pay the swap's value in cash. Returns what was settled: positive
        when the fund received it."""
        value = self._swap_value

        if value > 0.0:
            portfolio.receive_cash(value, date, SWAP)
        elif value < 0.0:
            short = -value - (portfolio.cash_balance - self._fees_payable)

            if short > 0.0:
                with portfolio.trading(reasons.SETTLEMENT):
                    self._sell_pro_rata(portfolio, short, date)

            paid = min(-value, portfolio.cash_balance)
            portfolio.receive_cash(-paid, date, SWAP)
            value = -paid

        # Anything the cash could not cover stays owed into the next period.
        self._swap_value -= value

        return value

    def _open_swap(self,
                   portfolio: Portfolio,
                   date: pd.Timestamp,
                   reason: str,
                   settled: float) -> None:
        """Start a new period: the notional set to the fund's net assets, the
        index and the basket measured from today."""
        # Normally zero: what a negative settlement could not pay from cash.
        self._swap_owed = self._swap_value
        self._swap_notional = self._net_value(portfolio)
        self._swap_level = self._level(date)
        self._swap_basket = 1.0
        self._swap_spread = 0.0
        self._swap_open = True
        self._swap_resets.append(SwapReset(date=date, settled=settled, reason=reason,
                                           notional=self._swap_notional))

        logger.debug("[%s] Swap reset (%s): %.2f settled, notional %.2f.",
                     date.date(), reason, settled, self._swap_notional)

    def _record_swap(self,
                     portfolio: Portfolio,
                     date: pd.Timestamp) -> None:
        """Record the day's value, and the basket's value at the close that
        tomorrow's basket return is measured from."""
        if self.swap is None:
            return

        self._swap_history[date] = self._swap_value
        self._swap_cursor = len(portfolio.cash_flows)
        self._swap_base = sum(holding.quantity * (holding.current_price or 0.0)
                              for holding in portfolio.holdings.values())

    def _level(self,
               date: pd.Timestamp) -> float:
        """The tracked index's level on *date*, or its last before it."""
        level = self._swap_levels.asof(date)

        if pd.isna(level):
            level = self._swap_levels.iloc[0]

        return float(level)
