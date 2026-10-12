# src/beacon/backtest/hedging.py
"""
The currency hedge a hedged share class carries.

A hedged class is denominated in another currency from the fund's book and
hedges its share of the assets back into its own: it sells the book's
currency forward, for its own, in the amount of its net assets times its
hedge ratio. The forward's gains and losses are the class's alone; the other
classes never see them.

**Pricing.** A forward to *T* is priced at covered interest parity, from the
day's spot rate *X* (class currency per unit of the book's) and the two
currencies' short-term interest rates over the days left, each on its own day
count (ACT/360 unless its reference data says ACT/365):

    F = X x (1 + r_class x t_class) / (1 + r_book x t_book)

Its value to the class, in its own currency, is the notional times the
contracted rate less today's forward for the same date, discounted at the
class currency's rate. A currency's rate is its short rate in the data (see
`beacon.data.kinds`), else the run's cash rate.

**Rolls.** A one-month forward is opened at the first session of the run,
and rolled at the first session of each month: settled in cash, and a new
one opened for the next month, sized to the class's net assets. Between
rolls the hedge ratio drifts as the assets move, and when it leaves the
class's band (its hedge ratio plus or minus *hedge_band*) the forward is
settled and reopened to the same date at the right size.

**Settlement** moves the forward's value into the book's cash, and the class
keeps it: its net assets do not change, because the value was already
counted in them.

The carry this earns or costs is the interest rate differential: a class in
a higher-rate currency than the book's earns about the difference a year.
"""
# BN-307. Decided with Karan on 2026-10-12: interest rates from series in the
# data, a monthly roll with a tolerance band, and unhedged classes in another
# currency in the same change.
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import HEDGE

logger = logging.getLogger(__name__)

# Why a hedge was traded.
OPEN = "open"
ROLL = "roll"
BAND = "band"

# Money-market days in a year for a currency whose day count the data does
# not give.
FORWARD_DAY_COUNT = 360.0

# A payment short by less than this has been made: what a pro-rata sale
# raises can differ from what was asked by a rounding error.
SHORTFALL = 1e-6


@dataclass
class Forward:
    """An open forward: the book's currency sold for the class's.

    Attributes:
        notional: How much of the book's currency is sold forward.
        rate: The contracted rate, class currency per unit of the book's.
        expiry: The day it settles.
        value: What it is worth to the class today, in the book's currency.
    """
    notional: float
    rate: float
    expiry: pd.Timestamp
    value: float = 0.0


@dataclass(frozen=True)
class HedgeTrade:
    """One trade of a class's hedge.

    Attributes:
        date: The day.
        reason: ``"open"``, ``"roll"`` (a new month) or ``"band"`` (the
            hedge ratio left its band).
        notional: The new forward's notional, in the book's currency.
        rate: The new forward's contracted rate, class currency per unit of
            the book's.
        settled: What the old forward was settled for, in the class's
            currency: gained when positive. 0 when one was opened.
    """
    date: pd.Timestamp
    reason: str
    notional: float
    rate: float
    settled: float


class HedgingMixin:
    """Marks, rolls and resizes each hedged share class's forward."""

    # The engine's own, which BacktestEngine and its other mixins declare.
    currency: str
    data_provider: Any
    modelling_assumptions: ModellingAssumptions
    _ledgers: list[Any]
    _fees_payable: float
    _hedge_value: float

    # Last in BacktestEngine's bases, so this declaration never shadows the
    # AccountingMixin method it stands for.
    def _sell_pro_rata(self,
                       portfolio: Portfolio,
                       amount: float,
                       date: pd.Timestamp,
                       free: bool = False) -> bool:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _invest_inflow(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _start_hedges(self) -> None:
        """No forwards are open, and no rate has been missed yet."""
        self._hedge_value = 0.0
        self._hedge_history: dict[pd.Timestamp, float] = {}
        self._rates_missed: set[str] = set()

    def _mark_hedges(self,
                     date: pd.Timestamp) -> None:
        """Value each open forward at today's rates, the change the class's
        alone."""
        for ledger in self._ledgers:
            forward = ledger.hedge

            if forward is None:
                continue

            worth = self._worth(ledger, forward, date)
            change = worth - forward.value
            forward.value = worth
            ledger.net += change
            self._hedge_value += change

    def _adjust_hedges(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Open, roll or resize each hedged class's forward, after the day's
        flows and trades. A gain settled is invested like an inflow, so the
        pool the other classes share is not left holding the hedged class's
        cash."""
        received = 0.0

        for ledger in self._ledgers:
            if not ledger.share_class.hedged or ledger.net <= 0.0:
                continue

            forward = ledger.hedge
            target = ledger.share_class.hedge_ratio

            if forward is None:
                received += self._open(ledger, portfolio, date, OPEN, _next_roll(date))
            elif date >= forward.expiry:
                received += self._open(ledger, portfolio, date, ROLL, _next_roll(date))
            elif (ledger.share_class.hedge_band is not None
                  and abs(forward.notional / ledger.net - target)
                  > ledger.share_class.hedge_band):
                received += self._open(ledger, portfolio, date, BAND, forward.expiry)

        if received > 0.0:
            self._invest_inflow(portfolio, date)

    def _record_hedges(self,
                       date: pd.Timestamp) -> None:
        """Each hedged class's forward value at the close, in its currency,
        and all of them together in the book's."""
        if any(ledger.share_class.hedged for ledger in self._ledgers):
            self._hedge_history[date] = self._hedge_value

        for ledger in self._ledgers:
            if ledger.share_class.hedged:
                value = ledger.hedge.value if ledger.hedge is not None else 0.0
                ledger.hedge_history[date] = value * ledger.fx

    def _open(self,
              ledger: Any,
              portfolio: Portfolio,
              date: pd.Timestamp,
              reason: str,
              expiry: pd.Timestamp) -> float:
        """Settle the class's forward, if it has one, and open another to
        *expiry* sized to its net assets. Returns what the settlement
        received, in the book's currency: 0 for a loss."""
        settled = self._settle(ledger, portfolio, date)
        notional = ledger.share_class.hedge_ratio * max(ledger.net, 0.0)
        rate = self._forward_rate(ledger, date, expiry)
        ledger.hedge = Forward(notional=notional, rate=rate, expiry=expiry)
        ledger.hedge_trades.append(HedgeTrade(date=date, reason=reason, notional=notional,
                                              rate=rate, settled=settled * ledger.fx))

        logger.debug("[%s] %s hedge %s: %.2f sold forward to %s at %.6f.", date.date(),
                     ledger.share_class.name, reason, notional, expiry.date(), rate)

        return max(settled, 0.0)

    def _settle(self,
                ledger: Any,
                portfolio: Portfolio,
                date: pd.Timestamp) -> float:
        """Move the forward's value into the book's cash. Returns it, in the
        book's currency."""
        forward = ledger.hedge

        if forward is None:
            return 0.0

        value = forward.value

        if value > 0.0:
            portfolio.receive_cash(value, date, HEDGE)
        elif value < 0.0:
            short = -value - (portfolio.cash_balance - self._fees_payable)

            if short > 0.0:
                self._sell_pro_rata(portfolio, short, date)

            paid = min(-value, portfolio.cash_balance)
            portfolio.receive_cash(-paid, date, HEDGE)

            if paid < -value - SHORTFALL:
                logger.warning("[%s] %s's hedge loss of %.2f could not be paid in full.",
                               date.date(), ledger.share_class.name, -value)

        self._hedge_value -= value
        ledger.hedge = None

        return float(value)

    def _worth(self,
               ledger: Any,
               forward: Forward,
               date: pd.Timestamp) -> float:
        """What *forward* is worth to the class on *date*, in the book's
        currency."""
        days = max((forward.expiry - date).days, 0)
        current = self._forward_rate(ledger, date, forward.expiry)
        discount = 1.0 + self._interest(ledger.currency, date, days)
        in_class_currency = forward.notional * (forward.rate - current) / discount

        return float(in_class_currency / ledger.fx)

    def _forward_rate(self,
                      ledger: Any,
                      date: pd.Timestamp,
                      expiry: pd.Timestamp) -> float:
        """The forward rate to *expiry*, class currency per unit of the
        book's, at covered interest parity."""
        days = max((expiry - date).days, 0)

        return float(ledger.fx * (1.0 + self._interest(ledger.currency, date, days))
                     / (1.0 + self._interest(self.currency, date, days)))

    def _interest(self,
                  currency: str,
                  date: pd.Timestamp,
                  days: int) -> float:
        """Interest on one unit of *currency* over *days* calendar days from
        *date*: its short rate in the data on its day count, else the run's
        cash rate on ACT/360, which is said once."""
        read = getattr(self.data_provider, "short_rate", None)
        described = read(currency) if read is not None else None

        if described is not None:
            rate = self.data_provider.short_rate_on(currency, date)

            if rate is not None:
                return float(rate) * days / float(described.day_count)

        if currency not in self._rates_missed:
            self._rates_missed.add(currency)
            logger.warning("No %s short rate in the data; the hedge prices %s at "
                           "the run's cash rate.", currency, currency)

        return (self.modelling_assumptions.cash_rate or 0.0) * days / FORWARD_DAY_COUNT


def _next_roll(date: pd.Timestamp) -> pd.Timestamp:
    """The first business day of the next month: when a forward opened on
    *date* settles."""
    return pd.Timestamp(date + pd.offsets.BMonthBegin(1))
