# src/beacon/backtest/dividends.py
"""
Cash distributions: what a backtest's holdings are paid, and what happens to it.

A holding is entitled to a cash distribution if it is held at the start of the
ex-date, before that day's trades. The cash arrives on the pay date, or on the
ex-date when the data gives none, converted into the book's currency on the
day it arrives and net of the withholding rate in the run's modelling
assumptions. A holding sold between the two dates is still paid.

What happens to the cash is the run's dividend policy:

- ``"accumulate"`` (the default): it stays in the book as cash and is invested
  at the next rebalance.
- ``"reinvest"``: it buys more of the current holdings, in proportion to their
  value, the day it arrives.
- ``"distribute"``: it is paid out of the book. The NAV falls by it and the
  returns add it back, so performance is still measured as total return.

Every payment is a `CashFlow` on `portfolio.cash_flows`: a ``DIVIDEND`` for
each name paid, and a ``DISTRIBUTION`` (negative) when it is paid out.

The price path must already contain the ex-date drop, as an unadjusted feed
and Beacon's synthetic data do. Paying cash against a price that never fell
would count the distribution twice.
"""
# BN-263, phase 1 of decisions/0006. Before this a backtest earned the price
# return only and trailed a total-return index by about the dividend yield.
import logging
from dataclasses import dataclass

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..data.corporate_actions import CASH, PAY_DATE_COLUMN, kind_of, without_cancelled
from ..data.fetcher import DataFetcher
from ..portfolio.base import Portfolio, TradeInstruction
from ..portfolio.cash_flows import DISTRIBUTION, DIVIDEND

logger = logging.getLogger(__name__)

ACCUMULATE = "accumulate"
REINVEST = "reinvest"
DISTRIBUTE = "distribute"
DIVIDEND_POLICIES = (ACCUMULATE, REINVEST, DISTRIBUTE)


@dataclass(frozen=True)
class CashEvent:
    """One cash distribution in the data: who pays, how much, when."""
    asset_id: str
    per_share: float
    ex_date: pd.Timestamp
    pay_date: pd.Timestamp


@dataclass(frozen=True)
class Entitlement:
    """A distribution the book is owed: an event and the shares held for it."""
    event: CashEvent
    shares: float


def cash_events(data_provider: DataFetcher) -> dict[pd.Timestamp, list[CashEvent]]:
    """Every cash distribution the data holds, by ex-date.

    Cancelled distributions are left out. A pay date before its ex-date is
    read as the ex-date, since nothing can be paid before it is owed.
    """
    actions = getattr(data_provider, "corporate_actions", None)

    if actions is None or actions.is_empty:
        return {}

    frame = without_cancelled(actions.data.reset_index(drop=True))
    cash = frame[frame["TYPE"].map(lambda value: kind_of(value) == CASH)]
    events: dict[pd.Timestamp, list[CashEvent]] = {}

    for _, row in cash.iterrows():
        ex_date = pd.Timestamp(row["EX_DATE"])
        paid = row.get(PAY_DATE_COLUMN) if PAY_DATE_COLUMN in cash.columns else None
        pay_date = (max(pd.Timestamp(paid), ex_date) if pd.notna(paid)
                    else ex_date)

        events.setdefault(ex_date, []).append(
            CashEvent(asset_id=str(row["IDENTIFIER"]),
                      per_share=float(row["VALUE"]),
                      ex_date=ex_date, pay_date=pay_date))

    return events


class DividendsMixin:
    """Dividend entitlement and payment, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    data_provider: DataFetcher
    dividends: str
    modelling_assumptions: ModellingAssumptions
    transaction_cost_bps: float

    def _rate_for(self,
                  asset_id: str,
                  date: pd.Timestamp,
                  rates: dict[str, float] | None = None) -> float:
        """Provided by PricingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _fetch_price(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float | None:
        """Provided by PricingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _start_dividends(self) -> None:
        """Read the distributions once and clear what a previous run owed."""
        self._cash_events = cash_events(self.data_provider)
        self._owed: list[Entitlement] = []

    def _record_entitlements(self,
                             portfolio: Portfolio,
                             previous: pd.Timestamp,
                             date: pd.Timestamp) -> None:
        """Note what the book is owed for ex-dates after *previous*, up to
        *date*, for the shares held now, before the day's trades."""
        for ex_date, events in self._cash_events.items():
            if not previous < ex_date <= date:
                continue

            for event in events:
                holding = portfolio.holdings.get(event.asset_id)

                if holding is not None and holding.quantity > 0:
                    self._owed.append(Entitlement(event, holding.quantity))

    def _pay_dividends(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Receive what is due by *date*, then apply the dividend policy."""
        due = [owed for owed in self._owed if owed.event.pay_date <= date]

        if not due:
            return

        self._owed = [owed for owed in self._owed if owed.event.pay_date > date]
        kept = 1.0 - (self.modelling_assumptions.withholding_tax_rate or 0.0)
        rates: dict[str, float] = {}
        total = 0.0

        for owed in due:
            event = owed.event
            amount = (owed.shares * event.per_share
                      * self._rate_for(event.asset_id, date, rates) * kept)
            portfolio.receive_cash(amount, date, DIVIDEND, event.asset_id)
            total += amount

        if self.dividends == DISTRIBUTE:
            portfolio.receive_cash(-total, date, DISTRIBUTION)
        elif self.dividends == REINVEST:
            self._reinvest(portfolio, total, date)

    def _reinvest(self,
                  portfolio: Portfolio,
                  amount: float,
                  date: pd.Timestamp) -> None:
        """Buy the current holdings with *amount*, in proportion to value."""
        values = {name: holding.quantity * price
                  for name, holding in portfolio.holdings.items()
                  if holding.quantity > 0
                  and (price := self._fetch_price(name, date)) is not None
                  and price > 0}
        held = sum(values.values())

        if held <= 0.0 or amount <= 0.0:
            return

        cost_rate = self.transaction_cost_bps / 10_000.0

        for name, value in values.items():
            price = value / portfolio.holdings[name].quantity
            spend = amount * value / held
            quantity = spend / (price * (1.0 + cost_rate))

            portfolio.apply(TradeInstruction(name, "BUY", quantity, price,
                                             quantity * price * cost_rate), date)
