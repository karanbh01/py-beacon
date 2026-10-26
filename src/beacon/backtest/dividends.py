# src/beacon/backtest/dividends.py
"""
Cash distributions: what a backtest's holdings are paid, and what happens to it.

A holding is entitled to a cash distribution if it is held at the start of the
ex-date, before that day's trades. The cash arrives on the pay date, or on the
ex-date when the data gives none, converted into the book's currency on the
day it arrives and net of withholding tax: by the fund's domicile and the
dividend's source country when the vehicle has a domicile (see
`beacon.backtest.withholding`), otherwise at the flat rate in the run's
modelling assumptions. A holding sold between the two dates is still paid.

What happens to the cash is the run's dividend policy:

- ``"reinvest"`` (the default): it is invested the day it arrives. Where is
  the run's ``reinvest_into``: ``"payer"`` (the default) buys more of the name
  that paid it, and ``"holdings"`` buys the current holdings in proportion to
  their value, as a total-return index assumes. A payer no longer held, or
  with no price that day, leaves its dividend in cash.
- ``"cash"``: it stays in the book as cash and is invested at the next
  rebalance.
- ``"distribute"``: it is paid out of the book. The NAV falls by it and the
  returns add it back, so performance is still measured as total return.

``"reinvest"`` and ``"cash"`` both keep the income in the book, as an
accumulating share class does; they differ only in when it is invested.
``"distribute"`` pays it out, as a distributing share class does.

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
from typing import Any

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..data.corporate_actions import CASH, PAY_DATE_COLUMN, kind_of, without_cancelled
from ..data.fetcher import DataFetcher
from ..portfolio.base import Portfolio, TradeInstruction
from ..portfolio.cash_flows import DISTRIBUTION, DIVIDEND
from .vehicle import Vehicle
from .withholding import source_country, withholding_rate

logger = logging.getLogger(__name__)

REINVEST = "reinvest"
KEEP_CASH = "cash"
DISTRIBUTE = "distribute"
DIVIDEND_POLICIES = (REINVEST, KEEP_CASH, DISTRIBUTE)

# Where reinvested dividends go (BN-343): into the name that paid them, the
# default, or across the current holdings in proportion to value.
PAYER = "payer"
HOLDINGS = "holdings"
REINVESTMENT_TARGETS = (PAYER, HOLDINGS)


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


def entitled_frame(history: dict[pd.Timestamp, tuple[float, float]]) -> pd.DataFrame:
    """Each day's gross dividends and tax withheld, as `dividends_entitled`."""
    return pd.DataFrame.from_dict(history, orient="index", columns=["GROSS", "WITHHELD"],
                                  dtype=float).sort_index()


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

    if cash.empty:
        return events

    # From whole columns rather than row by row (BN-348): `iterrows` over the
    # actions of six thousand names took seconds before the run began.
    ex_dates = [pd.Timestamp(value) for value in cash["EX_DATE"]]
    paid = (list(cash[PAY_DATE_COLUMN]) if PAY_DATE_COLUMN in cash.columns
            else [None] * len(cash))

    for name, value, ex_date, pay in zip(cash["IDENTIFIER"].tolist(), cash["VALUE"].tolist(),
                                         ex_dates, paid, strict=True):
        pay_date = max(pd.Timestamp(pay), ex_date) if pd.notna(pay) else ex_date

        events.setdefault(ex_date, []).append(
            CashEvent(asset_id=str(name), per_share=float(value),
                      ex_date=ex_date, pay_date=pay_date))

    return events


class DividendsMixin:
    """Dividend entitlement and payment, mixed into BacktestEngine."""

    # Provided by the BacktestEngine that mixes this in.
    data_provider: DataFetcher
    dividends: str
    modelling_assumptions: ModellingAssumptions
    transaction_cost_bps: float
    vehicle: Vehicle | None
    _ledgers: list[Any]

    def _class_distributions(self,
                             income: float,
                             date: pd.Timestamp) -> float:
        """Provided by ShareClassesMixin; declared so this one type-checks."""
        raise NotImplementedError

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

    # Provided by the BacktestEngine that mixes this in (BN-343).
    reinvest_into: str

    def _prices_for(self,
                    asset_ids: list[str],
                    date: pd.Timestamp) -> dict[str, float | None]:
        """Provided by PricingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _start_dividends(self) -> None:
        """Read the distributions once and clear what a previous run owed."""
        self._cash_events = cash_events(self.data_provider)
        self._owed: list[Entitlement] = []
        self._withheld: dict[str, float] = {}
        self._receivable = 0.0
        self._receivable_history: dict[pd.Timestamp, float] = {}
        # What the book became owed on each day, gross and withheld, at that
        # day's rates (BN-364).
        self._entitled_history: dict[pd.Timestamp, tuple[float, float]] = {}

    def _mark_receivable(self,
                         date: pd.Timestamp) -> None:
        """Value what the book is owed and not yet paid, at *date*'s rates
        and net of withholding, and record it.

        The NAV includes it from the ex-date, as a fund's accounts do: the
        holding falls by the dividend on the ex-date, and leaving the
        dividend out until the pay date made the NAV dip and recover against
        an index that reinvests it on the ex-date. It is not cash, so nothing
        is bought with it until it is paid.
        """
        # BN-356: against Karan's TTIX as a total-return index, tracking
        # error 0.81% with the dividend left out until paid, 0.16% with it
        # counted from the ex-date.
        rates: dict[str, float] = {}
        owed = 0.0

        for entitlement in self._owed:
            event = entitlement.event
            owed += (entitlement.shares * event.per_share
                     * self._rate_for(event.asset_id, date, rates)
                     * (1.0 - self._withholding(event.asset_id, date)))

        self._receivable = owed
        self._receivable_history[date] = owed

    def _record_entitlements(self,
                             portfolio: Portfolio,
                             previous: pd.Timestamp,
                             date: pd.Timestamp) -> None:
        """Note what the book is owed for ex-dates after *previous*, up to
        *date*, for the shares held now, before the day's trades."""
        rates: dict[str, float] = {}
        gross = withheld = 0.0

        for ex_date, events in self._cash_events.items():
            if not previous < ex_date <= date:
                continue

            for event in events:
                holding = portfolio.holdings.get(event.asset_id)

                if holding is not None and holding.quantity > 0:
                    self._owed.append(Entitlement(event, holding.quantity))
                    amount = (holding.quantity * event.per_share
                              * self._rate_for(event.asset_id, date, rates))
                    gross += amount
                    withheld += amount * self._withholding(event.asset_id, date)

        if gross > 0.0:
            self._entitled_history[date] = (gross, withheld)

    def _pay_dividends(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Receive what is due by *date*, then apply the dividend policy."""
        due = [owed for owed in self._owed if owed.event.pay_date <= date]

        if not due:
            return

        self._owed = [owed for owed in self._owed if owed.event.pay_date > date]
        rates: dict[str, float] = {}
        total = 0.0
        paid: dict[str, float] = {}

        for owed in due:
            event = owed.event
            kept = 1.0 - self._withholding(event.asset_id, date)
            amount = (owed.shares * event.per_share
                      * self._rate_for(event.asset_id, date, rates) * kept)
            portfolio.receive_cash(amount, date, DIVIDEND, event.asset_id)
            total += amount
            paid[event.asset_id] = paid.get(event.asset_id, 0.0) + amount

        if self._ledgers:
            # BN-279: the distributing classes' shares are paid out, and the
            # accumulating classes' kept as the run's policy says.
            paid_out = self._class_distributions(total, date)

            if paid_out > 0.0:
                portfolio.receive_cash(-paid_out, date, DISTRIBUTION)

            if self.dividends == REINVEST and total > 0.0:
                self._reinvest(portfolio, paid, (total - paid_out) / total, date)
        elif self.dividends == DISTRIBUTE:
            portfolio.receive_cash(-total, date, DISTRIBUTION)
        elif self.dividends == REINVEST:
            self._reinvest(portfolio, paid, 1.0, date)

    def _withholding(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float:
        """The share of *asset_id*'s dividends withheld from this fund: by
        its domicile and the source country when the vehicle sets either,
        otherwise the run's flat rate. Looked up once per name per run."""
        flat = self.modelling_assumptions.withholding_tax_rate or 0.0
        vehicle = self.vehicle

        if vehicle is None or (vehicle.domicile is None and not vehicle.withholding_rates):
            return flat

        if asset_id not in self._withheld:
            source = source_country(self.data_provider, asset_id, date)
            self._withheld[asset_id] = withholding_rate(vehicle.domicile, source,
                                                        vehicle.withholding_rates, flat)

        return self._withheld[asset_id]

    def _reinvest(self,
                  portfolio: Portfolio,
                  paid: dict[str, float],
                  share: float,
                  date: pd.Timestamp) -> None:
        """Reinvest *share* of what each name in *paid* paid: in that name,
        or across the holdings in proportion to value, as `reinvest_into`
        says."""
        if self.reinvest_into == PAYER:
            self._reinvest_in_payers(portfolio, paid, share, date)
            return

        self._reinvest_in_holdings(portfolio, share * sum(paid.values()), date)

    def _reinvest_in_payers(self,
                            portfolio: Portfolio,
                            paid: dict[str, float],
                            share: float,
                            date: pd.Timestamp) -> None:
        """Buy each paying name with its own dividend. A payer no longer
        held, or with no price, leaves its dividend in cash."""
        payers = [name for name, amount in paid.items()
                  if amount > 0.0 and name in portfolio.holdings
                  and portfolio.holdings[name].quantity > 0]
        prices = self._prices_for(payers, date)
        cost_rate = self.transaction_cost_bps / 10_000.0

        for name in payers:
            price = prices.get(name)

            if price is None or price <= 0.0:
                continue

            quantity = share * paid[name] / (price * (1.0 + cost_rate))
            portfolio.apply(TradeInstruction(name, "BUY", quantity, price,
                                             quantity * price * cost_rate), date)

    def _reinvest_in_holdings(self,
                              portfolio: Portfolio,
                              amount: float,
                              date: pd.Timestamp) -> None:
        """Buy the current holdings with *amount*, in proportion to value."""
        # Priced in one read (BN-341), with the answers `_fetch_price` gives
        # name by name: it was one read per holding on every payment day.
        held_names = [name for name, holding in portfolio.holdings.items() if holding.quantity > 0]
        prices = self._prices_for(held_names, date)
        values = {name: portfolio.holdings[name].quantity * price
                  for name in held_names
                  if (price := prices.get(name)) is not None and price > 0}
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
