# src/beacon/backtest/futures.py
"""
Index futures held beside the books: for a futures-based fund, its whole
exposure; for a physical fund, the cash buffer it equitises.

The position's profit and loss is settled in cash every day, as variation
margin is, so the fund's net assets need nothing added for it: the futures
are worth nothing after each day's settlement, and what they made or lost is
already in the cash. What is recorded is the notional they carried, the rolls
and what each cost.

Each contract is priced at fair value, ``F = S x exp((r - q) x T)``: S the
tracked index's level, r the run's cash rate, q the index's dividend yield
and T the time to expiry over calendar days (ACT/365). See
`beacon.strategy.futures`.

The position opens at the first rebalance, rolls to the next contract the set
number of sessions before expiry (paying the roll cost on its notional), and
is resized to its target at each rebalance, each roll, on any day money flows
in or out, and on any other day its exposure has drifted from the target by
more than the band: the fund's net assets for a futures-based fund, its
uninvested cash for one equitising. Each contract is worth its price times
the multiplier, and the contracts are whole unless the terms say otherwise.

Every trade is recorded with what it cost: per contract, and in basis points
of its notional between rolls (see `beacon.strategy.futures`); so is each
day's position, its price and the fair value it was priced from.
"""
# BN-305. Variation margin rather than a valued position: the cash already
# moves with the futures, so the NAV needs no second term (unlike the swap of
# BN-270, which settles only at resets).
import logging
from dataclasses import dataclass

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..data.fetcher import DataFetcher
from ..derivatives.pricing import cost_of_carry_fair_value
from ..index.result import IndexResult
from ..portfolio import reasons
from ..portfolio.base import Portfolio
from ..portfolio.cash_flows import FUTURES
from ..strategy.futures import IndexFutures, expiries
from .histories import history
from .implementation import Implementation

logger = logging.getLogger(__name__)

# Calendar days in the year the time to expiry is measured over.
DAYS_A_YEAR = 365.0


@dataclass(frozen=True)
class FuturesRoll:
    """One roll of a fund's index futures to the next contract.

    Attributes:
        date: The day it rolled.
        expiry: The expiry of the contract rolled into.
        notional: The exposure carried into the new contract.
        cost: What the roll cost, paid from cash.
        dividend_yield: The dividend yield the new contract is priced with.
        contracts: How many contracts of the new expiry were held; the
            contracts times the price times the multiplier is the notional.
        price: The new contract's price, at fair value.
        multiplier: The value of one index point per contract.
    """
    date: pd.Timestamp
    expiry: pd.Timestamp
    notional: float
    cost: float
    dividend_yield: float
    # BN-315: for the API's record of each roll.
    contracts: float = 0.0
    price: float = 0.0
    # BN-358.
    multiplier: float = 1.0


# BN-409: why a futures trade was made.
OPEN = "open"
ROLL = "roll"
RESIZE = "resize"


@dataclass(frozen=True)
class FuturesTrade:
    """One trade of a fund's index futures.

    Attributes:
        date: The day.
        reason: ``open``, ``roll`` (the old contract sold and the next
            bought) or ``resize`` (the position brought back to its target).
        before: The contracts held before.
        after: The contracts held after.
        traded: The contracts traded: for a roll both legs, the old contracts
            and the new.
        price: The price of the contract held after.
        notional: The notional traded: the contracts traded times the price
            times the multiplier.
        cost: What the trade cost, paid from cash: for a roll, the roll cost
            and the cost per contract.
    """
    date: pd.Timestamp
    reason: str
    before: float
    after: float
    traded: float
    price: float
    notional: float
    cost: float


def futures_trades_frame(trades: list[FuturesTrade]) -> pd.DataFrame:
    """The trades as a frame indexed by date."""
    columns = [name for name in FuturesTrade.__dataclass_fields__ if name != "date"]

    if not trades:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    return pd.DataFrame([trade.__dict__ for trade in trades]).set_index("date")[columns]


def futures_days_frame(days: dict[pd.Timestamp, dict[str, float]]) -> pd.DataFrame:
    """Each day's position as a frame indexed by date."""
    frame = pd.DataFrame.from_dict(days, orient="index",
                                   columns=["contracts", "price", "fair_value", "level",
                                            "expiry"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    frame.index = pd.DatetimeIndex(frame.index, name="date")

    return frame


class FuturesMixin:
    """Marks, rolls and sizes a fund's index futures."""

    # The engine's own, which BacktestEngine and its other mixins declare.
    futures: IndexFutures | None
    implementation: Implementation
    index_result: IndexResult
    data_provider: DataFetcher
    modelling_assumptions: ModellingAssumptions
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    _fees_payable: float

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

    @property
    def _futures_terms(self) -> IndexFutures | None:
        """The futures this run holds: a futures-based fund's, or the ones
        equitising a buffer."""
        return self.futures or self.implementation.equitise

    def _start_futures(self) -> None:
        """Forget any earlier run's position; none is held until the first
        rebalance."""
        self._futures_open = False
        self._contracts = 0.0
        self._futures_price = 0.0
        self._futures_yield = 0.0
        self._expiry = pd.Timestamp(self.start_date)
        self._futures_notional: dict[pd.Timestamp, float] = {}
        self._futures_rolls: list[FuturesRoll] = []
        self._futures_trades: list[FuturesTrade] = []
        self._futures_days: dict[pd.Timestamp, dict[str, float]] = {}
        self._expiries = expiries(self.start_date, self.end_date)
        self._futures_levels = self.index_result.index_levels.sort_index()

    def _mark_futures(self,
                      portfolio: Portfolio,
                      date: pd.Timestamp) -> None:
        """Settle the day's profit or loss on the position in cash."""
        if not self._futures_open:
            return

        price = self._fair_value(date, self._expiry)
        self._settle_margin(portfolio, self._contracts * (price - self._futures_price)
                            * self._multiplier, date)
        self._futures_price = price

    def _adjust_futures(self,
                        portfolio: Portfolio,
                        date: pd.Timestamp,
                        rebalancing: bool,
                        flowed: bool) -> None:
        """Open, roll or resize the position, after the day's trades."""
        terms = self._futures_terms

        if terms is None or not (self._futures_open or rebalancing):
            return

        rolling = self._futures_open and date >= self._roll_date(self._expiry, terms)
        opening = not self._futures_open
        before = self._contracts

        if opening or rolling:
            self._expiry = next(expiry for expiry in self._expiries
                                if self._roll_date(expiry, terms) > date)
            self._futures_yield = self._dividend_yield(terms, date)
            self._futures_price = self._fair_value(date, self._expiry)
            self._futures_open = True

        # A roll's cost is paid before the new contracts are sized, so they
        # carry what the fund has left after paying it.
        cost = (self._futures_target(portfolio) * terms.roll_cost_bps / 10_000.0
                if rolling else 0.0)

        if cost > 0.0:
            self._settle_margin(portfolio, -cost, date)

        # Between those days, resized once the exposure has drifted from its
        # target by more than the band (BN-358): delisting proceeds, a
        # dividend kept as cash or a fee all move the cash the futures cover.
        if (rolling or rebalancing or flowed or self._contracts == 0.0
                or self._outside_band(portfolio, terms)):
            self._contracts = terms.contracts(self._futures_target(portfolio),
                                              self._futures_price)

        traded = before + self._contracts if rolling else abs(self._contracts - before)

        if traded > 0.0:
            trading = terms.trade_cost(traded, self._futures_price, rolling)

            if trading > 0.0:
                self._settle_margin(portfolio, -trading, date)

            cost += trading
            self._futures_trades.append(FuturesTrade(
                date=date, reason=ROLL if rolling else OPEN if opening else RESIZE,
                before=before, after=self._contracts, traded=traded,
                price=self._futures_price,
                notional=traded * self._futures_price * terms.multiplier, cost=cost))

        if rolling:
            notional = self._contracts * self._futures_price * terms.multiplier
            self._futures_rolls.append(FuturesRoll(date=date, expiry=self._expiry,
                                                   notional=notional, cost=cost,
                                                   dividend_yield=self._futures_yield,
                                                   contracts=self._contracts,
                                                   price=self._futures_price,
                                                   multiplier=terms.multiplier))
            logger.debug("[%s] Futures rolled to %s: notional %.2f, cost %.2f.",
                         date.date(), self._expiry.date(), notional, cost)

    def _record_futures(self,
                        date: pd.Timestamp) -> None:
        """Record the notional the position carried at the close."""
        if self._futures_terms is not None:
            self._futures_notional[date] = (self._contracts * self._futures_price
                                            * self._multiplier)

        if self._futures_open:
            self._futures_days[date] = {
                "contracts": self._contracts, "price": self._futures_price,
                "fair_value": self._fair_value(date, self._expiry),
                "level": float(self._futures_levels.asof(date)),
                "expiry": self._expiry.value}

    @property
    def _multiplier(self) -> float:
        """The value of one index point per contract, 1 with no futures."""
        terms = self._futures_terms

        return terms.multiplier if terms is not None else 1.0

    def _outside_band(self,
                      portfolio: Portfolio,
                      terms: IndexFutures) -> bool:
        """Whether the exposure is further from its target than the band."""
        exposure = self._contracts * self._futures_price * terms.multiplier
        gap = abs(exposure - self._futures_target(portfolio))

        return gap > terms.band * max(self._net_value(portfolio), 0.0)

    def _futures_target(self,
                        portfolio: Portfolio) -> float:
        """What the position should carry: the fund's net assets for a
        futures-based fund, its uninvested cash for one equitising."""
        if self.futures is not None:
            return max(self._net_value(portfolio), 0.0)

        return max(portfolio.cash_balance - self._fees_payable, 0.0)

    def _settle_margin(self,
                       portfolio: Portfolio,
                       amount: float,
                       date: pd.Timestamp) -> None:
        """Receive or pay variation margin, selling holdings pro rata for a
        payment the cash does not cover."""
        if amount >= 0.0:
            portfolio.receive_cash(amount, date, FUTURES)
            return

        short = -amount - (portfolio.cash_balance - self._fees_payable)

        if short > 0.0:
            with portfolio.trading(reasons.SETTLEMENT):
                self._sell_pro_rata(portfolio, short, date)

        paid = min(-amount, portfolio.cash_balance)

        if paid < -amount - 1e-6:
            logger.warning("[%s] Futures margin of %.2f could not be paid in full; "
                           "%.2f went unpaid.", date.date(), -amount, -amount - paid)

        portfolio.receive_cash(-paid, date, FUTURES)

    def _fair_value(self,
                    date: pd.Timestamp,
                    expiry: pd.Timestamp) -> float:
        """A contract's fair value on *date*."""
        level = self._futures_levels.asof(date)

        if pd.isna(level):
            level = self._futures_levels.iloc[0]

        return cost_of_carry_fair_value(
            spot=float(level),
            risk_free_rate=self.modelling_assumptions.cash_rate or 0.0,
            dividend_yield=self._futures_yield,
            time_to_expiry_years=max((expiry - date).days, 0) / DAYS_A_YEAR)

    @staticmethod
    def _roll_date(expiry: pd.Timestamp,
                   terms: IndexFutures) -> pd.Timestamp:
        """The day a position in the contract expiring on *expiry* rolls."""
        return pd.Timestamp(expiry - pd.offsets.BDay(terms.roll_days))

    def _dividend_yield(self,
                        terms: IndexFutures,
                        date: pd.Timestamp) -> float:
        """The dividend yield a new contract is priced with: the one given,
        or the index constituents' trailing twelve months at their weights."""
        if terms.dividend_yield is not None:
            return terms.dividend_yield

        snapshots = {day: weights for day, weights in self.index_result.weight_snapshots.items()
                     if day <= date}
        trailing = getattr(self.data_provider, "fetch_trailing_dividend_yield", None)

        if not snapshots or trailing is None:
            return 0.0

        weights = snapshots[max(snapshots)]
        total = sum(weights.values())

        if total <= 0.0:
            return 0.0

        return sum(weight * (trailing(name, date, self._close_for_yield(name, date)) or 0.0)
                   for name, weight in weights.items()) / total

    def _close_for_yield(self,
                         name: str,
                         date: pd.Timestamp) -> float | None:
        """*name*'s last close on or before *date*, from its history read once
        per run (BN-346), as the yield read it from the data; None leaves the
        data provider to find it."""
        if not isinstance(self.data_provider, DataFetcher):
            return None

        return history(self.data_provider, name, None).last_close(date)
