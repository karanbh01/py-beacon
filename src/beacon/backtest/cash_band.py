# src/beacon/backtest/cash_band.py
"""
Keeping a fund's cash within a band around its target, every day.

A backtest's `Implementation.cash_buffer` is the share of the book it means to
hold in cash, and each rebalance invests the rest. Between rebalances the cash
moves: a delisted holding is settled into it, a dividend kept as cash lands in
it, a fee or a futures margin call is paid from it. With a `cash_band` the run
checks it at the close of every day: above the band's upper edge the cash is
invested back down to the target, below its lower edge holdings are sold pro
rata to bring it back up, and inside the band nothing is traded. Without one,
the cash waits for the next rebalance, as it always has.
"""
# BN-358: on the TTIX setup at 600 names, seven held names (4.7% of the fund)
# delisted into the March 2020 crash and their cash waited for the April
# rebalance while the index reinvested it at once.
from dataclasses import dataclass

import pandas as pd

from ..portfolio import reasons
from ..portfolio.base import Portfolio
from .implementation import Implementation

# BN-407: what each event's kind is called.
RAISE = "raise"
SPEND_DOWN = "spend-down"


@dataclass(frozen=True)
class CashBandEvent:
    """One time the cash left its band and was brought back.

    Attributes:
        date: The day.
        kind: ``raise`` when the cash fell below the band and holdings were
            sold, ``spend-down`` when it rose above and was invested.
        cash_before: The cash's share of the fund at the close, before.
        cash_after: Its share after the trades.
        target: The share it was brought back to: the cash buffer.
        lower: The band's lower edge.
        upper: The band's upper edge.
        traded: The value bought or sold.
        cost: What the trades cost.
        trades: How many trades it took.
    """
    date: pd.Timestamp
    kind: str
    cash_before: float
    cash_after: float
    target: float
    lower: float
    upper: float
    traded: float
    cost: float
    trades: int


def band_events_frame(events: list[CashBandEvent]) -> pd.DataFrame:
    """The events as a frame indexed by date."""
    columns = [name for name in CashBandEvent.__dataclass_fields__ if name != "date"]

    if not events:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name="date"))

    frame = pd.DataFrame([event.__dict__ for event in events]).set_index("date")

    return frame[columns]


class CashBandMixin:
    """Holds the cash within its band between rebalances, mixed into
    BacktestEngine."""

    implementation: Implementation
    _fees_payable: float
    _band_events: list[CashBandEvent]

    def _net_value(self,
                   portfolio: Portfolio) -> float:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _invest_inflow(self,
                       portfolio: Portfolio,
                       date: pd.Timestamp) -> None:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _sell_pro_rata(self,
                       portfolio: Portfolio,
                       amount: float,
                       date: pd.Timestamp,
                       free: bool = False) -> bool:
        """Provided by AccountingMixin; declared so this one type-checks."""
        raise NotImplementedError

    def _update_portfolio_prices(self,
                                 portfolio: Portfolio,
                                 date: pd.Timestamp) -> None:
        """Provided by BacktestEngine; declared so this one type-checks."""
        raise NotImplementedError

    def _keep_cash_in_band(self,
                           portfolio: Portfolio,
                           date: pd.Timestamp) -> bool:
        """Bring the cash back to its target if it has left the band.

        Returns:
            bool: Whether anything was traded, so the futures that equitise
            the cash can be resized to it.
        """
        band = self.implementation.cash_band
        net = self._net_value(portfolio)

        if band is None or net <= 0.0:
            return False

        lower, upper = band
        cash = portfolio.cash_balance - self._fees_payable
        share = cash / net

        if lower <= share <= upper:
            return False

        made = len(portfolio.transactions)

        if share > upper:
            # Invests what is above the target, as money arriving is.
            with portfolio.trading(reasons.CASH_SPEND_DOWN):
                self._invest_inflow(portfolio, date)
        else:
            with portfolio.batch(), portfolio.trading(reasons.CASH_RAISE):
                self._sell_pro_rata(portfolio, self.implementation.cash_buffer * net - cash,
                                    date)

        self._update_portfolio_prices(portfolio, date)
        self._record_band_event(portfolio, date, share, made)

        return True

    def _record_band_event(self,
                           portfolio: Portfolio,
                           date: pd.Timestamp,
                           before: float,
                           made: int) -> None:
        """Record the event just traded: the trades from *made* on, and the
        cash's share before (*before*) and now."""
        lower, upper = self.implementation.cash_band or (0.0, 0.0)
        new = portfolio.transactions[made:]
        net = self._net_value(portfolio)
        after = (portfolio.cash_balance - self._fees_payable) / net if net > 0.0 else 0.0

        self._band_events.append(CashBandEvent(
            date=date, kind=SPEND_DOWN if before > upper else RAISE,
            cash_before=before, cash_after=after, target=self.implementation.cash_buffer,
            lower=lower, upper=upper,
            traded=sum(trade.quantity * trade.price for trade in new),
            cost=sum(trade.transaction_cost for trade in new), trades=len(new)))
