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
import pandas as pd

from ..portfolio.base import Portfolio
from .implementation import Implementation


class CashBandMixin:
    """Holds the cash within its band between rebalances, mixed into
    BacktestEngine."""

    implementation: Implementation
    _fees_payable: float

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

        if share > upper:
            # Invests what is above the target, as money arriving is.
            self._invest_inflow(portfolio, date)
        else:
            with portfolio.batch():
                self._sell_pro_rata(portfolio, self.implementation.cash_buffer * net - cash,
                                    date)

        self._update_portfolio_prices(portfolio, date)

        return True
