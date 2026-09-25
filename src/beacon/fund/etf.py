# src/beacon/fund/etf.py
"""
ETF: an exchange-traded fund, an IndexFund with a ticker, a creation unit size
and a simulated market price.
"""
import logging
from typing import Any

import pandas as pd

from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator
from ..index.constructor import IndexDefinition
from ..portfolio.base import Portfolio
from .base import IndexFund

logger = logging.getLogger(__name__)

class ETF(IndexFund):
    """
    Represents an Exchange Traded Fund (ETF), which is a type of IndexFund
    with a ticker, a creation/redemption unit size and a market price.
    """
    def __init__(self,
                 fund_id: str,
                 etf_ticker: str,
                 target_index_definition: IndexDefinition,
                 index_agent: IndexCalculator,
                 portfolio: Portfolio,
                 data_provider: DataFetcher,
                 management_fee_bps: int = 0,
                 creation_unit_size: int = 50000): # Typical size of a creation unit
        """
        Initializes an ETF.

        Args:
            fund_id: A unique identifier for the fund.
            etf_ticker: The market ticker symbol for the ETF.
            target_index_definition: The definition of the index the ETF tracks.
            index_agent: An IndexCalculator for the target index. Only its
                ``price_column`` is read.
            portfolio: The Portfolio seeding the ETF's capital: its cash
                balance is the backtest's initial capital. It is never
                mutated.
            data_provider: DataFetcher for market data.
            management_fee_bps: Annual management fee in basis points.
            creation_unit_size: The number of ETF shares in a creation/redemption unit.

        Raises:
            ValueError: If an argument is missing or empty,
                *management_fee_bps* is negative, or *creation_unit_size* is
                not positive.
        """
        super().__init__(fund_id=fund_id,
                         target_index_definition=target_index_definition,
                         index_agent=index_agent,
                         portfolio=portfolio,
                         data_provider=data_provider,
                         management_fee_bps=management_fee_bps)
        if not etf_ticker:
            raise ValueError("etf_ticker cannot be empty.")
        if creation_unit_size <= 0:
            raise ValueError("creation_unit_size must be positive.")

        self.etf_ticker: str = etf_ticker
        self.creation_unit_size: int = creation_unit_size
        self.market_price: float | None = None # Simulated or actual market price

    def simulate_market_price(self,
                              current_date: pd.Timestamp,
                              market_factors: dict[str, Any] | None = None) -> float:
        """
        Simulates the ETF's market price and stores it on :attr:`market_price`.

        The price is currently the fund's fee-adjusted NAV from
        :meth:`calculate_nav`, so it tracks NAV perfectly: no premium,
        discount or bid-ask spread is modelled. The NAV is the fund's total
        value, not a per-share figure.

        Args:
            current_date: The date for which to simulate the price.
            market_factors: Accepted for future use and currently ignored.

        Returns:
            The simulated market price of the ETF.
        """
        # Future scope: market price might be NAV plus some noise or bid-ask
        # spread, or be supplied externally when backtesting against actual
        # ETF data.
        nav_per_share = self.calculate_nav(current_date) # Assuming NAV is total value.
        # If NAV per share requires number of ETF shares outstanding:
        # num_etf_shares = self.portfolio.get_total_shares() # Needs implementation if ETF
        # shares tracked
        # nav_per_share = self.calculate_nav(current_date) / num_etf_shares if num_etf_shares
        # else nav_per_share

        # Simplistic simulation: market price = NAV (perfect tracking for now)
        self.market_price = nav_per_share
        logger.debug(f"Simulated market price for ETF '{self.etf_ticker}' on "
                     f"{current_date.strftime('%Y-%m-%d')}: {self.market_price:.2f} (based on NAV)")
        # Add more complex logic here later, e.g., premium/discount simulation
        return self.market_price


    def get_tracking_performance(self,
                                 result: BacktestResult) -> dict[str, float | str]:
        """Calculate tracking metrics from a completed backtest.

        Compares the backtest's ``trading_nav`` against the levels of the
        index the run tracked (``result.index.tracked``) using the tracking
        methods built into :class:`~beacon.backtest.result.BacktestResult`.
        The run must have tracked an index for the comparison to be possible.

        Args:
            result: A BacktestResult produced by tracking this ETF's index. It
                already contains both the portfolio NAV and the target index.

        Returns:
            A dictionary with float ``tracking_error`` and
            ``tracking_difference`` entries. If the result has no target index
            to compare against, a single ``error`` entry is returned instead,
            whose value is an explanatory string (hence the ``float | str``
            value type).

        Raises:
            ValueError: If *result* is None.
        """
        if result is None:
            raise ValueError("A BacktestResult must be provided.")

        logger.info(f"Calculating tracking performance for ETF '{self.etf_ticker}'.")

        tracking_err = result.get_tracking_error()
        tracking_diff = result.get_tracking_difference()

        if tracking_err is None or tracking_diff is None:
            logger.error(
                f"BacktestResult for ETF '{self.etf_ticker}' has no target index "
                "to compare against."
            )
            return {"error": "BacktestResult has no target index for tracking comparison."}

        logger.info(
            f"Tracking performance for '{self.etf_ticker}': "
            f"TE={tracking_err:.4f}, TD={tracking_diff:.4f}"
        )
        return {
            "tracking_error": tracking_err,
            "tracking_difference": tracking_diff,
        }

    def __repr__(self) -> str:
        return (f"ETF(fund_id='{self.fund_id}', etf_ticker='{self.etf_ticker}', "
                f"target_index='{self.target_index_definition.index_name}', "
                f"management_fee_bps={self.management_fee_bps}, "
                f"creation_unit_size={self.creation_unit_size})")
