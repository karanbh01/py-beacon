# beacon/fund/base.py
"""
Module defining the IndexFund class.
"""
import pandas as pd
from typing import Optional
import logging

from ..index.constructor import IndexDefinition
from ..portfolio.base import Portfolio
from ..data.fetcher import DataFetcher
from ..index.calculation import IndexCalculator
from ..index.result import IndexResult
from ..backtest.result import BacktestResult
from ..backtest.engine import BacktestEngine


logger = logging.getLogger(__name__)


class IndexFund:
    """An index fund that tracks a target index.

    The fund composes an :class:`~beacon.index.calculation.IndexCalculator`
    (to compute the target weight schedule) and a
    :class:`~beacon.backtest.engine.BacktestEngine` (to simulate the tracking
    portfolio). It contains no buy/sell logic of its own — rebalancing and
    portfolio accounting are delegated entirely to the backtest engine.
    """

    def __init__(self,
                 fund_id: str,
                 target_index_definition: IndexDefinition,
                 index_agent: IndexCalculator,
                 portfolio: Portfolio,
                 data_provider: DataFetcher,
                 management_fee_bps: int = 0):
        """
        Initializes an IndexFund.

        Args:
            fund_id: A unique identifier for the fund.
            target_index_definition: The definition of the index the fund aims to track.
            index_agent: The IndexCalculator used to compute the target index's
                         weight schedule.
            portfolio: The Portfolio object seeding the fund's capital. Its cash
                       balance is used as the backtest engine's initial capital;
                       the fund no longer mutates this portfolio directly.
            data_provider: DataFetcher instance for market data.
            management_fee_bps: The annual management fee in basis points (e.g., 10 bps = 0.1%).
        """
        if not fund_id:
            raise ValueError("fund_id cannot be empty.")
        if not target_index_definition:
            raise ValueError("target_index_definition must be provided.")
        if not index_agent:
            raise ValueError("index_agent must be provided.")
        if not portfolio:
            raise ValueError("portfolio must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")
        if management_fee_bps < 0:
            raise ValueError("management_fee_bps cannot be negative.")

        self.fund_id: str = fund_id
        self.target_index_definition: IndexDefinition = target_index_definition
        self.index_agent: IndexCalculator = index_agent
        self.portfolio: Portfolio = portfolio
        self.data_provider: DataFetcher = data_provider
        self.management_fee_bps: int = management_fee_bps  # e.g., 20 for 0.20%

        # Cached outputs of the composed calculator + engine pipeline.
        self._index_result: Optional[IndexResult] = None
        self._backtest_result: Optional[BacktestResult] = None

    # ------------------------------------------------------------------
    # Composed pipeline
    # ------------------------------------------------------------------

    @property
    def index_result(self) -> Optional[IndexResult]:
        """The target :class:`IndexResult` from the most recent run, if any."""
        return self._index_result

    @property
    def backtest_result(self) -> Optional[BacktestResult]:
        """The :class:`BacktestResult` from the most recent run, if any."""
        return self._backtest_result

    def run_backtest(self,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     transaction_cost_bps: float = 0.0) -> BacktestResult:
        """Compute target weights and simulate the tracking portfolio.

        Runs the index calculator to produce the target weight schedule, then
        hands that schedule to a :class:`BacktestEngine` which manages its own
        portfolio. The resulting :class:`BacktestResult` is cached and returned.

        Args:
            start_date: First simulation date (YYYY-MM-DD). Defaults to the
                target index's base date.
            end_date: Last simulation date (YYYY-MM-DD). Required.
            transaction_cost_bps: Trading cost applied by the engine to each
                trade's notional. Distinct from the fund's management fee.

        Returns:
            The BacktestResult produced by the engine.
        """
        if end_date is None:
            raise ValueError("end_date must be provided to run the fund backtest.")

        base_date = self.target_index_definition.base_date
        start = start_date or base_date.strftime('%Y-%m-%d')

        logger.info(
            f"Fund '{self.fund_id}': computing target weights for "
            f"'{self.target_index_definition.index_name}' from {start} to {end_date}."
        )

        # 1. Target weight schedule from the index calculator.
        self._index_result = self.index_agent.run(start_date=start, end_date=end_date)

        # 2. Simulate the tracking portfolio with the backtest engine.
        engine = BacktestEngine(
            start_date=start,
            end_date=end_date,
            initial_capital=self.portfolio.cash_balance,
            data_provider=self.data_provider,
            target_index_result=self._index_result,
            transaction_cost_bps=transaction_cost_bps,
        )
        self._backtest_result = engine.run()

        logger.info(
            f"Fund '{self.fund_id}': backtest complete "
            f"({len(self._backtest_result.portfolio_nav)} days, "
            f"{len(self._backtest_result.transactions)} transactions)."
        )
        return self._backtest_result

    def rebalance_to_index(self,
                           current_date: pd.Timestamp) -> None:
        """Align the fund's tracking portfolio with the target index.

        Thin wrapper that ensures the composed calculator + engine pipeline has
        been run through *current_date*. All weight computation is delegated to
        the :class:`IndexCalculator` and all trading to the
        :class:`BacktestEngine`; this class performs no buy/sell logic itself.

        Args:
            current_date: The date through which to simulate.
        """
        self._ensure_backtest(pd.Timestamp(current_date))

    def _ensure_backtest(self,
                         through_date: pd.Timestamp) -> None:
        """Run (or re-run) the backtest so it covers *through_date*."""
        base_date = self.target_index_definition.base_date
        if through_date < base_date:
            # Nothing to simulate before the index exists.
            return

        nav = self._backtest_result.portfolio_nav if self._backtest_result else None
        if nav is not None and not nav.empty and nav.index[-1] >= through_date:
            return  # Cached result already covers the requested date.

        self.run_backtest(end_date=through_date.strftime('%Y-%m-%d'))

    # ------------------------------------------------------------------
    # NAV
    # ------------------------------------------------------------------

    def calculate_nav(self,
                      current_date: pd.Timestamp) -> float:
        """Return the fund's Net Asset Value as of *current_date*.

        The gross NAV is read from the backtest-engine-managed portfolio; the
        accrued management fee is then deducted.

        Args:
            current_date: The date for which to calculate NAV.

        Returns:
            The fee-adjusted Net Asset Value.
        """
        ts = pd.Timestamp(current_date)
        self._ensure_backtest(ts)

        if self._backtest_result is None or self._backtest_result.portfolio_nav.empty:
            # Date precedes the simulation window — only seed capital exists.
            return float(self.portfolio.cash_balance)

        nav_series = self._backtest_result.portfolio_nav
        on_or_before = nav_series.index[nav_series.index <= ts]
        if len(on_or_before) == 0:
            return float(self.portfolio.cash_balance)

        as_of = on_or_before[-1]
        gross_nav = float(nav_series.loc[as_of])
        elapsed_days = nav_series.index.get_loc(as_of)  # 0 on the first day

        net_nav = self._apply_management_fee(gross_nav, elapsed_days)
        logger.debug(
            f"NAV for fund '{self.fund_id}' on {ts.strftime('%Y-%m-%d')}: "
            f"gross={gross_nav:.2f}, net={net_nav:.2f}"
        )
        return net_nav

    def _apply_management_fee(self,
                              gross_nav: float,
                              elapsed_days: int) -> float:
        """Deduct the accrued management fee from *gross_nav*.

        The annual fee is accrued daily (ACT/252) and compounded over the number
        of elapsed days since the start of the simulation.
        """
        if self.management_fee_bps <= 0 or elapsed_days <= 0:
            return gross_nav
        daily_fee_rate = (self.management_fee_bps / 10000.0) / 252.0
        fee_factor = (1.0 - daily_fee_rate) ** elapsed_days
        return gross_nav * fee_factor

    def __repr__(self) -> str:
        return (f"IndexFund(fund_id='{self.fund_id}', "
                f"target_index='{self.target_index_definition.index_name}', "
                f"management_fee_bps={self.management_fee_bps})")
