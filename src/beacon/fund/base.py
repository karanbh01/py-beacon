# src/beacon/fund/base.py
"""
IndexFund: a fund that tracks a target index by running a backtest of it.
"""
import logging
from typing import Any, Literal

import pandas as pd

from ..assumptions import ModellingAssumptions
from ..backtest.dividends import REINVEST
from ..backtest.implementation import Implementation
from ..backtest.main import Backtest
from ..backtest.result import BacktestResult
from ..backtest.rules import BacktestModifier
from ..data.fetcher import DataFetcher
from ..index.cache import IndexResultCache
from ..index.calculation import IndexCalculator
from ..index.constructor import IndexDefinition
from ..index.result import IndexResult
from ..portfolio.base import Portfolio

logger = logging.getLogger(__name__)


# Delegation to `Backtest` is BN-161.
class IndexFund:
    """An index fund that tracks a target index.

    The fund delegates the whole pipeline (target weight calculation and the
    simulated tracking portfolio) to :class:`~beacon.backtest.main.Backtest`,
    the front door composing :class:`~beacon.index.calculation.IndexCalculator`
    and :class:`~beacon.backtest.engine.BacktestEngine`. It contains no
    buy/sell logic of its own: rebalancing and portfolio accounting are
    delegated entirely to the backtest engine.
    """

    def __init__(self,
                 fund_id: str,
                 target_index_definition: IndexDefinition,
                 index_agent: IndexCalculator,
                 portfolio: Portfolio,
                 data_provider: DataFetcher,
                 management_fee_bps: int = 0,
                 *,
                 currency: str | None = None,
                 modelling_assumptions: ModellingAssumptions | None = None,
                 dividends: str = REINVEST,
                 implementation: Implementation | None = None,
                 modifiers: list[BacktestModifier] | None = None,
                 benchmark: IndexResult | pd.Series | None = None,
                 cache: IndexResultCache | Literal[False] | None = None):
        """
        Initializes an IndexFund.

        Args:
            fund_id: A unique identifier for the fund.
            target_index_definition: The definition of the index the fund aims to track.
            index_agent: An IndexCalculator for the target index. Only its
                         ``price_column`` is read: the backtest calculates
                         the index itself from *target_index_definition*.
            portfolio: The Portfolio object seeding the fund's capital. Its cash
                       balance is used as the backtest engine's initial capital;
                       the fund never mutates this portfolio.
            data_provider: DataFetcher instance for market data.
            management_fee_bps: The annual management fee in basis points (e.g., 10 bps = 0.1%).

        The keyword-only settings are passed to the
        :class:`~beacon.backtest.main.Backtest` the fund runs, and each takes
        that class's default when unset:

        Keyword Args:
            currency: The book's currency. None keeps it in the index's.
            modelling_assumptions: What the run takes as given about markets
                and data, laid over the process-wide default.
            dividends: What happens to cash distributions: ``"reinvest"``
                (the default), ``"cash"`` or ``"distribute"``.
            implementation: How the index is carried out at the fund's size:
                screens, caps, market impact and execution limits.
            modifiers: Hooks that can skip rebalances or adjust trades.
            benchmark: The benchmark of record, stored on every result.
            cache: Where calculated indices are kept between runs. None uses
                the default location, and False turns caching off.

        Raises:
            ValueError: If any argument is missing or empty, or
                *management_fee_bps* is negative.
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

        # Passed to every Backtest the fund builds (BN-262): a fund used to
        # run with the defaults whatever the caller needed.
        self._backtest_settings: dict[str, Any] = {
            "currency": currency,
            "modelling_assumptions": modelling_assumptions,
            "dividends": dividends,
            "implementation": implementation,
            "modifiers": modifiers,
            "benchmark": benchmark,
            "cache": cache,
        }

        # Cached outputs of the composed calculator + engine pipeline.
        self._index_result: IndexResult | None = None
        self._backtest_result: BacktestResult | None = None

        # The settings of the last run, which a run extended to a later date
        # keeps (BN-247): it used to re-run at zero cost from the base date.
        self._start_date: str | None = None
        self._transaction_cost_bps: float = 0.0

    # ------------------------------------------------------------------
    # Composed pipeline
    # ------------------------------------------------------------------

    @property
    def index_result(self) -> IndexResult | None:
        """The target :class:`IndexResult` from the most recent run, if any."""
        return self._index_result

    @property
    def backtest_result(self) -> BacktestResult | None:
        """The :class:`BacktestResult` from the most recent run, if any."""
        return self._backtest_result

    def run_backtest(self,
                     start_date: str | None = None,
                     end_date: str | None = None,
                     transaction_cost_bps: float = 0.0) -> BacktestResult:
        """Compute target weights and simulate the tracking portfolio.

        Delegates the whole calculate-then-simulate composition to
        :class:`~beacon.backtest.main.Backtest`, which fingerprints the
        calculation, reuses a cached IndexResult when the data source allows
        it, and hands the schedule to a backtest engine that manages its own
        portfolio. The run's initial capital is the seed portfolio's cash
        balance. The resulting :class:`BacktestResult` is cached on the fund
        and returned.

        Args:
            start_date: First simulation date (YYYY-MM-DD). Defaults to the
                target index's base date.
            end_date: Last simulation date (YYYY-MM-DD). Required.
            transaction_cost_bps: Trading cost applied by the engine to each
                trade's notional. Distinct from the fund's management fee.

        Returns:
            The BacktestResult produced by the engine.

        Raises:
            ValueError: If *end_date* is not provided.
        """
        if end_date is None:
            raise ValueError("end_date must be provided to run the fund backtest.")

        base_date = self.target_index_definition.base_date
        start = start_date or base_date.strftime('%Y-%m-%d')

        logger.info(
            f"Fund '{self.fund_id}': computing target weights for "
            f"'{self.target_index_definition.index_name}' from {start} to {end_date}."
        )

        self._start_date = start_date
        self._transaction_cost_bps = transaction_cost_bps

        backtest = Backtest(initial_capital=self.portfolio.cash_balance,
                            transaction_cost_bps=transaction_cost_bps,
                            price_column=self.index_agent.price_column,
                            data_provider=self.data_provider,
                            **self._backtest_settings)
        self._backtest_result = backtest.run(self.target_index_definition,
                                             start=start,
                                             end=end_date)

        # The calculation the run tracked, kept for the fund's own accessor.
        book = self._backtest_result.index.target
        self._index_result = book.source if book is not None else None

        logger.info(
            f"Fund '{self.fund_id}': backtest complete "
            f"({len(self._backtest_result.trading_nav)} days, "
            f"{len(self._backtest_result.portfolio.transactions)} transactions)."
        )
        return self._backtest_result

    def rebalance_to_index(self,
                           current_date: pd.Timestamp) -> None:
        """Align the fund's tracking portfolio with the target index.

        Thin wrapper that ensures the composed calculator + engine pipeline has
        been run through *current_date*: if the cached run does not reach that
        date, the backtest is run again to it, with the start date and
        transaction cost of the last run (or from the base date at no cost if
        there has been none). Nothing is run for a date before the base date.
        All weight computation is delegated to the :class:`IndexCalculator`
        and all trading to the :class:`BacktestEngine`; this class performs no
        buy/sell logic itself.

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

        nav = self._backtest_result.trading_nav if self._backtest_result else None
        if nav is not None and not nav.empty and nav.index[-1] >= through_date:
            return  # Cached result already covers the requested date.

        self.run_backtest(start_date=self._start_date,
                          end_date=through_date.strftime('%Y-%m-%d'),
                          transaction_cost_bps=self._transaction_cost_bps)

    # ------------------------------------------------------------------
    # NAV
    # ------------------------------------------------------------------

    def calculate_nav(self,
                      current_date: pd.Timestamp) -> float:
        """Return the fund's Net Asset Value as of *current_date*.

        Runs (or re-runs) the backtest if it does not yet cover
        *current_date*. The gross NAV is the backtest's trading NAV on the
        last simulated day on or before *current_date*. The management fee is
        then deducted: the annual fee divided by 252 is charged per elapsed
        NAV-series day, compounded, so the first simulated day carries no fee.
        Before the simulation window the seed portfolio's cash balance is
        returned.

        Args:
            current_date: The date for which to calculate NAV.

        Returns:
            The fee-adjusted Net Asset Value.
        """
        ts = pd.Timestamp(current_date)
        self._ensure_backtest(ts)

        if self._backtest_result is None or self._backtest_result.trading_nav.empty:
            # Date precedes the simulation window: only seed capital exists.
            return float(self.portfolio.cash_balance)

        # The trading NAV, deliberately: the fee accrues over elapsed
        # NAV-series days, and the day-zero row (decision 11) is a starting
        # fact, not an elapsed day. Reading it here would shift every day
        # count by one and change accrued fees; excluding it keeps them
        # bit-identical to the pre-redesign series -- the fund tests are the
        # proof.
        nav_series = self._backtest_result.trading_nav
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
