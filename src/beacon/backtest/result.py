# src/beacon/backtest/result.py
"""
BacktestResult — output container for backtest runs.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from ..portfolio.base import Transaction
from .asset_view import BacktestAssetView


@dataclass(frozen=True)
class UnfilledOrder:
    """A buy the simulation could not execute in full.

    Recorded on the result rather than only logged: a partially filled
    rebalance leaves the portfolio off its target weights, and a caller
    comparing tracking error against expectations needs to know that happened
    rather than reading it as a modelling result.

    Attributes:
        date: The rebalance date.
        asset_id: Asset that could not be fully bought.
        requested_quantity: Quantity the rebalance asked for.
        filled_quantity: Quantity actually bought; 0.0 when nothing was.
        price: Execution price used.
        shortfall_value: Notional value that went unfilled, at *price*.
    """
    date: pd.Timestamp
    asset_id: str
    requested_quantity: float
    filled_quantity: float
    price: float
    shortfall_value: float


@dataclass
class BacktestResult:
    """Container holding the output of a backtest run.

    Args:
        portfolio_id: Identifier of the backtested portfolio.
        initial_capital: Starting capital for the backtest.
        portfolio_nav: Time series of portfolio NAV indexed by
            ``pd.DatetimeIndex``.
        cash_history: Time series of cash balances indexed by
            ``pd.DatetimeIndex``.
        transactions: List of Transaction objects executed during the
            backtest.
        actual_weight_history: DataFrame indexed by date with
            ``{asset_id}_weight`` columns.
        target_index_result: The IndexResult of the target index, if
            available.
        unfilled: Buys the simulation could not execute in full. Empty for a
            run where every rebalance leg filled, so a non-empty list is
            itself the signal that the portfolio drifted off target for a
            reason other than price movement.
    """
    portfolio_id: str
    initial_capital: float
    portfolio_nav: pd.Series
    cash_history: pd.Series
    transactions: list[Transaction]
    actual_weight_history: pd.DataFrame
    target_index_result: IndexResult | None = None
    unfilled: list[UnfilledOrder] = field(default_factory=list)
    _data_fetcher: DataFetcher | None = field(default=None, repr=False, compare=False)

    @property
    def total_unfilled_value(self) -> float:
        """Total notional that went unfilled across the run."""
        return sum(order.shortfall_value for order in self.unfilled)

    def with_data(self,
                  data_fetcher: DataFetcher) -> 'BacktestResult':
        """Bind a DataFetcher for asset-level queries. Returns self for chaining."""
        self._data_fetcher = data_fetcher
        return self

    def asset(self,
              asset_id: str) -> BacktestAssetView:
        """Return a BacktestAssetView for a held asset.

        Args:
            asset_id: Identifier of the asset.

        Returns:
            BacktestAssetView

        Raises:
            RuntimeError: If no DataFetcher has been bound via
                :meth:`with_data`.
            KeyError: If *asset_id* is not found in the weight history.
        """
        if self._data_fetcher is None:
            raise RuntimeError(
                "No DataFetcher bound. Call .with_data(fetcher) first."
            )

        col = f"{asset_id}_weight"
        if col not in self.actual_weight_history.columns:
            raise KeyError(
                f"Asset '{asset_id}' not found in backtest weight history."
            )

        target_snapshots = None
        if self.target_index_result is not None:
            target_snapshots = self.target_index_result.weight_snapshots

        return BacktestAssetView(
            asset_id=asset_id,
            data_fetcher=self._data_fetcher,
            actual_weight_history=self.actual_weight_history,
            portfolio_nav=self.portfolio_nav,
            transactions=self.transactions,
            target_weight_snapshots=target_snapshots,
        )

    def get_returns(self) -> pd.Series:
        """Derive a return series from portfolio NAV.

        Returns:
            pd.Series: Percentage returns (first entry is dropped).
        """
        if self.portfolio_nav.empty:
            return pd.Series(dtype=float)
        return self.portfolio_nav.pct_change().dropna()

    def get_tracking_error(self) -> float | None:
        """Calculate annualised tracking error against the target index.

        Tracking error is the annualised standard deviation of the
        difference between portfolio returns and index returns.

        Returns:
            float or None: Annualised tracking error, or None if no target
            index is available.
        """
        if self.target_index_result is None:
            return None

        port_returns = self.get_returns()
        index_returns = self.target_index_result.get_returns()

        # Align on common dates
        aligned = pd.DataFrame({
            "port": port_returns,
            "index": index_returns,
        }).dropna()

        if aligned.empty:
            return None

        active_returns = aligned["port"] - aligned["index"]
        return float(active_returns.std() * np.sqrt(252))

    def get_tracking_difference(self) -> float | None:
        """Calculate cumulative tracking difference against the target index.

        Tracking difference is the difference between the cumulative
        portfolio return and the cumulative index return over the
        full backtest period.

        Returns:
            float or None: Tracking difference, or None if no target index
            is available.
        """
        if self.target_index_result is None:
            return None

        port_returns = self.get_returns()
        index_returns = self.target_index_result.get_returns()

        if port_returns.empty or index_returns.empty:
            return None

        port_cumulative = (1 + port_returns).prod() - 1
        index_cumulative = (1 + index_returns).prod() - 1
        return float(port_cumulative - index_cumulative)

    def summary(self) -> dict[str, float | None]:
        """Calculate key performance metrics for the backtest.

        Returns:
            dict: Dictionary containing: total_return, annualised_return,
            volatility, sharpe_ratio, max_drawdown, and optionally
            tracking_error and tracking_difference.
        """
        returns = self.get_returns()
        n_periods = len(returns)

        # Total return
        if self.portfolio_nav.empty or self.initial_capital == 0:
            total_return = 0.0
        else:
            total_return = float(self.portfolio_nav.iloc[-1] / self.initial_capital - 1)

        # Annualised return
        if n_periods > 0:
            years = n_periods / 252.0
            annualised_return = float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0
        else:
            annualised_return = 0.0

        # Volatility (annualised)
        volatility = float(returns.std() * np.sqrt(252)) if n_periods > 1 else 0.0

        # Sharpe ratio (assumes risk-free rate = 0)
        sharpe_ratio = float(annualised_return / volatility) if volatility > 0 else 0.0

        # Max drawdown
        if not self.portfolio_nav.empty:
            cumulative_max = self.portfolio_nav.cummax()
            drawdown = (self.portfolio_nav - cumulative_max) / cumulative_max
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0

        result: dict[str, float | None] = {
            "total_return": total_return,
            "annualised_return": annualised_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

        # Tracking metrics (only if target index available)
        te = self.get_tracking_error()
        td = self.get_tracking_difference()
        if te is not None:
            result["tracking_error"] = te
        if td is not None:
            result["tracking_difference"] = td

        return result

    def __repr__(self) -> str:
        n_dates = len(self.portfolio_nav)
        n_txns = len(self.transactions)
        bound = self._data_fetcher is not None
        has_target = self.target_index_result is not None
        return (
            f"BacktestResult(portfolio_id='{self.portfolio_id}', "
            f"dates={n_dates}, transactions={n_txns}, "
            f"target_index={has_target}, data_bound={bound})"
        )
