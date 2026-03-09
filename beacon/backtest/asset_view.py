# beacon/backtest/asset_view.py
"""
BacktestAssetView — asset view extended with backtest-specific context
such as actual weight history, holding periods, and transaction analysis.
"""
import pandas as pd
from typing import Dict, List, Optional, TYPE_CHECKING

from ..asset.view import AssetView

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher
    from ..portfolio.base import Transaction


class BacktestAssetView(AssetView):
    """AssetView with backtest weight history for a specific asset.

    Parameters
    ----------
    asset_id : str
        The identifier used to look up data in the DataFetcher.
    data_fetcher : DataFetcher
        The data provider instance.
    actual_weight_history : pd.DataFrame
        DataFrame indexed by date with ``{asset_id}_weight`` columns.
    portfolio_nav : pd.Series
        Portfolio NAV time series from the parent BacktestResult.
    transactions : list
        All transactions from the backtest.
    target_weight_snapshots : dict, optional
        Mapping of rebalance date -> dict of {asset_id: weight} from
        the target IndexResult.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: 'DataFetcher',
                 actual_weight_history: pd.DataFrame,
                 portfolio_nav: pd.Series,
                 transactions: List['Transaction'],
                 target_weight_snapshots: Optional[Dict[pd.Timestamp, Dict[str, float]]] = None):
        super().__init__(asset_id, data_fetcher)
        self._actual_weight_history = actual_weight_history
        self._portfolio_nav = portfolio_nav
        self._transactions = transactions
        self._target_weight_snapshots = target_weight_snapshots or {}

    def trades(self) -> pd.DataFrame:
        """Filter transactions for this asset.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, type, quantity, price, cost.
            Empty DataFrame if no trades exist for this asset.
        """
        asset_txns = [t for t in self._transactions if t.asset_id == self._asset_id]
        if not asset_txns:
            return pd.DataFrame(columns=["date", "type", "quantity", "price", "cost"])
        rows = [
            {
                "date": t.transaction_date,
                "type": t.transaction_type,
                "quantity": t.quantity,
                "price": t.price,
                "cost": t.transaction_cost,
            }
            for t in asset_txns
        ]
        return pd.DataFrame(rows)

    def holding_periods(self) -> List[Dict[str, pd.Timestamp]]:
        """Identify continuous periods when this asset was held.

        Returns
        -------
        list of dict
            Each dict has ``"start"`` and ``"end"`` keys with Timestamps.
            An open position at the end of the backtest will have ``"end"``
            set to the last date in the weight history.
        """
        col = f"{self._asset_id}_weight"
        if col not in self._actual_weight_history.columns:
            return []

        weights = self._actual_weight_history[col].fillna(0)
        held = weights > 0
        periods = []
        in_period = False
        start = None

        for date, is_held in held.items():
            if is_held and not in_period:
                start = date
                in_period = True
            elif not is_held and in_period:
                periods.append({"start": start, "end": prev_date})
                in_period = False
            prev_date = date

        if in_period:
            periods.append({"start": start, "end": prev_date})

        return periods

    def actual_weight_series(self) -> pd.Series:
        """Return time series of this asset's actual portfolio weight.

        Alias for :meth:`weight_series` with a clearer name for
        backtest context.

        Returns
        -------
        pd.Series
            Weight at each date where the asset was held.
        """
        return self.weight_series()

    def weight_series(self) -> pd.Series:
        """Return time series of this asset's portfolio weight.

        Returns
        -------
        pd.Series
            Weight at each date where the asset was held. Dates where the
            asset had zero or no weight are excluded.
        """
        col = f"{self._asset_id}_weight"
        if col not in self._actual_weight_history.columns:
            return pd.Series(dtype=float)
        series = self._actual_weight_history[col].dropna()
        return series[series > 0]

    def target_weight_series(self) -> pd.Series:
        """Return time series of this asset's target index weight.

        Returns
        -------
        pd.Series
            Target weight at each rebalance date. Rebalance dates
            where the asset was not a constituent are excluded.
            Empty Series if no target data is available.
        """
        if not self._target_weight_snapshots:
            return pd.Series(dtype=float)
        data = {}
        for rebal_date in sorted(self._target_weight_snapshots):
            weights = self._target_weight_snapshots[rebal_date]
            if self._asset_id in weights:
                data[rebal_date] = weights[self._asset_id]
        return pd.Series(data, dtype=float)

    def slippage_vs_target(self) -> pd.Series:
        """Calculate difference between actual and target weights over time.

        For each date in the actual weight history, finds the applicable
        target weight (most recent rebalance on or before that date) and
        computes actual - target.

        Returns
        -------
        pd.Series
            Slippage series indexed by date. Positive values mean the
            asset is overweight vs target. Empty Series if no target
            data is available.
        """
        if not self._target_weight_snapshots:
            return pd.Series(dtype=float)

        actual = self.weight_series()
        if actual.empty:
            return pd.Series(dtype=float)

        sorted_rebal_dates = sorted(self._target_weight_snapshots.keys())

        def _target_on_date(date):
            applicable = [d for d in sorted_rebal_dates if d <= date]
            if not applicable:
                return 0.0
            latest = applicable[-1]
            return self._target_weight_snapshots[latest].get(self._asset_id, 0.0)

        target = actual.index.to_series().apply(_target_on_date)
        target.index = actual.index
        return actual - target

    def total_cost(self) -> float:
        """Sum of all transaction costs for this asset.

        Returns
        -------
        float
            Total transaction costs incurred for this asset.
        """
        return sum(t.transaction_cost for t in self._transactions if t.asset_id == self._asset_id)

    def weight_on_date(self, date: pd.Timestamp) -> Optional[float]:
        """Get this asset's portfolio weight on a specific date.

        Parameters
        ----------
        date : pd.Timestamp
            The query date.

        Returns
        -------
        float or None
            The weight, or None if the asset was not held on that date.
        """
        col = f"{self._asset_id}_weight"
        if col not in self._actual_weight_history.columns:
            return None
        if date not in self._actual_weight_history.index:
            applicable = self._actual_weight_history.index[
                self._actual_weight_history.index <= date
            ]
            if applicable.empty:
                return None
            date = applicable[-1]
        val = self._actual_weight_history.loc[date, col]
        if pd.isna(val) or val == 0:
            return None
        return float(val)

    def __repr__(self) -> str:
        return f"BacktestAssetView(asset_id='{self._asset_id}')"
