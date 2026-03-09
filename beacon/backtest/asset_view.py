# beacon/backtest/asset_view.py
"""
BacktestAssetView — asset view extended with backtest-specific context
such as actual weight history and holding periods.
"""
import pandas as pd
from typing import Dict, Optional, TYPE_CHECKING

from ..asset.view import AssetView

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher


class BacktestAssetView(AssetView):
    """AssetView with backtest weight history for a specific asset.

    Parameters
    ----------
    asset_id : str
        The identifier used to look up data in the DataFetcher.
    data_fetcher : DataFetcher
        The data provider instance.
    actual_weight_history : pd.DataFrame
        DataFrame indexed by date with asset_id columns holding weights.
    portfolio_nav : pd.Series
        Portfolio NAV time series from the parent BacktestResult.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: 'DataFetcher',
                 actual_weight_history: pd.DataFrame,
                 portfolio_nav: pd.Series):
        super().__init__(asset_id, data_fetcher)
        self._actual_weight_history = actual_weight_history
        self._portfolio_nav = portfolio_nav

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
            # Find the most recent date on or before the query date
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
