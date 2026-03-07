# beacon/index/asset_view.py
"""
IndexAssetView — asset view extended with index-specific context
such as weight history and contribution analysis.
"""
import pandas as pd
from typing import Dict, Optional, TYPE_CHECKING

from ..asset.view import AssetView

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher


class IndexAssetView(AssetView):
    """AssetView with index weight history and contribution analysis.

    Parameters
    ----------
    asset_id : str
        The identifier used to look up data in the DataFetcher.
    data_fetcher : DataFetcher
        The data provider instance.
    weight_snapshots : dict
        Mapping of rebalance date -> dict of {asset_id: weight} from
        the parent IndexResult.
    index_levels : pd.Series
        Index level time series from the parent IndexResult.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: 'DataFetcher',
                 weight_snapshots: Dict[pd.Timestamp, Dict[str, float]],
                 index_levels: pd.Series):
        super().__init__(asset_id, data_fetcher)
        self._weight_snapshots = weight_snapshots
        self._index_levels = index_levels

    def weight_on_date(self, date: pd.Timestamp) -> Optional[float]:
        """Get this asset's index weight on a specific date.

        Finds the most recent rebalance on or before *date* and returns
        the asset's weight. Returns ``None`` if the asset was not a
        constituent at that point.

        Parameters
        ----------
        date : pd.Timestamp
            The query date.

        Returns
        -------
        float or None
        """
        applicable = [d for d in self._weight_snapshots if d <= date]
        if not applicable:
            return None
        latest = max(applicable)
        return self._weight_snapshots[latest].get(self._asset_id)

    def weight_series(self) -> pd.Series:
        """Return a Series of this asset's weight at each rebalance date.

        Returns
        -------
        pd.Series
            Indexed by rebalance date. Rebalance dates where the asset
            was not a constituent are excluded.
        """
        data = {}
        for rebal_date in sorted(self._weight_snapshots):
            weights = self._weight_snapshots[rebal_date]
            if self._asset_id in weights:
                data[rebal_date] = weights[self._asset_id]
        return pd.Series(data, dtype=float)

    def contribution(self,
                     start: str,
                     end: str,
                     price_column: str = "CLOSE") -> pd.Series:
        """Calculate this asset's contribution to index returns.

        Contribution on day *t* = weight_{t-1} * asset_return_t.

        Parameters
        ----------
        start : str
            Start date (YYYY-MM-DD).
        end : str
            End date (YYYY-MM-DD).
        price_column : str
            Column name for return calculation.

        Returns
        -------
        pd.Series
            Contribution series indexed by date.
        """
        asset_returns = self.returns(start, end, price_column=price_column)
        if asset_returns.empty:
            return pd.Series(dtype=float)

        # Build a weight series aligned to the return dates
        # For each return date, look up the weight from the most recent rebalance
        weights = asset_returns.index.to_series().apply(
            lambda d: self.weight_on_date(d)
        ).shift(1)  # weight_{t-1}

        contribution = weights * asset_returns
        return contribution.dropna()

    def __repr__(self) -> str:
        return f"IndexAssetView(asset_id='{self._asset_id}')"
