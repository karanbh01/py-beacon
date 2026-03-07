# beacon/index/result.py
"""
IndexResult — output container for index calculation results.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from ..data.fetcher import DataFetcher
    from .asset_view import IndexAssetView


@dataclass
class IndexResult:
    """Container holding the output of an index calculation run.

    Parameters
    ----------
    index_id : str
        Identifier of the calculated index.
    index_levels : pd.Series
        Time series of index levels indexed by ``pd.DatetimeIndex``.
    divisor_history : pd.Series
        Time series of divisor values indexed by ``pd.DatetimeIndex``.
    constituent_snapshots : dict
        Mapping of rebalance date -> list of asset_id strings.
    weight_snapshots : dict
        Mapping of rebalance date -> dict of {asset_id: weight}.
    """
    index_id: str
    index_levels: pd.Series
    divisor_history: pd.Series
    constituent_snapshots: Dict[pd.Timestamp, List[str]]
    weight_snapshots: Dict[pd.Timestamp, Dict[str, float]]
    _data_fetcher: Optional['DataFetcher'] = field(default=None, repr=False, compare=False)

    def with_data(self, data_fetcher: 'DataFetcher') -> 'IndexResult':
        """Bind a DataFetcher for asset-level queries. Returns self for chaining."""
        self._data_fetcher = data_fetcher
        return self

    def asset(self, asset_id: str) -> 'IndexAssetView':
        """Return an IndexAssetView for a constituent.

        Parameters
        ----------
        asset_id : str
            Identifier of the constituent asset.

        Returns
        -------
        IndexAssetView

        Raises
        ------
        RuntimeError
            If no DataFetcher has been bound via :meth:`with_data`.
        KeyError
            If *asset_id* is not found in any constituent snapshot.
        """
        if self._data_fetcher is None:
            raise RuntimeError(
                "No DataFetcher bound. Call .with_data(fetcher) first."
            )

        all_constituents = set()
        for ids in self.constituent_snapshots.values():
            all_constituents.update(ids)

        if asset_id not in all_constituents:
            raise KeyError(
                f"Asset '{asset_id}' not found in any constituent snapshot."
            )

        from .asset_view import IndexAssetView
        return IndexAssetView(
            asset_id=asset_id,
            data_fetcher=self._data_fetcher,
            weight_snapshots=self.weight_snapshots,
            index_levels=self.index_levels,
        )

    def get_returns(self) -> pd.Series:
        """Derive a return series from index levels.

        Returns
        -------
        pd.Series
            Percentage returns (first entry is dropped).
        """
        if self.index_levels.empty:
            return pd.Series(dtype=float)
        return self.index_levels.pct_change().dropna()

    def get_weights_on_date(self, date: pd.Timestamp) -> Dict[str, float]:
        """Get constituent weights effective on a given date.

        Locates the most recent rebalance date on or before *date*.

        Parameters
        ----------
        date : pd.Timestamp
            The query date.

        Returns
        -------
        dict
            Mapping of asset_id to weight. Empty dict if no rebalance
            has occurred on or before *date*.
        """
        applicable_dates = [d for d in self.weight_snapshots if d <= date]
        if not applicable_dates:
            return {}
        latest = max(applicable_dates)
        return self.weight_snapshots[latest]

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten index levels and divisor history into a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ``index_level``, ``divisor``.
        """
        df = pd.DataFrame({
            "index_level": self.index_levels,
            "divisor": self.divisor_history,
        })
        df.index.name = "date"
        return df

    def __repr__(self) -> str:
        n_dates = len(self.index_levels)
        n_rebalances = len(self.weight_snapshots)
        bound = self._data_fetcher is not None
        return (
            f"IndexResult(index_id='{self.index_id}', "
            f"dates={n_dates}, rebalances={n_rebalances}, "
            f"data_bound={bound})"
        )
