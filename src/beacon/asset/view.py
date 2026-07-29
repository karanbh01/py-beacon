# src/beacon/asset/view.py
"""
AssetView — convenience wrapper combining an asset identifier with a DataFetcher
for streamlined data retrieval.
"""

import pandas as pd

from ..data.fetcher import DataFetcher


class AssetView:
    """Queryable wrapper that pairs an asset identifier with a DataFetcher.

    Args:
        asset_id: The identifier used to look up data in the DataFetcher.
        data_fetcher: The data provider instance.
    """

    def __init__(self,
                 asset_id: str,
                 data_fetcher: DataFetcher):
        if not asset_id:
            raise ValueError("asset_id cannot be empty.")
        if data_fetcher is None:
            raise ValueError("data_fetcher must be provided.")

        self._asset_id = asset_id
        self._data_fetcher = data_fetcher

    @property
    def asset_id(self) -> str:
        return self._asset_id

    def prices(self,
               start: str,
               end: str) -> pd.DataFrame:
        """Retrieve historical OHLCV price data.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Price data indexed by date.
        """
        return self._data_fetcher.fetch_market_data(self._asset_id, start, end)

    def returns(self,
                start: str,
                end: str,
                frequency: str = "daily",
                price_column: str = "CLOSE") -> pd.Series:
        """Calculate a return series from price data.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            frequency: One of ``"daily"``, ``"weekly"``, or ``"monthly"``.
            price_column: Column name to use for return calculation. Defaults
                to ``"CLOSE"``.

        Returns:
            pd.Series: Percentage returns indexed by date.

        Raises:
            ValueError: If *frequency* is not one of the supported values.
        """
        supported = ("daily", "weekly", "monthly")
        if frequency not in supported:
            raise ValueError(
                f"Unsupported frequency '{frequency}'. Must be one of {supported}."
            )

        prices = self._data_fetcher.fetch_market_data(
            self._asset_id, start, end, columns=[price_column]
        )

        if prices.empty:
            return pd.Series(dtype=float)

        series = prices[price_column]

        resample_map = {"daily": None, "weekly": "W", "monthly": "ME"}
        rule = resample_map[frequency]
        if rule is not None:
            series = series.resample(rule).last().dropna()

        return series.pct_change().dropna()

    def reference_data(self,
                       date: str | None = None) -> pd.DataFrame:
        """Fetch static reference data (e.g. name, sector, exchange).

        Args:
            date: Point-in-time date for the reference snapshot.

        Returns:
            pd.DataFrame: Reference data for this asset.
        """
        return self._data_fetcher.fetch_reference_data(self._asset_id, date)

    def corporate_actions(self,
                          start: str,
                          end: str) -> pd.DataFrame:
        """Retrieve corporate action events (dividends, splits, etc.).

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            pd.DataFrame: Corporate actions within the date range.
        """
        return self._data_fetcher.fetch_market_data(
            self._asset_id, start, end, columns=["DIVIDEND", "SPLIT"]
        )

    def __repr__(self) -> str:
        return f"AssetView(asset_id='{self._asset_id}')"
