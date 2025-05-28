# beacon/asset/equity.py
"""
Module defining the Equity asset class.
"""
import pandas as pd
from typing import List, Dict, Optional, Any
from .base import Asset
# Forward declaration for DataFetcher to avoid circular import issues at load time.
# from ..data.data_fetcher import DataFetcher


class Equity(Asset):
    """
    Represents an equity security.
    """
    def __init__(self,
                 ticker: str,
                 name: str,
                 currency: str,
                 exchange: str,
                 isin: Optional[str] = None,
                 asset_id: Optional[str] = None):
        """
        Initializes an Equity asset.

        Args:
            ticker: The stock ticker symbol.
            name: The name of the company.
            currency: The currency in which the equity is traded.
            exchange: The exchange where the equity is listed.
            isin: The ISIN code of the equity (optional).
            asset_id: A unique identifier for the asset. If None, defaults to ticker.
        """
        super().__init__(asset_id=asset_id if asset_id else ticker, asset_type="EQUITY")
        if not ticker:
            raise ValueError("ticker cannot be empty.")
        if not name:
            raise ValueError("name cannot be empty.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if not exchange:
            raise ValueError("exchange cannot be empty.")

        self.ticker: str = ticker
        self.name: str = name
        self.currency: str = currency
        self.exchange: str = exchange
        self.isin: Optional[str] = isin

    def get_historical_data(self, start_date: str, end_date: str, data_source: 'DataFetcher') -> pd.DataFrame:
        """
        Fetches historical price data for the equity.
        Delegates the actual fetching to a data_source object.

        Args:
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).
            data_source: An object conforming to the DataFetcher interface,
                         responsible for retrieving data.

        Returns:
            A pandas DataFrame with historical price data (Open, High, Low, Close, Volume, Adj Close).
        """
        # Example of how it might delegate. The actual call depends on DataFetcher's interface.
        # from ..data.data_fetcher import DataFetcher # Local import if necessary
        if not hasattr(data_source, 'fetch_prices'):
            raise AttributeError("Provided data_source object does not have a 'fetch_prices' method.")
        return data_source.fetch_prices(ticker=self.ticker, start_date=start_date, end_date=end_date)

    def get_corporate_actions(self, start_date: str, end_date: str, data_source: 'DataFetcher') -> List[Dict[str, Any]]:
        """
        Fetches corporate actions for the equity.
        Delegates the actual fetching to a data_source object.

        Args:
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).
            data_source: An object conforming to the DataFetcher interface.

        Returns:
            A list of dictionaries, where each dictionary represents a corporate action
            (e.g., {'type': 'DIVIDEND', 'date': 'YYYY-MM-DD', 'value': 0.5}).
        """
        # from ..data.data_fetcher import DataFetcher # Local import if necessary
        if not hasattr(data_source, 'fetch_corporate_actions'):
            raise AttributeError("Provided data_source object does not have a 'fetch_corporate_actions' method.")
        return data_source.fetch_corporate_actions(ticker=self.ticker, start_date=start_date, end_date=end_date)

    def __repr__(self) -> str:
        return (f"Equity(ticker='{self.ticker}', name='{self.name}', currency='{self.currency}', "
                f"exchange='{self.exchange}', isin='{self.isin}', asset_id='{self.asset_id}')")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Equity):
            return NotImplemented
        return super().__eq__(other) and self.ticker == other.ticker

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.ticker))