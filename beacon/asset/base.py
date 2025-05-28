# beacon/asset/base.py
"""
Module defining the base class for financial assets.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any
# Forward declaration for DataFetcher to avoid circular import if needed
# from ..data.data_fetcher import DataFetcher

class Asset(ABC):
    """
    Abstract base class for a financial asset.
    """
    def __init__(self, asset_id: str, asset_type: str):
        """
        Initializes the Asset.

        Args:
            asset_id: A unique identifier for the asset.
            asset_type: The type of the asset (e.g., 'EQUITY', 'BOND').
        """
        if not asset_id:
            raise ValueError("asset_id cannot be empty.")
        if not asset_type:
            raise ValueError("asset_type cannot be empty.")
        self._asset_id = asset_id
        self._asset_type = asset_type

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def asset_type(self) -> str:
        return self._asset_type

    @abstractmethod
    def get_historical_data(self, start_date: str, end_date: str, data_source: Any) -> pd.DataFrame: # 'Any' for DataFetcher initially
        """
        Fetches historical price/return data for the asset.

        Args:
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).
            data_source: The data provider object (e.g., an instance of a DataFetcher).

        Returns:
            A pandas DataFrame with historical data.
        """
        pass

    @abstractmethod
    def get_corporate_actions(self, start_date: str, end_date: str, data_source: Any) -> List[Dict[str, Any]]: # 'Any' for DataFetcher initially
        """
        Fetches corporate actions for the asset.

        Args:
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).
            data_source: The data provider object.

        Returns:
            A list of dictionaries, each representing a corporate action.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(asset_id='{self.asset_id}', asset_type='{self.asset_type}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Asset):
            return NotImplemented
        return self.asset_id == other.asset_id and self.asset_type == other.asset_type

    def __hash__(self) -> int:
        return hash((self.asset_id, self.asset_type))