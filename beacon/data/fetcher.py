"""
DataFetcher — unified interface for querying market and reference data.

Accepts single identifiers or lists and passes through column names as-is.
"""

from typing import List, Optional, Union
import pandas as pd
from .base import MarketData, ReferenceData


class DataFetcher:
    """Unified query interface over MarketData and ReferenceData.

    Parameters
    ----------
    market_data : MarketData
        Time-series market data container.
    reference_data : ReferenceData, optional
        Reference data container.
    """

    def __init__(self,
                 market_data: MarketData,
                 reference_data: Optional[ReferenceData] = None):
        self._market = market_data
        self._reference = reference_data

    # -- properties ----------------------------------------------------------

    @property
    def identifiers(self) -> List[str]:
        """Unique identifiers present in market data."""
        return self._market.identifiers

    @property
    def market_columns(self) -> List[str]:
        """Column names in the market data."""
        return self._market.columns

    @property
    def reference_columns(self) -> Optional[List[str]]:
        """Column names in the reference data, or None if not loaded."""
        if self._reference is None:
            return None
        return self._reference.columns

    @property
    def date_range(self):
        """(earliest, latest) timestamps in the market data."""
        return self._market.date_range

    # -- market data ---------------------------------------------------------

    def fetch_market_data(self,
                          identifier: Union[str, List[str]],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch time-series market data for one or more identifiers.

        Parameters
        ----------
        identifier : str or list of str
            One identifier or a list of identifiers.
        start_date, end_date : str, optional
            Date strings to filter the date range.
        columns : list of str, optional
            Subset of columns to return.

        Returns
        -------
        pd.DataFrame
            Single identifier: indexed by ``DATE``.
            Multiple identifiers: MultiIndexed by ``(IDENTIFIER, DATE)``.
            Empty DataFrame if no matching data is found.
        """
        return self._market.get(identifier, start_date, end_date, columns)

    # -- reference data ------------------------------------------------------

    def fetch_reference_data(self,
                             identifier: Union[str, List[str]],
                             date: Optional[str] = None,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fetch reference data for one or more identifiers.

        Parameters
        ----------
        identifier : str or list of str
            One identifier or a list of identifiers.
        date : str, optional
            Point-in-time date. Only rows valid at this date are returned.
        columns : list of str, optional
            Subset of columns to return.

        Returns
        -------
        pd.DataFrame
            Indexed by ``IDENTIFIER``. Empty DataFrame if no reference data
            is loaded or identifier is not found.
        """
        if self._reference is None:
            return pd.DataFrame()

        return self._reference.get(identifier, date, columns)
