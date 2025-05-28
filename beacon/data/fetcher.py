# beacon/data/fetcher.py
"""
Module for fetching financial data from various sources.
Includes an abstract base class DataFetcher and concrete implementations
like CSVSecurityDataProvider.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os
import logging
from ..exceptions import DataNotFoundError # Using the package level exception

logger = logging.getLogger(__name__)

class DataFetcher(ABC):
    """
    Abstract base class (interface) for data fetching operations.
    Subclasses will implement methods to fetch data from specific sources
    (e.g., CSV files, APIs).
    """

    @abstractmethod
    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical price data (OHLCV, Adjusted Close) for a given ticker.

        Args:
            ticker: The security ticker symbol.
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).

        Returns:
            A pandas DataFrame with columns like ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'].
            The 'Date' column should be of datetime type. Returns an empty DataFrame if no data.
        """
        pass

    @abstractmethod
    def fetch_corporate_actions(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetches corporate action data for a given ticker.

        Args:
            ticker: The security ticker symbol.
            start_date: The start date for filtering actions (YYYY-MM-DD).
            end_date: The end date for filtering actions (YYYY-MM-DD).

        Returns:
            A list of dictionaries, each representing a corporate action.
            Example: [{'type': 'DIVIDEND', 'date': 'YYYY-MM-DD', 'value': 0.5},
                      {'type': 'SPLIT', 'date': 'YYYY-MM-DD', 'value': '2:1'}]
            Returns an empty list if no actions.
        """
        pass

    @abstractmethod
    def fetch_fx_rates(self, base_currency: str, quote_currency: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetches historical FX rates for a currency pair.

        Args:
            base_currency: The base currency code (e.g., 'USD').
            quote_currency: The quote currency code (e.g., 'EUR').
            start_date: The start date for the data (YYYY-MM-DD).
            end_date: The end date for the data (YYYY-MM-DD).

        Returns:
            A pandas Series indexed by date, with FX rates as values.
            Returns an empty Series if no data.
        """
        pass

    @abstractmethod
    def fetch_shares_outstanding(self, ticker: str, date: str) -> Optional[float]:
        """
        Fetches the number of shares outstanding for a ticker on a specific date.

        Args:
            ticker: The security ticker symbol.
            date: The date for which to fetch shares outstanding (YYYY-MM-DD).

        Returns:
            The number of shares outstanding as a float, or None if not available.
        """
        pass

    @abstractmethod
    def fetch_free_float_factor(self, ticker: str, date: str) -> Optional[float]:
        """
        Fetches the free-float factor for a ticker on a specific date.

        Args:
            ticker: The security ticker symbol.
            date: The date for which to fetch the free-float factor (YYYY-MM-DD).

        Returns:
            The free-float factor (between 0 and 1) as a float, or None if not available.
        """
        pass


class CSVSecurityDataProvider(DataFetcher):
    """
    A DataFetcher implementation that reads security data from local CSV files.
    It expects a certain directory structure and file naming convention.
    """
    def __init__(self, base_path: str,
                 price_path_template: str = "{ticker}_prices.csv",
                 ca_path_template: str = "{ticker}_corp_actions.csv",
                 fx_path_template: str = "{base}_{quote}_fx.csv",
                 shares_path_template: str = "{ticker}_shares_outstanding.csv",
                 freefloat_path_template: str = "{ticker}_free_float.csv"
                 ):
        """
        Initializes the CSVSecurityDataProvider.

        Args:
            base_path: The base directory where data CSV files are stored.
            price_path_template: Template for price CSV filenames, e.g., "{ticker}_prices.csv".
            ca_path_template: Template for corporate action CSV filenames.
            fx_path_template: Template for FX rate CSV filenames.
            shares_path_template: Template for shares outstanding CSV filenames.
            freefloat_path_template: Template for free-float factor CSV filenames.
        """
        self.base_path = base_path
        self.price_path_template = price_path_template
        self.ca_path_template = ca_path_template
        self.fx_path_template = fx_path_template
        self.shares_path_template = shares_path_template
        self.freefloat_path_template = freefloat_path_template

        if not os.path.isdir(base_path):
            logger.error(f"Base data path does not exist: {base_path}")
            raise FileNotFoundError(f"Base data path does not exist: {base_path}")

    def _read_csv_data(self, file_path: str, date_col: Optional[str] = 'Date', expected_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Helper to read and do basic validation on CSV."""
        if not os.path.exists(file_path):
            logger.warning(f"Data file not found: {file_path}")
            return pd.DataFrame()
        try:
            df = pd.read_csv(file_path)
            if date_col and date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col])
            else:
                logger.warning(f"Date column '{date_col}' not found in {file_path}")

            if expected_cols:
                if not all(col in df.columns for col in expected_cols):
                    logger.warning(f"Missing one or more expected columns in {file_path}. Expected: {expected_cols}, Found: {df.columns.tolist()}")
                    # Decide if this should return empty or raise error based on strictness
                    # return pd.DataFrame()
            return df
        except Exception as e:
            logger.error(f"Error reading CSV file {file_path}: {e}")
            return pd.DataFrame()

    def fetch_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches historical price data from a CSV file for a given ticker.
        CSV format should include: Date, Open, High, Low, Close, Volume, Adj Close.
        Standardizes column names to 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'.
        """
        file_name = self.price_path_template.format(ticker=ticker)
        file_path = os.path.join(self.base_path, "prices", file_name) # Assuming a 'prices' subdirectory

        expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] # Case-sensitive
        df = self._read_csv_data(file_path, date_col='Date', expected_cols=None) # Allow flexibility in source columns for now

        if df.empty:
            return pd.DataFrame()

        # Column name standardization (handle potential case variations or common alternatives)
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'date': rename_map[col] = 'Date'
            elif col_lower == 'open': rename_map[col] = 'Open'
            elif col_lower == 'high': rename_map[col] = 'High'
            elif col_lower == 'low': rename_map[col] = 'Low'
            elif col_lower == 'close': rename_map[col] = 'Close'
            elif 'volume' in col_lower : rename_map[col] = 'Volume' # e.g. "Trade Volume"
            elif 'adj' in col_lower and 'close' in col_lower: rename_map[col] = 'Adj Close'
        df.rename(columns=rename_map, inplace=True)

        # Ensure all standard columns exist, fill with NaN if not after rename
        for std_col in expected_cols:
            if std_col not in df.columns:
                if std_col == 'Date': # Date is critical
                    logger.error(f"'Date' column missing in {file_path} after standardization attempt.")
                    return pd.DataFrame()
                df[std_col] = np.nan

        df = df[expected_cols] # Select and order standard columns

        # Filter by date
        pd_start_date = pd.to_datetime(start_date)
        pd_end_date = pd.to_datetime(end_date)
        df_filtered = df[(df['Date'] >= pd_start_date) & (df['Date'] <= pd_end_date)].copy()
        
        # Set Date as index if desired by consumers, but blueprint returns DataFrame
        # df_filtered.set_index('Date', inplace=True)
        return df_filtered


    def fetch_corporate_actions(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetches corporate action data from a CSV file.
        CSV format should include: Date, Type, Value (e.g., for dividend amount or split ratio as str).
        """
        file_name = self.ca_path_template.format(ticker=ticker)
        file_path = os.path.join(self.base_path, "corporate_actions", file_name) # Assuming subdirectory

        df = self._read_csv_data(file_path, date_col='Date', expected_cols=['Date', 'Type', 'Value'])
        if df.empty:
            return []

        pd_start_date = pd.to_datetime(start_date)
        pd_end_date = pd.to_datetime(end_date)
        df_filtered = df[(df['Date'] >= pd_start_date) & (df['Date'] <= pd_end_date)]

        # Convert to list of dicts as per spec
        actions = []
        for _, row in df_filtered.iterrows():
            actions.append({
                'date': row['Date'].strftime('%Y-%m-%d'), # Standardize date string format
                'type': str(row['Type']).upper(), # e.g. DIVIDEND, SPLIT
                'value': row['Value'] # Can be float for dividend, str for split like "2:1"
            })
        return actions

    def fetch_fx_rates(self, base_currency: str, quote_currency: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetches FX rates from a CSV file.
        CSV format: Date, Rate (for Base/Quote).
        """
        file_name = self.fx_path_template.format(base=base_currency.upper(), quote=quote_currency.upper())
        file_path = os.path.join(self.base_path, "fx_rates", file_name) # Assuming subdirectory

        df = self._read_csv_data(file_path, date_col='Date', expected_cols=['Date', 'Rate'])
        if df.empty:
            return pd.Series(dtype=float)

        pd_start_date = pd.to_datetime(start_date)
        pd_end_date = pd.to_datetime(end_date)
        df_filtered = df[(df['Date'] >= pd_start_date) & (df['Date'] <= pd_end_date)].copy()

        if df_filtered.empty:
            return pd.Series(dtype=float)
            
        df_filtered.set_index('Date', inplace=True)
        return df_filtered['Rate'].astype(float)


    def fetch_shares_outstanding(self, ticker: str, date: str) -> Optional[float]:
        """
        Fetches shares outstanding from a CSV.
        CSV format: Date, SharesOutstanding. Assumes data is point-in-time or use last known if exact date not found.
        """
        file_name = self.shares_path_template.format(ticker=ticker)
        file_path = os.path.join(self.base_path, "fundamentals", file_name) # Assuming subdirectory

        df = self._read_csv_data(file_path, date_col='Date', expected_cols=['Date', 'SharesOutstanding'])
        if df.empty:
            return None

        target_date = pd.to_datetime(date)
        # Find the latest record on or before the target date
        df_filtered = df[df['Date'] <= target_date].sort_values(by='Date', ascending=False)

        if df_filtered.empty:
            logger.warning(f"No shares outstanding data found for {ticker} on or before {date}")
            return None
        
        shares = df_filtered['SharesOutstanding'].iloc[0]
        return float(shares) if pd.notna(shares) else None

    def fetch_free_float_factor(self, ticker: str, date: str) -> Optional[float]:
        """
        Fetches free-float factor from a CSV.
        CSV format: Date, FreeFloatFactor. Assumes data is point-in-time or use last known.
        """
        file_name = self.freefloat_path_template.format(ticker=ticker)
        file_path = os.path.join(self.base_path, "fundamentals", file_name) # Assuming subdirectory

        df = self._read_csv_data(file_path, date_col='Date', expected_cols=['Date', 'FreeFloatFactor'])
        if df.empty:
            return None

        target_date = pd.to_datetime(date)
        # Find the latest record on or before the target date
        df_filtered = df[df['Date'] <= target_date].sort_values(by='Date', ascending=False)

        if df_filtered.empty:
            logger.warning(f"No free-float factor data found for {ticker} on or before {date}")
            return None
            
        free_float = df_filtered['FreeFloatFactor'].iloc[0]
        if pd.notna(free_float) and 0.0 <= free_float <= 1.0:
            return float(free_float)
        else:
            logger.warning(f"Invalid free-float factor {free_float} found for {ticker} on {date}. Returning None.")
            return None