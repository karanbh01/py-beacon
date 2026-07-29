"""
DataFetcher — unified interface for querying market and reference data.

Accepts single identifiers or lists and passes through column names as-is.
"""


import pandas as pd

from .base import MarketData, ReferenceData


class DataFetcher:
    """Unified query interface over MarketData and ReferenceData.

    Args:
        market_data: Time-series market data container.
        reference_data: Reference data container.
    """

    def __init__(self,
                 market_data: MarketData,
                 reference_data: ReferenceData | None = None):
        self._market = market_data
        self._reference = reference_data

    # -- properties ----------------------------------------------------------

    @property
    def identifiers(self) -> list[str]:
        """Unique identifiers present in market data."""
        return self._market.identifiers

    @property
    def market_columns(self) -> list[str]:
        """Column names in the market data."""
        return self._market.columns

    @property
    def reference_identifiers(self) -> list[str] | None:
        """Unique identifiers in the reference data, or None if not loaded."""
        if self._reference is None:
            return None

        return self._reference.identifiers

    @property
    def reference_columns(self) -> list[str] | None:
        """Column names in the reference data, or None if not loaded."""
        if self._reference is None:
            return None
        return self._reference.columns

    @property
    def date_range(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest, latest) timestamps in the market data."""
        return self._market.date_range

    # -- market data ---------------------------------------------------------

    def fetch_market_data(self,
                          identifier: str | list[str],
                          start_date: str | None = None,
                          end_date: str | None = None,
                          columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch time-series market data for one or more identifiers.

        Args:
            identifier: One identifier or a list of identifiers.
            start_date: Date string to filter the start of the date range.
            end_date: Date string to filter the end of the date range.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Single identifier: indexed by ``DATE``. Multiple
            identifiers: MultiIndexed by ``(IDENTIFIER, DATE)``. Empty
            DataFrame if no matching data is found.
        """
        return self._market.get(identifier, start_date, end_date, columns)

    # -- auxiliary market data -----------------------------------------------
    #
    # Shares outstanding, free-float factors and FX rates are all sourced from
    # the market-data container, which is the single home for these series:
    #   * shares outstanding  -> a per-(identifier, date) column
    #     (``SHARES_OUTSTANDING`` by default)
    #   * free-float factor    -> a per-(identifier, date) column
    #     (``FREE_FLOAT`` by default)
    #   * FX rates             -> a currency pair stored as its own identifier,
    #     named ``"{FROM}{TO}"`` (e.g. ``GBPUSD``)
    # Each accessor returns ``None`` / an empty series when the backing column
    # or identifier is absent, so callers can fall back gracefully.

    def fetch_shares_outstanding(self,
                                 identifier: str,
                                 date: str,
                                 column: str = "SHARES_OUTSTANDING") -> float | None:
        """Return shares outstanding for *identifier* on *date*.

        Sourced from the *column* market-data field. Returns ``None`` if the
        column is not present or there is no value on that date.
        """
        return self._market_scalar(identifier, date, column)

    def fetch_free_float_factor(self,
                                identifier: str,
                                date: str,
                                column: str = "FREE_FLOAT") -> float | None:
        """Return the free-float factor for *identifier* on *date*.

        Sourced from the *column* market-data field. Returns ``None`` if the
        column is not present or there is no value on that date.
        """
        return self._market_scalar(identifier, date, column)

    def _market_scalar(self,
                       identifier: str,
                       date: str,
                       column: str) -> float | None:
        """Read a single market-data value for *identifier* on *date*."""
        if column not in self._market.columns:
            return None
        df = self._market.get(identifier, date, date, columns=[column])
        if df.empty:
            return None
        val = df[column].iloc[0]
        return float(val) if pd.notna(val) else None

    def fetch_fx_rates(self,
                       from_currency: str,
                       to_currency: str,
                       start_date: str | None = None,
                       end_date: str | None = None,
                       column: str = "RATE") -> pd.Series:
        """Return the FX rate series converting *from_currency* into *to_currency*.

        The pair is looked up as a market-data identifier named
        ``f"{from_currency}{to_currency}"`` (upper-cased). The *column* field is
        used if present, otherwise the first data column. Returns an empty
        Series if the pair is not found.
        """
        pair = f"{from_currency}{to_currency}".upper()
        if pair not in self._market.identifiers:
            return pd.Series(dtype=float)
        df = self._market.get(pair, start_date, end_date)
        if df.empty:
            return pd.Series(dtype=float)
        rate_col = column if column in df.columns else df.columns[0]
        return df[rate_col]

    # -- reference data ------------------------------------------------------

    def fetch_reference_data(self,
                             identifier: str | list[str],
                             date: str | None = None,
                             columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch reference data for one or more identifiers.

        Args:
            identifier: One identifier or a list of identifiers.
            date: Point-in-time date. Only rows valid at this date are
                returned.
            columns: Subset of columns to return.

        Returns:
            pd.DataFrame: Indexed by ``IDENTIFIER``. Empty DataFrame if no
            reference data is loaded or identifier is not found.
        """
        if self._reference is None:
            return pd.DataFrame()

        return self._reference.get(identifier, date, columns)
