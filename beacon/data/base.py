"""
Base data containers for market and reference data.
"""

import os
from typing import List, Optional, Tuple, Union

import pandas as pd


def _read_file(file_path: str) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame based on extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xls", ".xlsx"):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Use .csv, .xls, or .xlsx.")


class MarketData:
    """Time-series data container backed by a MultiIndex DataFrame.

    The source file must contain at least ``IDENTIFIER`` and ``DATE`` columns.
    After loading the DataFrame is indexed on ``(IDENTIFIER, DATE)`` and sorted,
    enabling fast ``.loc`` slicing by identifier or list of identifiers.
    """

    def __init__(self, 
                 file_path: str, 
                 date_format: str = "%Y-%m-%d"):
        df = _read_file(file_path)

        missing = [c for c in ("IDENTIFIER", "DATE") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        df["DATE"] = pd.to_datetime(df["DATE"], format=date_format)
        df = df.set_index(["IDENTIFIER", "DATE"]).sort_index()
        self._df = df

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

    @property
    def identifiers(self) -> List[str]:
        """Unique identifiers present in the dataset."""
        return list(self._df.index.get_level_values("IDENTIFIER").unique())

    @property
    def columns(self) -> List[str]:
        """Non-index column names."""
        return list(self._df.columns)

    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """(earliest, latest) timestamps in the dataset."""
        dates = self._df.index.get_level_values("DATE")
        return dates.min(), dates.max()

    # -- query ---------------------------------------------------------------

    def get(
        self,
        identifier: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return data for one or more identifiers, optionally filtered.

        Parameters
        ----------
        identifier : str or list of str
            Single identifier or list of identifiers.
        start_date, end_date : str, optional
            Date strings to slice the date range.
        columns : list of str, optional
            Subset of columns to return.

        Returns
        -------
        pd.DataFrame
            Single identifier: indexed by ``DATE``.
            List of identifiers: MultiIndexed by ``(IDENTIFIER, DATE)``.
            Empty DataFrame if no matching data is found.
        """
        if isinstance(identifier, list):
            existing = self._df.index.get_level_values("IDENTIFIER")
            identifier = [i for i in identifier if i in existing]
            if not identifier:
                return pd.DataFrame()

        try:
            subset = self._df.loc[identifier]
        except KeyError:
            return pd.DataFrame()

        if start_date is not None or end_date is not None:
            date_level = "DATE" if isinstance(identifier, list) else None
            if isinstance(identifier, list):
                dates = subset.index.get_level_values("DATE")
            else:
                dates = subset.index

            mask = pd.Series(True, index=subset.index)
            if start_date is not None:
                mask &= dates >= pd.Timestamp(start_date)
            if end_date is not None:
                mask &= dates <= pd.Timestamp(end_date)
            subset = subset.loc[mask]

        if columns is not None:
            subset = subset[columns]

        return subset


class ReferenceData:
    """Reference data container with validity ranges.

    The source file must contain ``IDENTIFIER``, ``DATE_FROM``, and ``DATE_TO``
    columns. ``DATE_TO`` may be NaT to indicate a currently-active record.

    Indexed on ``IDENTIFIER`` (non-unique, since an identifier may have
    multiple validity periods).
    """

    def __init__(self, file_path: str):
        df = _read_file(file_path)

        missing = [c for c in ("IDENTIFIER", "DATE_FROM") if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        if "DATE_TO" in df.columns:
            df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
        else:
            df["DATE_TO"] = pd.NaT

        df = df.set_index("IDENTIFIER").sort_index()
        self._df = df

    # -- properties ----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._df.copy()

    @property
    def identifiers(self) -> List[str]:
        """Unique identifiers present in the dataset."""
        return list(self._df.index.unique())

    @property
    def columns(self) -> List[str]:
        """Column names (including DATE_FROM, DATE_TO)."""
        return list(self._df.columns)

    # -- query ---------------------------------------------------------------

    def get(
        self,
        identifier: Union[str, List[str]],
        date: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return reference data for one or more identifiers.

        Parameters
        ----------
        identifier : str or list of str
            Single identifier or list of identifiers.
        date : str, optional
            Point-in-time date. If provided, only rows where
            ``DATE_FROM <= date`` and (``DATE_TO >= date`` or ``DATE_TO`` is
            NaT) are returned.
        columns : list of str, optional
            Subset of columns to return.

        Returns
        -------
        pd.DataFrame
            Indexed by ``IDENTIFIER``. Empty DataFrame if no match.
        """
        if isinstance(identifier, list):
            existing = self._df.index
            identifier = [i for i in identifier if i in existing]
            if not identifier:
                return pd.DataFrame()

        try:
            subset = self._df.loc[identifier]
        except KeyError:
            return pd.DataFrame()

        # .loc on a non-unique index with a single str returns a Series
        # if exactly one row matches — normalize to DataFrame.
        if isinstance(subset, pd.Series):
            subset = subset.to_frame().T
            subset.index.name = "IDENTIFIER"

        if date is not None:
            ts = pd.Timestamp(date)
            mask = subset["DATE_FROM"] <= ts
            mask &= subset["DATE_TO"].isna() | (subset["DATE_TO"] >= ts)
            subset = subset.loc[mask]

        if columns is not None:
            subset = subset[columns]

        return subset
