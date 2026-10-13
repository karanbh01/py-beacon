# src/beacon/plot/notes.py
"""
What a result chart's notes line says: where the data came from, the dates
it covers, and whatever the chart itself needs said, run together as
sentences rather than listed.
"""
# BN-299: moved from beacon.plot.frame so the result charts, which only
# describe charts, need no matplotlib.
from typing import Any

import pandas as pd

# What a data store records as its source, in words.
SOURCE_NAMES = {
    "synthetic": "Synthetic data",
    "yfinance": "Yahoo Finance",
    "local": "Local data store",
    "imported": "Imported data",
    "database": "Database",
}


def provenance(result: Any,
               what: str,
               dates: pd.Index) -> str:
    """Where a result's data came from and the dates it covers, as a sentence.

    Args:
        result: Anything bound to the data it was calculated from.
        what: What the dates are of, such as "backtest" or "index".
        dates: The dates drawn.

    Returns:
        str: Such as "Source: Synthetic data, backtest dated from 02/01/2024
        to 31/12/2024.", or "weights as of 01/10/2024" for a single date. The
        source is left out when the data recorded none, and the whole is
        empty when there are no dates.
    """
    if not len(dates):
        return ""

    window = (f"{what} as of {_day(dates[0])}" if len(dates) == 1
              else f"{what} dated from {_day(dates[0])} to {_day(dates[-1])}")
    fetcher = getattr(result, "_data_fetcher", None)
    source = getattr(fetcher, "source", None)

    if source:
        return f"Source: {SOURCE_NAMES.get(source, source)}, {window}."

    return f"{window[0].upper()}{window[1:]}."


def sentences(*parts: str) -> str:
    """Notes run together as sentences, skipping the empty ones."""
    return " ".join(part.strip() for part in parts if part and part.strip())


def _day(when: Any) -> str:
    """A date as dd/mm/yyyy."""
    return str(pd.Timestamp(when).strftime("%d/%m/%Y"))
