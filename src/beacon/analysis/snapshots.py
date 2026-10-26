# src/beacon/analysis/snapshots.py
"""
What a run's names looked like at each rebalance, kept with the run.

A rerun of a run's analysis months later should give the same answer even if
the data store has been refreshed since, so what the analysis groups and
weighs names by is read as it stood at each rebalance and kept with the run.

Each snapshot is a long table indexed by date, a row per name per rebalance,
with an ``IDENTIFIER`` column and a column per field. A name with no record in
force on a date has no row for it. Between rebalances, the latest snapshot on
or before a day is the one in force.
"""
# BN-366.
import pandas as pd

from ..backtest.result import BacktestResult
from ..data.fetcher import DataFetcher

IDENTIFIER = "IDENTIFIER"

# The fields names are grouped by, where the reference data has them.
GROUPING_FIELDS = ("SECTOR", "GICS_INDUSTRY_GROUP", "GICS_INDUSTRY", "SUB_INDUSTRY",
                   "REGION", "COUNTRY_LISTING", "COUNTRY_DOMICILE", "CURRENCY",
                   "EXCHANGE")


def rebalance_dates(result: BacktestResult) -> list[pd.Timestamp]:
    """Every day the fund or its index rebalanced, within the run, sorted."""
    days = result.portfolio.nav.index
    dates = {pd.Timestamp(row.date) for row in result.rebalance_pricing}
    books = [result.index.target, result.index.optimised, result.benchmark]

    for book in books:
        if book is not None and book.source is not None:
            dates |= {pd.Timestamp(date) for date in book.source.constituent_snapshots}

    if len(days) == 0:
        return []

    return sorted(date for date in dates if days[0] <= date <= days[-1])


def reference_snapshots(source: DataFetcher,
                        names: list[str],
                        dates: list[pd.Timestamp]) -> pd.DataFrame:
    """The grouping fields of *names* as they stood on each of *dates*.

    Args:
        source: The data the run used.
        names: The names to describe.
        dates: The days to describe them on.

    Returns:
        pd.DataFrame: Indexed by date, with ``IDENTIFIER`` and each of
        `GROUPING_FIELDS` the reference data has, as text; missing where a
        name has no value for a field.
    """
    frames = []

    for date in dates:
        frame = source.fetch_reference_data(names, str(date.date()))

        if frame.empty:
            continue

        fields = [field for field in GROUPING_FIELDS if field in frame.columns]
        rows = frame[fields].copy()
        rows.insert(0, IDENTIFIER, rows.index.astype(str))
        rows.index = pd.DatetimeIndex([date] * len(rows))
        frames.append(rows)

    if not frames:
        return pd.DataFrame(columns=[IDENTIFIER])

    return pd.concat(frames)
