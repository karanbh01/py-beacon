# src/beacon/analysis/liquidity.py
"""
Liquidity measures computed from held market data.

Average daily volume (ADV) is the figure a universe table shows for every name.

## The trailing window is calendar months, not a day count

"ADV 3M" means the last three *calendar* months, so the window is
``as_of - 3 months < date <= as_of``. The two differ by several days across a
quarter, and quoting a 63-trading-day average as "3M" makes two vendors
disagree for no reason anybody can see. This matches the trailing-twelve-month
dividend convention in `beacon.data.corporate_actions`, including the
half-open boundary: a day exactly three months old has rolled out.

## Missing data is NaN, not zero

A name whose volume is missing on every day in the window gets NaN rather
than 0.0, and a name with no rows in the window is left out of the result.
Zero is a claim (that it traded and nobody bought) and a liquidity screen
reading it would exclude the name for the wrong reason. NaN says the question
has no answer here, and serialises to `null`.
"""
# ADV was the one universe-table figure nothing in Beacon computed. Without it
# a client wanting ADV alongside reference data had to pull a full price
# series per identifier and average it itself: five hundred requests to
# produce five hundred numbers the server already holds every input for.
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# The market-data column holding traded share counts.
VOLUME_COLUMN = "VOLUME"

# Months in the default trailing window: "ADV 3M".
TRAILING_MONTHS = 3


def average_daily_volume(market: pd.DataFrame,
                         as_of: pd.Timestamp,
                         months: int = TRAILING_MONTHS,
                         column: str = VOLUME_COLUMN) -> pd.Series:
    """Mean daily volume over the trailing window, per identifier.

    Args:
        market: Market data MultiIndexed by ``(IDENTIFIER, DATE)``, as
            ``DataFetcher.fetch_market_data`` returns for a list of
            identifiers.
        as_of: End of the window, inclusive.
        months: Length of the window in calendar months.
        column: Volume column to average.

    Returns:
        pd.Series: Indexed by identifier. Empty when the frame is empty,
        carries no volume column, or has no rows in the window (the last
        case logs a warning). An absent column is a property of the dataset,
        not a failure of the request, so it produces no answer rather than
        an error.
    """
    if market.empty or column not in market.columns:
        return pd.Series(dtype="float64")

    dates = market.index.get_level_values("DATE")
    start = pd.Timestamp(as_of) - pd.DateOffset(months=months)

    window = market.loc[(dates > start) & (dates <= pd.Timestamp(as_of))]
    if window.empty:
        logger.warning(
            "No volume in the %d month(s) to %s, so no ADV could be computed.",
            months, pd.Timestamp(as_of).date())

        return pd.Series(dtype="float64")

    return window[column].groupby(level="IDENTIFIER").mean()
