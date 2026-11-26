# src/beacon/analysis/analytics/profile/weights.py
"""A name's profile's weights: at each rebalance and each day."""
# BN-468 (Karan, 2026-11-25).
import pandas as pd

from ...panels import REBALANCE_WEIGHTS
from ...reports.context import FUND
from ...reports.statistics import RUN_PREDATES, Unavailable
from .columns import ACTIVE, REMOVED

# The rebalance table's columns as the profile names them.
STAGES = {"TARGET": "target", "SCREENED": "screened", "WEIGHT": "weight", "CAPPED": "capped",
          "REMOVED": "removed"}
HELD = ("target", "screened", "weight")


def rebalance_weights(name: str,
                      tables: dict[str, pd.DataFrame],
                      days: pd.DatetimeIndex,
                      portfolios: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """At each rebalance over *days*: the strategy's weight in *name* at each
    implementation stage, any cap or screen applied to it, and each
    comparator's weight in it that day.

    Raises:
        Unavailable: If the run was kept before its rebalances were.
    """
    if REBALANCE_WEIGHTS not in tables:
        raise Unavailable(RUN_PREDATES)

    kept: pd.DataFrame = tables[REBALANCE_WEIGHTS]
    dates = pd.DatetimeIndex(pd.unique(kept.index))
    dates = dates[(dates >= days[0]) & (dates <= days[-1])]
    mine = kept.loc[kept["IDENTIFIER"] == name, list(STAGES)].rename(columns=STAGES)
    mine = mine[~mine.index.duplicated(keep="last")]

    found = mine.reindex(dates)
    found[list(HELD)] = found[list(HELD)].fillna(0.0)
    # Blank as None, as a stored run's all-blank column reads back NaN.
    found[REMOVED] = found[REMOVED].astype(object).where(found[REMOVED].notna(), None)

    for book, weights in portfolios.items():
        if book != FUND:
            found[book] = column(weights, name).reindex(dates)

    return found.rename_axis("DATE")


def daily_weights(name: str,
                  days: pd.DatetimeIndex,
                  portfolios: dict[str, pd.DataFrame],
                  against: str | None) -> pd.DataFrame:
    """Each of *days*: the fund's and each comparator's weight in *name*, and
    the fund's active weight against *against*."""
    found = pd.DataFrame({book: column(weights, name).reindex(days).ffill()
                          for book, weights in portfolios.items()}, index=days)

    if FUND in found and against in found:
        found[ACTIVE] = found[FUND] - found[against]

    found.attrs = {"against": against if ACTIVE in found else None}

    return found.rename_axis("DATE")


def column(weights: pd.DataFrame,
           name: str) -> pd.Series:
    """*name*'s column of *weights*, 0 where it is blank or absent."""
    if name not in weights.columns:
        return pd.Series(0.0, index=weights.index)

    return weights[name].fillna(0.0)
