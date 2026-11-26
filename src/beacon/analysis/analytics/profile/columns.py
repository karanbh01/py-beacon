# src/beacon/analysis/analytics/profile/columns.py
"""The names, labels and units of the columns of a name's profile."""
# BN-468..BN-471.
import pandas as pd

from ....catalogue import COUNT, DAYS, FRACTION, MONEY, RATIO
from ...reports.statistics import Column

ACTIVE = "active"
REMOVED = "removed"
CONTRIBUTION, ACTIVE_CONTRIBUTION = "contribution", "active_contribution"
TO_DATE = "_to_date"
COUNTRY, SECTOR, CURRENCY = "country", "sector", "currency"

# Each column's label, and its unit when it is not a fraction.
LABELS = {"target": "Target", "screened": "After the screens", "weight": "Traded to",
          "capped": "Capped to", REMOVED: "Removed by", "fund": "Fund", "index": "Index",
          "optimised": "Optimised index", "benchmark": "Benchmark", ACTIVE: "Active",
          CONTRIBUTION: "Contribution", ACTIVE_CONTRIBUTION: "Active contribution",
          CONTRIBUTION + TO_DATE: "Contribution to date",
          ACTIVE_CONTRIBUTION + TO_DATE: "Active contribution to date",
          "volatility_contribution": "Contribution to volatility",
          "volatility_share": "Share of volatility",
          "tracking_error_contribution": "Contribution to tracking error",
          "tracking_error_share": "Share of tracking error", "value": "Holding",
          "traded_value": "Average daily traded value", "days_to_sell": "Days to sell",
          COUNTRY: "Country", SECTOR: "Sector", CURRENCY: "Currency",
          "r_squared": "R-squared", "observations": "Observations"}
UNITS = {"value": MONEY, "traded_value": MONEY, "days_to_sell": DAYS, "observations": COUNT}
TEXT = (REMOVED, COUNTRY, SECTOR, CURRENCY)

# A factor's columns, named by the factor after the prefix.
FACTOR_COLUMNS = {"exposure/": "{}", "beta/": "{} beta", "t_stat/": "{} t-statistic"}


def columns_of(table: pd.DataFrame) -> list[Column]:
    """The label and unit of each of a part's columns, in its order."""
    return [_column(str(name)) for name in table.columns]


def _column(name: str) -> Column:
    for prefix, label in FACTOR_COLUMNS.items():
        if name.startswith(prefix):
            factor = name[len(prefix):].replace("_", " ")

            return Column(name, label.format(factor[:1].upper() + factor[1:]), RATIO)

    return Column(name, LABELS[name], None if name in TEXT else UNITS.get(name, FRACTION))


__all__ = ["ACTIVE", "LABELS", "columns_of"]
