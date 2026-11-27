# src/beacon/analysis/analytics/profile/columns.py
"""The names, labels and units of the columns of a name's profile."""
# BN-468..BN-474; BN-477 the return and risk's.
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ....catalogue import COUNT, DAYS, FRACTION, MONEY, RATIO
from ...reports.statistics import Column
from .trading import EVENT_LABELS

ACTIVE = "active"
REMOVED = "removed"
CONTRIBUTION, ACTIVE_CONTRIBUTION = "contribution", "active_contribution"
TO_DATE = "_to_date"
COUNTRY, SECTOR, CURRENCY = "country", "sector", "currency"
# The cuts of the return and risk, by code.
RUN, REBALANCE, YEAR, QUARTER, MONTH = "run", "rebalance", "year", "quarter", "month"
PERIODS = {RUN: "Whole window", REBALANCE: "Rebalance period", YEAR: "Year",
           QUARTER: "Quarter", MONTH: "Month"}

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
          "r_squared": "R-squared", "observations": "Observations",
          "size_position": "Size position", "style_position": "Style position", "size": "Size",
          "style": "Style", "moved": "Moved box", "name": "Name", "market_cap": "Market cap",
          "free_float": "Free float", "type": "Type", "amount": "Amount or ratio",
          "side": "Side", "quantity": "Shares", "price": "Price", "notional": "Notional",
          "cost": "Cost", "impact": "Market impact", "reason": "Reason", "tags": "Tags",
          "event": "Event", "detail": "Detail", "rule": "Rule", "rule_name": "Rule type",
          "measure": "Measured", "measured": "Value", "comparison": "Comparison",
          "threshold": "Threshold", "margin": "Margin", "passed": "Passed",
          "period": "Period", "start": "From", "return": "Return", "volatility": "Volatility"}
UNITS = {"value": MONEY, "traded_value": MONEY, "days_to_sell": DAYS, "observations": COUNT,
         "market_cap": MONEY, "size_position": None, "style_position": None, "amount": None,
         "quantity": None, "price": MONEY, "notional": MONEY, "cost": MONEY, "impact": MONEY,
         "rule": None, "measured": None, "threshold": None}
# The labels of a column of codes, by code.
CODES = {"event": EVENT_LABELS,
         "comparison": {"gt": "Above", "ge": "At least", "lt": "Below", "le": "At most",
                        "eq": "Equal to", "ne": "Not equal to", "in": "One of",
                        "between": "Between"},
         "period": PERIODS}

# A factor's columns, named by the factor after the prefix.
FACTOR_COLUMNS = {"exposure/": "{}", "beta/": "{} beta", "t_stat/": "{} t-statistic"}
# A figure of the name against a portfolio, named ``<figure>/<portfolio>``.
PORTFOLIO_COLUMNS = {"excess_return": ("Excess return over the {}", FRACTION),
                     "contribution": ("Contribution to the {}", FRACTION),
                     "beta": ("Beta to the {}", RATIO),
                     "correlation": ("Correlation with the {}", RATIO),
                     "volatility_share": ("Share of the {}'s volatility", FRACTION)}
PORTFOLIO_NAMES = {"fund": "fund", "index": "index", "optimised": "optimised index",
                   "benchmark": "benchmark"}


def columns_of(table: pd.DataFrame) -> list[Column]:
    """The label and unit of each of a part's columns, in its order: a
    text or yes-or-no column has none, and a reference field the run kept
    is labelled from its name."""
    return [_column(str(name), is_numeric_dtype(dtype) and not is_bool_dtype(dtype))
            for name, dtype in table.dtypes.items()]


def _column(name: str,
            numeric: bool) -> Column:
    figure, _, portfolio = name.partition("/")

    if figure in PORTFOLIO_COLUMNS and portfolio in PORTFOLIO_NAMES:
        label, unit = PORTFOLIO_COLUMNS[figure]

        return Column(name, label.format(PORTFOLIO_NAMES[portfolio]), unit)

    for prefix, label in FACTOR_COLUMNS.items():
        if name.startswith(prefix):
            return Column(name, label.format(_titled(name[len(prefix):])), RATIO)

    label = LABELS.get(name) or _titled(name)

    return Column(name, label, UNITS.get(name, FRACTION) if numeric else None,
                  labels=CODES.get(name))


def _titled(name: str) -> str:
    """*name* as a label: ``country_domicile`` as ``Country domicile``."""
    words = name.replace("_", " ")

    return words[:1].upper() + words[1:]


__all__ = ["ACTIVE", "LABELS", "columns_of"]
