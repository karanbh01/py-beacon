# src/beacon/analysis/class_panels.py
"""
The tables a run with share classes keeps of each class: its days, its flows
and its hedge's trades, a row each, with the class's name in ``CLASS``.

Amounts are in the class's own currency, as its record keeps them, except
``BASE_NET_ASSETS`` and the hedge's ``CARRY`` and ``COST``, which are in the
fund's base currency; a day's net assets in both give the rate between them.
A run kept before 0.11.0 named ``BASE_NET_ASSETS`` ``BOOK_NET_ASSETS``, and
is read with the name it has now.
"""
# BN-408.
from typing import Any

import pandas as pd

from ..backtest.result import BacktestResult

CLASS_DAYS = "class_days"
CLASS_FLOWS = "class_flows"
CLASS_HEDGES = "class_hedges"

DAY_COLUMNS = ("CLASS", "NET_ASSETS", "BASE_NET_ASSETS", "UNITS", "NAV_PER_UNIT", "FEES",
               "DISTRIBUTIONS", "HEDGE_VALUE")
FLOW_COLUMNS = ("CLASS", "AMOUNT", "UNITS", "DEALING_PRICE", "ADJUSTMENT")
HEDGE_COLUMNS = ("CLASS", "REASON", "NOTIONAL", "RATE", "SPOT", "EXPIRY", "SETTLED", "CARRY",
                 "COST")

# BN-479: the columns a run kept before 0.11.0 named otherwise, by table.
OLD_COLUMNS = {CLASS_DAYS: {"BOOK_NET_ASSETS": "BASE_NET_ASSETS"}}


def class_tables(result: BacktestResult) -> dict[str, pd.DataFrame]:
    """The three tables, by name; none for a run without share classes."""
    if not result.fund.share_classes:
        return {}

    days, flows, hedges = [], [], []

    for name, share_class in result.fund.share_classes.items():
        frame = pd.DataFrame({"NET_ASSETS": share_class.net_assets,
                              "BASE_NET_ASSETS": share_class.base_net_assets,
                              "UNITS": share_class.units,
                              "NAV_PER_UNIT": share_class.nav_per_unit,
                              "FEES": share_class.fees,
                              "DISTRIBUTIONS": share_class.distributions,
                              "HEDGE_VALUE": share_class.hedge_value}).fillna(
            {"FEES": 0.0, "DISTRIBUTIONS": 0.0, "HEDGE_VALUE": 0.0})
        frame.insert(0, "CLASS", name)
        days.append(frame)
        flows += [{"DATE": record.date, "CLASS": name, "AMOUNT": record.amount,
                   "UNITS": record.units, "DEALING_PRICE": record.dealing_price,
                   "ADJUSTMENT": record.adjustment} for record in share_class.flows]
        hedges += [{"DATE": trade.date, "CLASS": name, "REASON": trade.reason,
                    "NOTIONAL": trade.notional, "RATE": trade.rate, "SPOT": trade.spot,
                    "EXPIRY": trade.expiry, "SETTLED": trade.settled, "CARRY": trade.carry,
                    "COST": trade.cost} for trade in share_class.hedges]

    day_table = pd.concat(days)
    day_table.index = pd.DatetimeIndex(day_table.index, name="DATE")

    return {CLASS_DAYS: day_table[list(DAY_COLUMNS)],
            CLASS_FLOWS: _dated(flows, FLOW_COLUMNS),
            CLASS_HEDGES: _dated(hedges, HEDGE_COLUMNS)}


def with_current_names(name: str,
                       frame: pd.DataFrame) -> pd.DataFrame:
    """*frame*, the stored table *name*, with each column a run kept before
    0.11.0 named otherwise under the name it has now."""
    renamed = OLD_COLUMNS.get(name)

    return frame.rename(columns=renamed) if renamed else frame


def _dated(rows: list[dict[str, Any]],
           columns: tuple[str, ...]) -> pd.DataFrame:
    """*rows* as a frame indexed by ``DATE``, with *columns* even when empty."""
    frame = pd.DataFrame(rows, columns=["DATE", *columns])
    frame.index = pd.DatetimeIndex(frame.pop("DATE"), name="DATE")

    return frame
