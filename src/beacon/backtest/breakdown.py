# src/beacon/backtest/breakdown.py
"""
A run's money, day by day: what it cost, and where its cash went.

    costs = daily_costs(result)

Each table has a row for every day of the run's NAV and a column for each
kind, in the book's currency, and is 0 on a day nothing of that kind
happened. Summed over the run, each column is the run's total of its kind.
"""
# BN-363. Built from what the run recorded as it went (each trade's cost and
# its market impact, each roll and each flow) rather than worked out again
# afterwards: a trade booked at no cost, such as a delisting's settlement,
# would otherwise be charged the fixed rate.
import pandas as pd

from ..exceptions import CalculationError
from .result import BacktestResult

# The columns of `daily_costs`.
FIXED = "fixed"
IMPACT = "impact"
FUTURES_ROLLS = "futures_rolls"
CREATION_FEES = "creation_fees"
COST_KINDS = (FIXED, IMPACT, FUTURES_ROLLS, CREATION_FEES)


def daily_costs(result: BacktestResult) -> pd.DataFrame:
    """What the run cost, by day and kind.

    Columns:

    - ``fixed``: trading costs at the run's fixed rate, on every trade that
      paid them.
    - ``impact``: the market impact trades paid, when the run models it.
    - ``futures_rolls``: what rolling index futures to the next contract
      cost.
    - ``creation_fees``: what investors dealing in or out paid into the fund
      toward the trading their flow caused, through the dealing price or a
      levy (an ETF's creation fee among them). Received by the fund, so it
      offsets the costs above rather than adding to them.

    Args:
        result: The finished run.

    Returns:
        pd.DataFrame: Dates of the NAV by kind, in the book's currency, 0
        where nothing of a kind happened.
    """
    trades = result.portfolio.transactions
    costs = pd.Series([trade.transaction_cost for trade in trades], dtype=float)
    impact = pd.Series([trade.impact_cost for trade in trades], dtype=float)
    trade_days = [trade.transaction_date for trade in trades]

    columns = {
        FIXED: _by_day(trade_days, costs - impact),
        IMPACT: _by_day(trade_days, impact),
        FUTURES_ROLLS: _by_day([roll.date for roll in result.futures_rolls],
                               [roll.cost for roll in result.futures_rolls]),
        CREATION_FEES: _by_day([flow.date for flow in result.flows],
                               [flow.adjustment for flow in result.flows]),
    }

    return _table(result, columns)


def _by_day(days: list[pd.Timestamp],
            amounts: pd.Series | list[float]) -> pd.Series:
    """*amounts* summed by the day they fell on."""
    if len(days) == 0:
        return pd.Series(dtype=float)

    dates = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize()

    return pd.Series(list(amounts), index=dates, dtype=float).groupby(level=0).sum()


def _table(result: BacktestResult,
           columns: dict[str, pd.Series]) -> pd.DataFrame:
    """*columns* on the run's NAV dates, 0 where a column has nothing.

    Raises:
        CalculationError: If an amount fell on a day the NAV does not cover,
            which would otherwise be dropped from the totals unseen.
    """
    days = result.portfolio.nav.index

    for name, series in columns.items():
        outside = series.index.difference(days)

        if len(outside):
            raise CalculationError(
                "Daily breakdown",
                f"{name} on {outside[0].date()}, a day the run's NAV does not cover")

    return pd.DataFrame({name: series.reindex(days, fill_value=0.0)
                         for name, series in columns.items()}, index=days)
