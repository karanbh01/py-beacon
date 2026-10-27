# src/beacon/analysis/performance/trading.py
"""
How much a run traded: its turnover each trading day, at each rebalance and
each year, and its trading by why it was made.

One-way turnover is the lesser of what was bought and what was sold, over the
fund's assets: the SEC's measure, which money arriving or leaving does not
inflate, as it is bought or sold on one side only. Two-way turnover is all of
it, bought and sold, over the same assets. A day's are over the NAV at the
close before its trades; a year's over its mean daily NAV.
"""
# BN-383. As agreed with Karan on 2026-10-27, with two-way turnover beside
# one-way and both on rebalance dates as well.
import numpy as np
import pandas as pd

from ...portfolio.reasons import REBALANCE

# The figures each view gives.
FIGURES = ("buys", "sells", "trades", "mean_trade", "one_way", "two_way")


def nav_before(nav: pd.Series,
               initial_capital: float) -> pd.Series:
    """The NAV at the close before each day: the day before's, the first's
    the starting capital."""
    before = nav.shift(1)

    if len(before):
        before.iloc[0] = initial_capital

    return before


def daily_turnover(trades: pd.DataFrame,
                   nav: pd.Series,
                   initial_capital: float) -> pd.DataFrame:
    """Each trading day's buys, sells, number of trades and their mean
    value, one-way and two-way turnover, and whether it was a rebalance."""
    table = _summed(trades, pd.DatetimeIndex(trades.index).normalize())
    assets = nav_before(nav, initial_capital).reindex(table.index)
    table = _turnover(table, assets)
    rebalanced = set(pd.DatetimeIndex(trades.index[trades["REASON"] == REBALANCE]).normalize())
    table["rebalance"] = [day in rebalanced for day in table.index]

    return table


def rebalance_turnover(trades: pd.DataFrame,
                       nav: pd.Series,
                       initial_capital: float) -> pd.DataFrame:
    """Each rebalance's own trades: its buys, sells, number of trades and
    their mean value, and its one-way and two-way turnover."""
    made = trades[trades["REASON"] == REBALANCE]
    table = _summed(made, pd.DatetimeIndex(made.index).normalize())

    return _turnover(table, nav_before(nav, initial_capital).reindex(table.index))


def yearly_turnover(trades: pd.DataFrame,
                    nav: pd.Series) -> pd.DataFrame:
    """Each calendar year's buys, sells, number of trades and their mean
    value, and its one-way and two-way turnover over its mean daily NAV."""
    years = pd.DatetimeIndex(trades.index).year.astype(str)
    table = _summed(trades, pd.Index(years, name="year"))
    assets = nav.groupby(pd.DatetimeIndex(nav.index).year.astype(str)).mean()

    return _turnover(table, assets.reindex(table.index))


def by_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """The run's trading by why it was made: the trades, what was bought and
    sold, and the share of all it traded, a row a reason, largest first.
    Each trade has one reason, so the rows add up to the run's trading."""
    values = trades["VALUE"].astype(float)
    table = pd.DataFrame({
        "trades": trades.groupby("REASON").size(),
        "buys": values.where(trades["SIDE"] == "BUY", 0.0).groupby(trades["REASON"]).sum(),
        "sells": values.where(trades["SIDE"] == "SELL", 0.0).groupby(trades["REASON"]).sum(),
    })
    traded = table["buys"] + table["sells"]
    table["share"] = traded / traded.sum() if traded.sum() > 0 else 0.0
    table.index.name = "reason"

    return table.sort_values("share", ascending=False)


def _summed(trades: pd.DataFrame,
            keys: pd.Index) -> pd.DataFrame:
    """The buys, sells, trades and mean trade by *keys*."""
    values = trades["VALUE"].astype(float).to_numpy()
    buying = (trades["SIDE"] == "BUY").to_numpy()
    frame = pd.DataFrame({"buys": np.where(buying, values, 0.0),
                          "sells": np.where(buying, 0.0, values),
                          "trades": 1}, index=keys)
    table = frame.groupby(level=0).sum()
    table["mean_trade"] = (table["buys"] + table["sells"]) / table["trades"]

    return table


def _turnover(table: pd.DataFrame,
              assets: pd.Series) -> pd.DataFrame:
    """*table* with one-way and two-way turnover over *assets*."""
    held = assets.where(assets > 0)
    table["one_way"] = np.minimum(table["buys"], table["sells"]) / held
    table["two_way"] = (table["buys"] + table["sells"]) / held

    return table[list(FIGURES)]
