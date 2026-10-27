# src/beacon/backtest/breakdown.py
"""
A run's money, day by day: what it cost, and where its cash went.

    costs = daily_costs(result)
    cash = daily_cash(result)

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
from ..portfolio.cash_flows import (
    DISTRIBUTION,
    DIVIDEND,
    FEE,
    FUTURES,
    HEDGE,
    INTEREST,
    REDEMPTION,
    SUBSCRIPTION,
    SWAP,
)
from .futures import ROLL
from .result import BacktestResult

# The columns of `daily_costs`.
FIXED = "fixed"
IMPACT = "impact"
FUTURES_ROLLS = "futures_rolls"
# BN-409.
FUTURES_TRADING = "futures_trading"
CREATION_FEES = "creation_fees"
COST_KINDS = (FIXED, IMPACT, FUTURES_ROLLS, FUTURES_TRADING, CREATION_FEES)

# The columns of `daily_cash`: balances at the end of each day, cash that
# moved, and what was charged or owed without cash moving.
BALANCES = ("cash", "dividends_receivable", "fees_payable", "futures_exposure")
MOVEMENTS = {"interest": INTEREST, "dividends_received": DIVIDEND,
             "distributions": DISTRIBUTION, "subscriptions": SUBSCRIPTION,
             "redemptions": REDEMPTION, "fees_paid": FEE, "futures_settled": FUTURES,
             "swap_settled": SWAP, "hedge_settled": HEDGE}
ACCRUALS = ("dividends_gross", "withholding_tax", "fee_charged", "flows_dealt")
CASH_KINDS = BALANCES + tuple(MOVEMENTS) + ACCRUALS


def daily_costs(result: BacktestResult) -> pd.DataFrame:
    """What the run cost, by day and kind.

    Columns:

    - ``fixed``: trading costs at the run's fixed rate, on every trade that
      paid them.
    - ``impact``: the market impact trades paid, when the run models it.
    - ``futures_rolls``: what rolling index futures to the next contract
      cost: the roll cost and the cost per contract of both legs.
    - ``futures_trading``: what trading them between rolls cost: opening the
      position and each resize.
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
    between = [trade for trade in result.futures_trades if trade.reason != ROLL]
    costs = pd.Series([trade.transaction_cost for trade in trades], dtype=float)
    impact = pd.Series([trade.impact_cost for trade in trades], dtype=float)
    trade_days = [trade.transaction_date for trade in trades]

    columns = {
        FIXED: _by_day(trade_days, costs - impact),
        IMPACT: _by_day(trade_days, impact),
        FUTURES_ROLLS: _by_day([roll.date for roll in result.futures_rolls],
                               [roll.cost for roll in result.futures_rolls]),
        FUTURES_TRADING: _by_day([trade.date for trade in between],
                                 [trade.cost for trade in between]),
        CREATION_FEES: _by_day([flow.date for flow in result.flows],
                               [flow.adjustment for flow in result.flows]),
    }

    return _table(result, columns)


# BN-364. Gross dividends and withholding are recorded by the engine on each
# ex-date, when the NAV first counts them; the rest is on the run's record.
# The trades table's columns (BN-406).
TRADE_COLUMNS = ("IDENTIFIER", "SIDE", "QUANTITY", "PRICE", "VALUE", "COST", "IMPACT",
                 "REASON", "TAGS")

# What joins a trade's tags in the table.
TAG_SEPARATOR = "; "


def trades(result: BacktestResult) -> pd.DataFrame:
    """Every trade the run made, in the order it made them.

    Returns:
        pd.DataFrame: Indexed by the trade's date (``DATE``), with the name,
        ``BUY`` or ``SELL``, the quantity, the price, the value traded
        (quantity times price), the cost and the part of it that was market
        impact, all in the book's currency; why the trade was made
        (``REASON``) and every tag it carries, joined by ``"; "`` (``TAGS``),
        as `beacon.portfolio.reasons` names them.
    """
    made = result.portfolio.transactions

    frame = pd.DataFrame({
        "IDENTIFIER": [trade.asset_id for trade in made],
        "SIDE": [trade.transaction_type for trade in made],
        "QUANTITY": pd.Series([trade.quantity for trade in made], dtype=float),
        "PRICE": pd.Series([trade.price for trade in made], dtype=float),
        "VALUE": pd.Series([trade.quantity * trade.price for trade in made], dtype=float),
        "COST": pd.Series([trade.transaction_cost for trade in made], dtype=float),
        "IMPACT": pd.Series([trade.impact_cost for trade in made], dtype=float),
        "REASON": [trade.reason for trade in made],
        "TAGS": [TAG_SEPARATOR.join(trade.tags) for trade in made],
    }, columns=list(TRADE_COLUMNS))
    frame.index = pd.DatetimeIndex([trade.transaction_date for trade in made], name="DATE")

    return frame


def daily_cash(result: BacktestResult) -> pd.DataFrame:
    """Where the run's cash went, and what it was owed and charged, by day.

    Balances, at the end of each day:

    - ``cash``: the cash held.
    - ``dividends_receivable``: dividends gone ex and not yet paid, net of
      withholding, which the NAV includes.
    - ``fees_payable``: the management fee charged and not yet paid, which
      the NAV is net of.
    - ``futures_exposure``: what any index futures were exposed to.

    Cash that moved, positive into the fund and negative out of it:
    ``interest``, ``dividends_received`` (net of withholding, on the pay
    date), ``distributions``, ``subscriptions``, ``redemptions``,
    ``fees_paid``, ``futures_settled`` (their daily margin and roll costs),
    ``swap_settled`` and ``hedge_settled``.

    Charged or owed without cash moving:

    - ``dividends_gross``: dividends the book became owed on their ex-dates,
      before tax.
    - ``withholding_tax``: the tax withheld from them, on the same days.
    - ``fee_charged``: the management fee charged for the day.
    - ``flows_dealt``: investors' flows as dealt, positive in.

    Args:
        result: The finished run.

    Returns:
        pd.DataFrame: Dates of the NAV by kind, in the book's currency, 0
        where nothing of a kind happened.
    """
    days = result.portfolio.nav.index
    flows = result.portfolio.cash_flows
    columns: dict[str, pd.Series] = {}

    for name, kind in MOVEMENTS.items():
        moved = [flow for flow in flows if flow.kind == kind]
        columns[name] = _by_day([flow.date for flow in moved], [flow.amount for flow in moved])

    entitled = result.dividends_entitled
    columns["dividends_gross"] = _by_day(list(entitled.index), entitled["GROSS"])
    columns["withholding_tax"] = _by_day(list(entitled.index), entitled["WITHHELD"])
    columns["flows_dealt"] = _by_day([flow.date for flow in result.flows],
                                     [flow.amount for flow in result.flows])

    table = _table(result, columns)
    balances = {"cash": result.portfolio.cash, "dividends_receivable": result.dividends_receivable,
                "fees_payable": result.fees_payable, "futures_exposure": result.futures_notional}

    for name, series in balances.items():
        table[name] = _balance(series, days)

    # The day's charge: what is owed at its end, less what was owed at the
    # start, plus what was paid.
    table["fee_charged"] = (table["fees_payable"].diff().fillna(table["fees_payable"])
                            - table["fees_paid"])

    return table[list(CASH_KINDS)]


def _balance(series: pd.Series,
             days: pd.DatetimeIndex) -> pd.Series:
    """A balance recorded when it changed, on every day: carried forward from
    its last record, 0 before its first."""
    if series.empty:
        return pd.Series(0.0, index=days)

    recorded = series.copy()
    recorded.index = pd.DatetimeIndex(recorded.index).normalize()
    recorded = recorded.groupby(level=0).last()

    return recorded.reindex(recorded.index.union(days)).ffill().reindex(days).fillna(0.0)


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
