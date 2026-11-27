# src/beacon/analysis/analytics/profile/risk.py
"""A name's profile's risk and liquidity at each rebalance."""
# BN-470 (Karan, 2026-11-26): the risk is kept with the run, as its analysis
# fitted it, else fitted when first asked for.
from typing import Any

import pandas as pd

from ...panels import TRADED_VALUES
from ...reports.context import FUND
from ...reports.statistics import NO_DATA_STORE, RUN_PREDATES, TOO_SHORT, Unavailable
from ...risk import risk_store
from ...risk.liquidity import LiquidityAnalytics


def risk(name: str,
         table: pd.DataFrame,
         days: pd.DatetimeIndex,
         against: str | None) -> pd.DataFrame:
    """At each rebalance over *days* in *table* (`risk_store.name_risk`):
    *name*'s contribution to and share of the fund's expected volatility,
    and of its expected tracking error against *against*.

    Raises:
        Unavailable: If no rebalance over *days* had its model fitted.
    """
    table = table.loc[(table.index >= days[0]) & (table.index <= days[-1])]

    if table.empty:
        raise Unavailable(TOO_SHORT)

    dates = pd.DatetimeIndex(sorted(table.index.unique()))
    mine = table.loc[table[risk_store.IDENTIFIER] == name]
    columns = {"volatility": FUND}

    if against is not None and f"share/{against}" in table.columns:
        columns["tracking_error"] = against
    else:
        against = None

    found = pd.DataFrame(index=dates)

    for label, book in columns.items():
        for part in (risk_store.CONTRIBUTION, risk_store.SHARE):
            found[f"{label}_{part}"] = mine[f"{part}/{book}"].reindex(dates).fillna(0.0)

    found.attrs = {"against": against}

    return found.rename_axis("DATE")


def liquidity(name: str,
              measured: LiquidityAnalytics,
              days: pd.DatetimeIndex,
              rebalances: list[pd.Timestamp]) -> pd.DataFrame:
    """At each of *rebalances* over *days* the run kept traded values on:
    the fund's holding of *name*, its average daily traded value and the
    days to sell the holding.

    Raises:
        Unavailable: If the run kept no traded values, or none in the
            window.
    """
    held_on = set(rebalances)
    dates = [day for day in liquidity_dates(measured)
             if day in held_on and days[0] <= day <= days[-1]]

    if not dates:
        raise Unavailable(TOO_SHORT)

    traded = measured.traded_values()
    rows = {}

    for day in dates:
        held = measured.days_to_sell(day)

        if name in held.index:
            rows[day] = held.loc[name, ["value", "traded_value", "days"]].tolist()
        else:
            value = traded.at[day, name] if name in traded.columns else None
            rows[day] = [0.0, value, 0.0]

    found = pd.DataFrame.from_dict(rows, orient="index",
                                   columns=["value", "traded_value", "days_to_sell"])

    return found.astype(float).rename_axis("DATE")


def liquidity_of(context: Any) -> LiquidityAnalytics:
    """The run's liquidity under the analysis's options, its traded values
    as `LiquidityAnalytics.traded_values` gives them; one for the context,
    so what it works out is shared."""
    # BN-481: one for the context rather than one a call, so a name's
    # profile does not pivot the traded values again.
    options = context.options.liquidity
    found: LiquidityAnalytics = context.kept(
        ("liquidity_of",), lambda: LiquidityAnalytics(context.run, context.source,
                                                      participation=options.participation,
                                                      lookback_days=options.lookback_days))

    return found


def liquidity_dates(measured: LiquidityAnalytics) -> list[pd.Timestamp]:
    """The dates the run kept traded values on, or `Unavailable`."""
    try:
        dates: list[pd.Timestamp] = measured.dates()
    except KeyError as error:
        code = RUN_PREDATES if TRADED_VALUES in str(error) else NO_DATA_STORE

        raise Unavailable(code) from error

    if not dates:
        raise Unavailable(TOO_SHORT)

    return dates
