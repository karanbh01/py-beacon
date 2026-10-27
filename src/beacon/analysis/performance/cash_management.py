# src/beacon/analysis/performance/cash_management.py
"""
How a fund managed its cash: each time the cash left its band and was raised
or spent back down to its target, how often and when that happened, what it
cost and how close it brought the cash to its target; and every trade made to
raise cash or spend it, whatever it was raised or spent for.

An event's gap is how far the cash stood from its target; the share of the gap
closed is one less the gap after over the gap before, so 1 is a cash brought
exactly to its target.
"""
# BN-407. As agreed with Karan on 2026-10-27.
import numpy as np
import pandas as pd

from ...backtest.breakdown import TAG_SEPARATOR
from ...backtest.cash_band import RAISE, SPEND_DOWN
from ...portfolio.reasons import CASH_RAISE, CASH_SPEND_DOWN

BASIS_POINTS = 10_000.0

# The columns of a recorded event.
EVENT_COLUMNS = ("kind", "cash_before", "cash_after", "target", "lower", "upper",
                 "traded", "cost", "trades")


def band_events(events: pd.DataFrame) -> pd.DataFrame:
    """Each event as the run recorded it, with the gap to the target before
    and after, the share of it closed, and the cost in basis points of the
    value traded."""
    table = events.reindex(columns=list(EVENT_COLUMNS)).copy()
    table["gap_before"] = (table["cash_before"] - table["target"]).abs()
    table["gap_after"] = (table["cash_after"] - table["target"]).abs()
    table["gap_closed"] = 1.0 - table["gap_after"] / table["gap_before"].where(
        table["gap_before"] > 0)
    table["cost_bps"] = table["cost"] / table["traded"].where(table["traded"] > 0) * BASIS_POINTS

    return table


def band_summary(events: pd.DataFrame) -> pd.DataFrame:
    """For raises, spend-downs and both: how many, the value traded, the
    total and mean cost, the cost in basis points of the value traded, and
    the mean share of the gap closed and gap left, a row each."""
    table = band_events(events)
    rows = {}

    for name, kinds in ((RAISE, (RAISE,)), (SPEND_DOWN, (SPEND_DOWN,)),
                        ("all", (RAISE, SPEND_DOWN))):
        some = table[table["kind"].isin(kinds)]
        traded, cost = float(some["traded"].sum()), float(some["cost"].sum())
        rows[name] = {"events": len(some), "traded": traded, "cost": cost,
                      "mean_cost": cost / len(some) if len(some) else np.nan,
                      "cost_bps": cost / traded * BASIS_POINTS if traded > 0 else np.nan,
                      "mean_gap_closed": float(some["gap_closed"].mean()) if len(some) else np.nan,
                      "mean_gap_after": float(some["gap_after"].mean()) if len(some) else np.nan}

    return pd.DataFrame.from_dict(rows, orient="index")


def band_calendar(events: pd.DataFrame,
                  frequency: str = "M") -> pd.DataFrame:
    """How many raises and spend-downs each month (``M``) or year (``Y``)
    that had one, and what they cost."""
    if events.empty:
        return pd.DataFrame(columns=[RAISE, SPEND_DOWN, "cost"],
                            index=pd.Index([], name="period"))

    periods = pd.DatetimeIndex(events.index).to_period(frequency).astype(str)
    counts = pd.crosstab(periods, events["kind"]).reindex(columns=[RAISE, SPEND_DOWN],
                                                          fill_value=0)
    counts["cost"] = events["cost"].astype(float).groupby(periods).sum()
    counts.index.name = "period"
    counts.columns.name = None

    return counts


def cash_trading(trades: pd.DataFrame) -> pd.DataFrame:
    """Every trade made to raise cash or spend it, by the tag and the reason
    it was made for (a redemption, a distribution, the band...): how many,
    the value traded and its cost."""
    tags = trades["TAGS"].fillna("").astype(str).str.split(TAG_SEPARATOR)
    rows = []

    for tag in (CASH_RAISE, CASH_SPEND_DOWN):
        some = trades[tags.map(lambda found, tag=tag: tag in found)]

        for reason, made in some.groupby("REASON"):
            rows.append({"tag": tag, "reason": reason, "trades": len(made),
                         "traded": float(made["VALUE"].sum()),
                         "cost": float(made["COST"].sum())})

    table = pd.DataFrame(rows, columns=["tag", "reason", "trades", "traded", "cost"])

    return table.set_index(["tag", "reason"])
