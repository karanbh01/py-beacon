# src/beacon/analysis/performance/costs.py
"""
What a run cost: its costs by kind (the fixed cost and the market impact of
its trades, the cost of rolling futures) and the creation fees dealing
investors paid into it, in money, in basis points of its assets a year and
per unit it traded, over the run, each year and by why it traded.

Creation fees are received by the fund, so they are set against its costs:
the net cost is the costs less the fees.
"""
# BN-384. As agreed with Karan on 2026-10-27.
import pandas as pd

from ...backtest.breakdown import CREATION_FEES, FIXED, FUTURES_ROLLS, IMPACT

# The rows of a cost table: what the fund paid, what it received, the net.
PAID = (FIXED, IMPACT, FUTURES_ROLLS)
RECEIVED = "creation_fees_received"
NET = "net"
ROWS = (*PAID, RECEIVED, NET)

BASIS_POINTS = 10_000.0


def _kinds(costs: pd.DataFrame) -> pd.DataFrame:
    """The daily costs as the table's rows: paid, received and net."""
    found = costs[list(PAID)].astype(float).copy()
    found[RECEIVED] = costs[CREATION_FEES].astype(float)
    found[NET] = found[list(PAID)].sum(axis=1) - found[RECEIVED]

    return found[list(ROWS)]


def cost_summary(costs: pd.DataFrame,
                 nav: pd.Series,
                 periods_per_year: int) -> pd.DataFrame:
    """Each kind's total over the run and in basis points of the mean NAV a
    year, a row a kind."""
    totals = _kinds(costs).sum()
    years = len(nav) / float(periods_per_year)
    assets = float(nav.mean()) if len(nav) else 0.0
    scale = BASIS_POINTS / (assets * years) if assets > 0 and years > 0 else float("nan")

    return pd.DataFrame({"total": totals, "bps_a_year": totals * scale})


def yearly_costs(costs: pd.DataFrame,
                 nav: pd.Series) -> pd.DataFrame:
    """Each calendar year's costs by kind, and the net in basis points of the
    year's mean NAV (for the year's days alone, not annualised), a row a
    year."""
    kinds = _kinds(costs)
    years = pd.DatetimeIndex(kinds.index).year.astype(str)
    table = kinds.groupby(years).sum()
    assets = nav.groupby(pd.DatetimeIndex(nav.index).year.astype(str)).mean()
    table["net_bps"] = table[NET] / assets.reindex(table.index) * BASIS_POINTS
    table.index.name = "year"

    return table


def per_traded(trades: pd.DataFrame) -> dict[str, float | None]:
    """The fixed cost, the market impact and both, in basis points of the
    value traded; None without trading."""
    traded = float(trades["VALUE"].astype(float).sum())

    if traded <= 0:
        return {"fixed_bps": None, "impact_bps": None, "total_bps": None}

    cost = float(trades["COST"].astype(float).sum())
    impact = float(trades["IMPACT"].astype(float).sum())

    return {"fixed_bps": (cost - impact) / traded * BASIS_POINTS,
            "impact_bps": impact / traded * BASIS_POINTS,
            "total_bps": cost / traded * BASIS_POINTS}


def by_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """The trading costs by why the trade was made: the cost, its market
    impact, the value traded and the cost in basis points of it, a row a
    reason, largest cost first. They add up to the run's trading costs."""
    grouped = trades.groupby("REASON")
    table = pd.DataFrame({"cost": grouped["COST"].sum().astype(float),
                          "impact": grouped["IMPACT"].sum().astype(float),
                          "traded": grouped["VALUE"].sum().astype(float)})
    table["bps"] = (table["cost"] / table["traded"].where(table["traded"] > 0)
                    * BASIS_POINTS)
    table.index.name = "reason"

    return table.sort_values("cost", ascending=False)
