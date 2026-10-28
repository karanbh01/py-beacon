# src/beacon/analysis/attribution/implementation.py
"""
The implementation breakdown: where a fund's tracking difference came from,
day by day from the run's own records, linked over the run.

Each day, with ``V`` the base the fund's return per unit is measured on (the
NAV at the close before, grown by any units dealt that day: today's units at
yesterday's NAV per unit), ``w`` and ``b`` the fund's and the index's
weights in the names at the close before (the fund's scaled onto ``V``, so
money dealt that day is held as cash), ``c = 1 - sum w`` the rest of the
fund (its cash) and ``r`` the names' total returns:

- **cap and screens**: ``sum g^R r``, where ``g^R = w^R - (1 - c^R) b^R`` is
  the gap between the fund's invested weights and the index's at the last
  rebalance: what the cap, the screens, the smallest position and rounding
  left the fund holding differently;
- **drift**: ``sum (g - g^R) r``, how that gap has moved since, as flows,
  dividends and the cash band traded the fund and not the index;
- **cash drag**: ``-c sum b r`` plus the interest earned over ``V``: the
  return the cash gave up against the index, less what it earned;
- **equitisation**: the futures' profit and loss before their costs, over
  ``V``, which wins back the cash drag;
- **fixed costs**, **market impact**, **futures rolls** and **futures
  trading**: each day's cost over ``V``;
- **fees**: the management fee charged over ``V``;
- **withholding tax**: the tax withheld from dividends over ``V``;
- **creation fees**: what dealing investors paid into the fund over ``V``;
- **dividend treatment**: ``sum b r`` less the index's own return: the
  dividends a price index leaves out, or a total return index's own timing.

They add up to the fund's return from its names, cash and futures, less its
costs and fees, less the index's; what else moved the NAV (a currency hedge,
trades dealt away from the close) is the residual.
"""
# BN-396. Agreed with Karan on 2026-10-28: an accounting from the run's
# records; switching features off is the tests' check.
import pandas as pd

from ...data.base import as_of_position
from ..data import RunData
from ..panels import CASH, COSTS, REFERENCE

CAP_AND_SCREENS = "cap_and_screens"
DRIFT = "drift"
CASH_DRAG = "cash_drag"
EQUITISATION = "equitisation"
FIXED_COSTS = "fixed_costs"
MARKET_IMPACT = "market_impact"
FUTURES_ROLLS = "futures_rolls"
FUTURES_TRADING = "futures_trading"
FEES = "fees"
WITHHOLDING_TAX = "withholding_tax"
CREATION_FEES = "creation_fees"
DIVIDEND_TREATMENT = "dividend_treatment"
PARTS = (CAP_AND_SCREENS, DRIFT, CASH_DRAG, EQUITISATION, FIXED_COSTS, MARKET_IMPACT,
         FUTURES_ROLLS, FUTURES_TRADING, FEES, WITHHOLDING_TAX, CREATION_FEES,
         DIVIDEND_TREATMENT)

# What each cost and cash part is read from, and its sign in the return.
COST_PARTS = {FIXED_COSTS: ("fixed", -1.0), MARKET_IMPACT: ("impact", -1.0),
              FUTURES_ROLLS: ("futures_rolls", -1.0), FUTURES_TRADING: ("futures_trading", -1.0),
              CREATION_FEES: ("creation_fees", 1.0)}
CASH_PARTS = {FEES: ("fee_charged", -1.0), WITHHOLDING_TAX: ("withholding_tax", -1.0)}


def daily_parts(run: RunData,
                fund: pd.DataFrame,
                index: pd.DataFrame,
                name_returns: pd.DataFrame,
                index_returns: pd.Series,
                days: pd.DatetimeIndex) -> pd.DataFrame:
    """Each day's parts of the tracking difference, by day and part: from the
    run's first day, whose close before is its starting capital all in cash,
    or else from the second of *days*."""
    w = fund.reindex(days, method="ffill").fillna(0.0)
    b = index.reindex(days, method="ffill").reindex(columns=w.columns.union(index.columns))
    names = list(w.columns.union(b.columns))
    w, b = w.reindex(columns=names, fill_value=0.0), b.reindex(columns=names).fillna(0.0)
    r = name_returns.reindex(index=days, columns=names).fillna(0.0)
    cash = 1.0 - w.sum(axis=1)
    gap = w - b.mul(1.0 - cash, axis=0)
    at_rebalance = _at_rebalances(run, gap, days)

    # The run's first day starts from its capital, held in cash, and its
    # costs of buying in are the first part of its tracking difference.
    first = bool(len(days)) and days[0] == run.nav.index[0]
    close_before = run.nav.reindex(days).shift(1)
    units = run.units.reindex(days)
    before = close_before * units / units.shift(1)

    if first:
        close_before.iloc[0] = before.iloc[0] = run.initial_capital

    # The weights at the close before, scaled onto the day's base: money
    # dealt that day joins as cash.
    scale = (close_before / before).fillna(1.0)
    lagged_w = w.shift(1).mul(scale, axis=0)
    lagged_b = b.shift(1)
    lagged_cash = 1.0 - lagged_w.sum(axis=1)
    lagged_gap = lagged_w - lagged_b.mul(1.0 - lagged_cash, axis=0)
    lagged_rebalance = at_rebalance.shift(1)
    index_holdings = (lagged_b * r).sum(axis=1)

    parts = pd.DataFrame(index=days)
    parts[CAP_AND_SCREENS] = (lagged_rebalance * r).sum(axis=1)
    parts[DRIFT] = ((lagged_gap - lagged_rebalance) * r).sum(axis=1)
    parts[CASH_DRAG] = -lagged_cash * index_holdings + _over(run, CASH, "interest", before)
    parts[EQUITISATION] = (_over(run, CASH, "futures_settled", before)
                           + _over(run, COSTS, "futures_rolls", before)
                           + _over(run, COSTS, "futures_trading", before))

    for part, (column, sign) in COST_PARTS.items():
        parts[part] = sign * _over(run, COSTS, column, before)

    for part, (column, sign) in CASH_PARTS.items():
        parts[part] = sign * _over(run, CASH, column, before)

    parts[DIVIDEND_TREATMENT] = index_holdings - index_returns.reindex(days)

    parts = parts[list(PARTS)].fillna(0.0)

    return parts if first else parts.iloc[1:]


def _at_rebalances(run: RunData,
                   gap: pd.DataFrame,
                   days: pd.DatetimeIndex) -> pd.DataFrame:
    """Each day's gap as it was at the close of the last rebalance on or
    before it."""
    reference = run.tables[REFERENCE]
    rebalances = pd.DatetimeIndex(sorted(reference.index.get_level_values(0).unique()))
    rows = []

    for day in days:
        at = as_of_position(rebalances, day)
        anchor = as_of_position(days, rebalances[at]) if at is not None else None
        rows.append(gap.iloc[anchor] if anchor is not None else gap.iloc[0] * 0.0)

    return pd.DataFrame(rows, index=days)


def _over(run: RunData,
          table: str,
          column: str,
          before: pd.Series) -> pd.Series:
    """A cash or cost table's column each day over the NAV at the close
    before; 0 where the run kept none."""
    found = run.tables.get(table)

    if found is None or column not in found.columns:
        return pd.Series(0.0, index=before.index)

    return (found[column].reindex(before.index).fillna(0.0) / before).fillna(0.0)
