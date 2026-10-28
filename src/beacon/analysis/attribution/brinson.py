# src/beacon/analysis/attribution/brinson.py
"""
Brinson-Fachler attribution: how much of the fund's return over a comparator
came from how it weighted each group (allocation), from what it held within
each (selection), and from the two together (interaction).

Each day, with w and b the fund's and the comparator's weights at the close
before, R and R^b their returns, by group g and in all:

    allocation_g  = (w_g - b_g) (R^b_g - R^b)
    selection_g   = b_g (R_g - R^b_g)
    interaction_g = (w_g - b_g) (R_g - R^b_g)

They add up, over the groups, to the holdings' active return that day. A
group the fund does not hold takes the comparator's group return as its own,
so it has no selection or interaction; one the comparator does not hold
takes the comparator's whole return, so its bet is all interaction. Cash is
a group of its own: the fund's weight in it is one less its names', and its
return the interest it earned over the cash it held.

The days are linked over the window by Carino's factors for an active return,
``(ln(1 + R_t) - ln(1 + R^b_t)) / (R_t - R^b_t)`` over the same for the
window, so the effects add up to the holdings' compounded active return. What
the holdings do not explain of the books' own returns (costs, fees, dividends
not yet paid, a price index's missing dividends) is the residual.
"""
# BN-394. Agreed with Karan on 2026-10-28: daily, Carino-linked; interaction
# and cash on lines of their own.
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from ..data import RunData
from ..panels import CASH as CASH_TABLE
from ..performance.comparisons import aligned
from .groups import by_group, memberships

CASH = "Cash"
EFFECTS = ("allocation", "selection", "interaction")

# Below this, two returns are treated as equal for the linking factor.
NEGLIGIBLE = 1e-12


@dataclass(frozen=True)
class BrinsonResult:
    """The fund's return over a comparator, by group and effect.

    Attributes:
        start: The first day attributed.
        end: The last.
        periods: The days.
        fund_return: The fund's own compounded return.
        comparator_return: The comparator's.
        active_return: The fund's less the comparator's.
        explained: The holdings' active return the effects add up to.
        residual: The active return less what the effects explain.
        groups: By group: the mean weights and compounded returns of the
            fund's and the comparator's holdings, and each effect, linked.
    """
    start: str
    end: str
    periods: int
    fund_return: float
    comparator_return: float
    active_return: float
    explained: float
    residual: float
    groups: pd.DataFrame

    @property
    def totals(self) -> pd.Series:
        """Each effect over all the groups."""
        return self.groups[list(EFFECTS)].sum()

    def reconciles(self,
                   tolerance: float = 1e-9) -> bool:
        """Whether the effects account for the active return."""
        return abs(self.residual) <= tolerance


def daily_effects(fund: pd.DataFrame,
                  comparator: pd.DataFrame,
                  fund_earned: pd.DataFrame,
                  comparator_earned: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Each day's effects by group, from each book's group weights at the
    close before and what each group earned (weight times return)."""
    groups = sorted(set(fund.columns) | set(comparator.columns))
    w = fund.reindex(columns=groups, fill_value=0.0)
    b = comparator.reindex(columns=groups, fill_value=0.0)
    ours = fund_earned.reindex(index=w.index, columns=groups, fill_value=0.0)
    theirs = comparator_earned.reindex(index=w.index, columns=groups, fill_value=0.0)
    total = theirs.sum(axis=1)
    benchmark = (theirs / b.where(b != 0.0)).apply(lambda column: column.fillna(total))
    own = (ours / w.where(w != 0.0)).fillna(benchmark)
    gap = w - b

    return {"allocation": gap.mul(benchmark.sub(total, axis=0)),
            "selection": b * (own - benchmark),
            "interaction": gap * (own - benchmark)}


def active_carino(fund: float,
                  comparator: float) -> float:
    """The linking factor for an active return: ``(ln(1 + R) - ln(1 + R^b))
    / (R - R^b)``, ``1 / (1 + R)`` where the two are equal."""
    if abs(fund - comparator) < NEGLIGIBLE:
        return 1.0 / (1.0 + fund)

    return (math.log1p(fund) - math.log1p(comparator)) / (fund - comparator)


def linked(effects: dict[str, pd.DataFrame],
           fund: pd.Series,
           comparator: pd.Series) -> dict[str, pd.Series]:
    """Each effect by group over the days, scaled by Carino's factors so they
    add up to the compounded active return of *fund* over *comparator*."""
    whole = active_carino(float((1.0 + fund).prod() - 1.0),
                          float((1.0 + comparator).prod() - 1.0))
    factors = pd.Series([active_carino(float(ours), float(theirs))
                         for ours, theirs in zip(fund, comparator, strict=True)],
                        index=fund.index) / whole

    return {name: table.mul(factors, axis=0).sum() for name, table in effects.items()}


def brinson(run: RunData,
            weights: pd.DataFrame,
            comparator_weights: pd.DataFrame,
            comparator_levels: pd.Series,
            name_returns: pd.DataFrame,
            field: str,
            days: pd.DatetimeIndex) -> BrinsonResult:
    """The fund's Brinson-Fachler attribution against a comparator, by
    *field*, over *days* after the first, whose close the first day's weights
    are held at."""
    fund_lagged = weights.reindex(days, method="ffill").shift(1).fillna(0.0)
    comparator_lagged = (comparator_weights.reindex(days, method="ffill").shift(1)
                         .fillna(0.0))
    returns = name_returns.reindex(days)
    groups = memberships(run, field)

    fund_held = by_group(fund_lagged, groups)
    fund_earned = by_group((fund_lagged * returns.reindex(columns=fund_lagged.columns))
                           .fillna(0.0), groups)
    comparator_held = by_group(comparator_lagged, groups)
    comparator_earned = by_group((comparator_lagged
                                  * returns.reindex(columns=comparator_lagged.columns))
                                 .fillna(0.0), groups)

    fund_held[CASH] = 1.0 - fund_held.sum(axis=1)
    fund_earned[CASH] = fund_held[CASH] * _cash_return(run, days)
    comparator_held[CASH] = 1.0 - comparator_held.sum(axis=1)
    comparator_earned[CASH] = comparator_held[CASH] * _cash_return(run, days)

    held_fund = fund_earned.sum(axis=1).iloc[1:]
    held_comparator = comparator_earned.sum(axis=1).iloc[1:]
    effects = {name: table.iloc[1:] for name, table in
               daily_effects(fund_held, comparator_held, fund_earned, comparator_earned).items()}
    totals = linked(effects, held_fund, held_comparator)

    ours, theirs = aligned(run.returns, comparator_levels)
    ours, theirs = ours.reindex(held_fund.index), theirs.reindex(held_fund.index)
    fund_return = float((1.0 + ours.fillna(0.0)).prod() - 1.0)
    comparator_return = float((1.0 + theirs.fillna(0.0)).prod() - 1.0)
    explained = float(sum(series.sum() for series in totals.values()))

    table = pd.DataFrame({"fund_weight": fund_held.iloc[1:].mean(),
                          "comparator_weight": comparator_held.iloc[1:].mean(),
                          "fund_return": _compounded(fund_earned.iloc[1:], fund_held.iloc[1:]),
                          "comparator_return": _compounded(comparator_earned.iloc[1:],
                                                           comparator_held.iloc[1:]),
                          **totals}).fillna(0.0)
    table["total"] = table[list(EFFECTS)].sum(axis=1)
    table.index.name = "group"
    # A group neither book held adds nothing; cash is always shown.
    held = (table["fund_weight"] != 0.0) | (table["comparator_weight"] != 0.0)
    table = table[held | (table.index == CASH)]

    return BrinsonResult(start=held_fund.index[0].isoformat(),
                         end=held_fund.index[-1].isoformat(), periods=len(held_fund),
                         fund_return=fund_return, comparator_return=comparator_return,
                         active_return=fund_return - comparator_return, explained=explained,
                         residual=fund_return - comparator_return - explained,
                         groups=table.sort_values("total", ascending=False))


def _cash_return(run: RunData,
                 days: pd.DatetimeIndex) -> pd.Series:
    """Each day's return on cash: the interest earned over the cash held at
    the close before; 0 where the run kept no cash table or held none."""
    cash = run.tables.get(CASH_TABLE)

    if cash is None or "interest" not in cash or "cash" not in cash:
        return pd.Series(0.0, index=days)

    balance = cash["cash"].reindex(days).shift(1)
    earned = cash["interest"].reindex(days).fillna(0.0)

    return (earned / balance.where(balance > 0.0)).fillna(0.0)


def _compounded(earned: pd.DataFrame,
                held: pd.DataFrame) -> pd.Series:
    """Each group's compounded return over the days, from what it earned
    over what it held each day."""
    daily = (earned / held.where(held != 0.0)).fillna(0.0)

    return (1.0 + daily).prod() - 1.0


def by_period(days: pd.DatetimeIndex,
              compute: Callable[[pd.DatetimeIndex], BrinsonResult],
              frequency: str = "Y") -> pd.DataFrame:
    """The effects over each year (``Y``) or month (``M``), each period's
    days linked within it, from the close before its first: a row a period,
    the effects and the residual."""
    periods = pd.DatetimeIndex(days).to_period(frequency)
    rows = {}

    for period in periods.unique():
        inside = [at for at, found in enumerate(periods == period) if found]
        first = max(inside[0] - 1, 0)
        window = days[first:inside[-1] + 1]

        if len(window) < 2:
            continue

        found = compute(window)
        rows[str(period)] = {**found.totals.to_dict(), "explained": found.explained,
                             "active_return": found.active_return, "residual": found.residual}

    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("period")
