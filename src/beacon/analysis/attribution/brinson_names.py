# src/beacon/analysis/attribution/brinson_names.py
"""
Brinson-Fachler selection and interaction split across each group's names,
so a group's row opens into the names that make it up.

Each day, with W_g and B_g the fund's and the comparator's weights in group
g at the close before, w_i and b_i a name's, r_i its return and R^b_g the
comparator's group return:

    d_i           = (w_i / W_g - b_i / B_g) (r_i - R^b_g)
    selection_i   = B_g d_i
    interaction_i = (W_g - B_g) d_i

The d_i of a group add up to R_g - R^b_g, so its names' selections and
interactions add up to its own. A name the comparator holds and the fund does
not has a negative share, so it shows what not holding it cost. A group the
fund does not hold takes the comparator's return, so its names have none; in
one the comparator does not hold, a name's share is its share of the fund's,
measured against the comparator's whole return, as the group's is. The days
are linked with the group effects' Carino factors. A name that changed group
has a row in each.
"""
# BN-415. Agreed with Karan on 2026-10-30: by active weight within the group,
# so the comparator's names are in it too.
import numpy as np
import pandas as pd

from ...data.base import as_of_position
from .groups import UNKNOWN

GROUP, NAME = "group", "name"


def name_effects(fund_lagged: pd.DataFrame,
                 comparator_lagged: pd.DataFrame,
                 returns: pd.DataFrame,
                 groups: dict[pd.Timestamp, pd.Series],
                 fund_held: pd.DataFrame,
                 comparator_held: pd.DataFrame,
                 comparator_earned: pd.DataFrame,
                 factors: pd.Series) -> pd.DataFrame:
    """Each name's linked selection and interaction within each group it was
    in, and its mean weights in each book, over the days of *factors* (the
    days' Carino factors over the window's).

    Args:
        fund_lagged: The fund's weights at the close before each day.
        comparator_lagged: The comparator's.
        returns: Each name's return each day.
        groups: Each rebalance's group of every name.
        fund_held: The fund's weight in each group each day.
        comparator_held: The comparator's.
        comparator_earned: What each group earned the comparator each day.
        factors: The linking factor of each day attributed.

    Returns:
        pd.DataFrame: Indexed by group and name, with ``fund_weight``,
        ``comparator_weight``, ``selection``, ``interaction`` and ``total``.
    """
    days = pd.DatetimeIndex(factors.index)
    names = sorted(set(fund_lagged.columns) | set(comparator_lagged.columns))
    w = fund_lagged.reindex(index=days, columns=names).fillna(0.0).to_numpy(dtype=float)
    b = comparator_lagged.reindex(index=days, columns=names).fillna(0.0).to_numpy(dtype=float)
    r = returns.reindex(index=days, columns=names).fillna(0.0).to_numpy(dtype=float)
    whole = comparator_earned.reindex(days).sum(axis=1).to_numpy(dtype=float)
    codes, labels = _labels(days, names, groups)
    linking = factors.to_numpy(dtype=float)[:, None]
    totals: dict[tuple[str, str], dict[str, float]] = {}

    for code, group in enumerate(labels):
        # Only the names ever in the group, so the arrays stay small.
        members = np.flatnonzero((codes == code).any(axis=0))

        if not len(members):
            continue

        inside = codes[:, members] == code
        held = _column(fund_held, group, days)
        theirs = _column(comparator_held, group, days)
        earned = _column(comparator_earned, group, days)
        benchmark = np.where(theirs != 0.0, earned / np.where(theirs != 0.0, theirs, 1.0), whole)
        comparator_share = np.where(theirs[:, None] != 0.0,
                                    b[:, members] / np.where(theirs != 0.0, theirs, 1.0)[:, None],
                                    0.0)
        fund_share = np.where(held[:, None] != 0.0,
                              w[:, members] / np.where(held != 0.0, held, 1.0)[:, None],
                              comparator_share)
        gap = np.where(inside, (fund_share - comparator_share)
                       * (r[:, members] - benchmark[:, None]), 0.0)
        selection = (theirs[:, None] * gap * linking).sum(axis=0)
        interaction = ((held - theirs)[:, None] * gap * linking).sum(axis=0)

        for place, position in enumerate(members):
            days_in = inside[:, place]
            totals[(str(group), names[position])] = {
                "fund_weight": float(w[days_in, position].mean()),
                "comparator_weight": float(b[days_in, position].mean()),
                "selection": float(selection[place]),
                "interaction": float(interaction[place])}

    table = pd.DataFrame.from_dict(totals, orient="index")

    if table.empty:
        return pd.DataFrame(columns=["fund_weight", "comparator_weight", "selection",
                                     "interaction", "total"],
                            index=pd.MultiIndex.from_tuples([], names=[GROUP, NAME]))

    table.index = pd.MultiIndex.from_tuples(table.index, names=[GROUP, NAME])
    table["total"] = table["selection"] + table["interaction"]
    held_any = (table["fund_weight"] != 0.0) | (table["comparator_weight"] != 0.0)

    return table[held_any]


def _labels(days: pd.DatetimeIndex,
            names: list[str],
            groups: dict[pd.Timestamp, pd.Series]) -> tuple[np.ndarray, list[str]]:
    """Each name's group each day, as `by_group` reads it (the last
    rebalance's before the day, or the first's for a day at or before it),
    as a code into the groups' names, with those names."""
    rebalances = pd.DatetimeIndex(sorted(groups))
    labels: dict[str, int] = {UNKNOWN: 0}
    codes = np.zeros((len(days), len(names)), dtype=np.int32)
    known: dict[pd.Timestamp, np.ndarray] = {}

    for row, day in enumerate(days):
        at = as_of_position(rebalances, day - pd.Timedelta(days=1))
        key = rebalances[at] if at is not None else rebalances[0] if len(rebalances) else None

        if key is None:
            continue

        if key not in known:
            found = groups[key].reindex(names).fillna(UNKNOWN).astype(str)
            known[key] = np.array([labels.setdefault(group, len(labels)) for group in found],
                                  dtype=np.int32)

        codes[row] = known[key]

    return codes, list(labels)


def _column(table: pd.DataFrame,
            group: str,
            days: pd.DatetimeIndex) -> np.ndarray:
    """A group's column of *table* over *days*, 0 where it has none."""
    if group not in table.columns:
        return np.zeros(len(days))

    return np.asarray(table[group].reindex(days).fillna(0.0), dtype=float)
