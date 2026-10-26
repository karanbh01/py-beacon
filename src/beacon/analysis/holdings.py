# src/beacon/analysis/holdings.py
"""
Daily weights worked out again from what was held and the closes.

A book's daily weights repeat information its holdings and the prices
already carry: a weight is a holding's value over the day's total, and a
holding changes only when the book trades or the index rebalances. So a
stored run keeps the holdings on the days they change, and the weights are
worked out again from them and the closes, in the same arithmetic as the
code that first recorded them.

Wherever a weight worked out this way is not the recorded one to the last
bit, the recorded value is kept in a patch and put back, so the weights
always come back exactly. The fund's weights need no patch; an index's,
recorded at single precision from units that were themselves rounded to
single precision, need one for about a quarter of their cells.
"""
# BN-403. At 6,000 names over five years the fund's weights were 45 MB and
# the index's 26 MB; their holdings are under 1 MB and the index's patch 7.6.
import numpy as np
import pandas as pd

from ..index.result import daily_weights_frame

# The precision an index records its daily weights at.
RECORDED_PRECISION = daily_weights_frame([])["WEIGHT"].dtype

# The columns of the holdings and patch tables.
NAME = "NAME"
AMOUNT = "AMOUNT"
HELD = "HELD"
WEIGHT = "WEIGHT"


def wide_of(frame: pd.DataFrame,
            index: str,
            columns: str,
            values: str) -> pd.DataFrame:
    """A long *frame* as dates (its *index* column) by name (its *columns*
    column) of *values*, each pair's value as it is, missing where a pair has
    none; what a pivot gives for pairs that occur once, without its cost."""
    # Hashed, not sorted row by row: sorting 4.8 million names took seconds.
    day_codes, days = pd.factorize(pd.DatetimeIndex(frame[index]), sort=True)
    name_codes, names = pd.factorize(frame[columns].astype(str), sort=True)
    grid = np.full((len(days), len(names)), np.nan)
    grid[day_codes, name_codes] = frame[values].to_numpy(dtype=float)

    return pd.DataFrame(grid, index=pd.DatetimeIndex(days, name=index),
                        columns=pd.Index(names, name=columns))


def holding_changes(amounts: pd.DataFrame) -> pd.DataFrame:
    """The days each name's holding changed, from *amounts* (dates by name,
    missing where not held).

    Returns:
        pd.DataFrame: Indexed by date, with ``NAME``, ``AMOUNT`` (missing on
        the day a holding ends) and ``HELD``, a row for each name on each day
        its holding began, changed or ended.
    """
    grid = amounts.to_numpy(dtype=float)
    held = ~np.isnan(grid)
    before = np.vstack([np.full((1, grid.shape[1]), np.nan), grid[:-1]])
    was_held = ~np.isnan(before)
    changed = (held != was_held) | (held & (grid != before))
    rows, columns = np.nonzero(changed)

    return pd.DataFrame({NAME: np.asarray(amounts.columns.astype(str))[columns],
                         AMOUNT: grid[rows, columns],
                         HELD: held[rows, columns]},
                        index=amounts.index[rows])


def held_amounts(changes: pd.DataFrame,
                 dates: pd.DatetimeIndex,
                 names: list[str]) -> pd.DataFrame:
    """*changes* as amounts held on every one of *dates*, by name in *names*,
    missing where a name was not held."""
    if changes.empty:
        return pd.DataFrame(np.nan, index=dates, columns=names)

    keyed = changes.rename_axis("DAY").reset_index()
    amounts = keyed.pivot(index="DAY", columns=NAME, values=AMOUNT)
    held = keyed.pivot(index="DAY", columns=NAME, values=HELD).astype(float)
    amounts = amounts.reindex(index=dates, columns=names).ffill()
    held = held.reindex(index=dates, columns=names).ffill()

    return amounts.where(held == 1.0)


def fund_weights_from(changes: pd.DataFrame,
                      totals: pd.Series,
                      prices: pd.DataFrame,
                      dates: pd.DatetimeIndex,
                      names: list[str]) -> pd.DataFrame:
    """The fund's daily weights: each holding's quantity times its price in
    the book's currency, over the day's total, as the books recorded them.

    Args:
        changes: The fund's quantities on the days they changed.
        totals: The books' total value (holdings and cash) on each day.
        prices: Each name's close in the book's currency, carried over gaps.
        dates: The days the books held anything.
        names: The names, in the recorded weights' order.
    """
    quantities = held_amounts(changes, dates, names)
    values = quantities * prices.reindex(index=dates, columns=names)

    frame = values.div(totals.reindex(dates), axis=0)
    frame.index.name, frame.columns.name = "DATE", "ASSET_ID"

    return frame


def index_weights_from(changes: pd.DataFrame,
                       closes: pd.DataFrame,
                       rates: pd.DataFrame,
                       dates: pd.DatetimeIndex,
                       names: list[str],
                       dtype: str) -> pd.DataFrame:
    """An index's daily weights: each holding's units times its close times
    its rate, over the day's total, in the index's own order of operations
    and the precision it records them at, then in *dtype*.

    Args:
        changes: The index's units on the days they changed.
        closes: Each name's close in its own currency, carried over gaps.
        rates: Each name's rate into the book's currency, by date and name.
        dates: The days the index held anything.
        names: The names, in the recorded weights' order.
        dtype: The weights table's dtype.
    """
    units = held_amounts(changes, dates, names).to_numpy()
    prices = closes.reindex(index=dates, columns=names).to_numpy(dtype=float)
    values = (units * prices) * rates.reindex(index=dates, columns=names).to_numpy(dtype=float)
    held = ~np.isnan(values)
    totals = np.where(held.any(axis=1), np.nansum(values, axis=1), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        shares = values / totals[:, None]

    # Recorded at the daily panel's precision, then read into the table; in
    # whole arrays, since a column at a time took most of a second.
    grid = shares.astype(RECORDED_PRECISION).astype(dtype)
    frame = pd.DataFrame(grid, index=pd.DatetimeIndex(dates, name="DATE"),
                         columns=pd.Index(names, name="IDENTIFIER"))

    return frame


def rate_grid(currencies: pd.DataFrame,
              rates: pd.DataFrame,
              dates: pd.DatetimeIndex,
              names: list[str]) -> pd.DataFrame:
    """Each name's rate into the book's currency on each of *dates*: its
    currency's kept rate, or 1 for a name in the book's own currency."""
    grid = pd.DataFrame(1.0, index=dates, columns=names)
    quoted = currencies["CURRENCY"]

    for code in rates.columns:
        held = [name for name in names if quoted.get(name) == code]

        if held:
            column = rates[code].reindex(dates).to_numpy()
            grid[held] = np.repeat(column[:, None], len(held), axis=1)

    return grid


def patch_of(rebuilt: pd.DataFrame,
             recorded: pd.DataFrame) -> pd.DataFrame:
    """Every cell where *rebuilt* is not *recorded* to the last bit, with the
    recorded value: indexed by date, with ``NAME`` and ``WEIGHT``."""
    ours = rebuilt.reindex(index=recorded.index, columns=recorded.columns).to_numpy()
    theirs = recorded.to_numpy()
    both_blank = np.isnan(ours.astype(float)) & np.isnan(theirs.astype(float))
    rows, columns = np.nonzero(~((ours == theirs) | both_blank))

    return pd.DataFrame({NAME: np.asarray(recorded.columns.astype(str))[columns],
                         WEIGHT: theirs[rows, columns]},
                        index=recorded.index[rows])


def patched(rebuilt: pd.DataFrame,
            patch: pd.DataFrame) -> pd.DataFrame:
    """*rebuilt* with *patch*'s recorded values put back."""
    if patch.empty:
        return rebuilt

    frame = rebuilt.copy()
    rows = frame.index.get_indexer(pd.DatetimeIndex(patch.index))
    columns = frame.columns.get_indexer(patch[NAME].astype(str))
    grid = frame.to_numpy(copy=True)
    grid[rows, columns] = patch[WEIGHT].to_numpy(dtype=grid.dtype)

    return pd.DataFrame(grid, index=frame.index, columns=frame.columns)
