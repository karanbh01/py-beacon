# src/beacon/analysis/trade_packing.py
"""
A run's trades, kept small: what was traded, and the rest worked out again.

A trade's price is its name's close in the book's currency that day, its
value is quantity times price, and its cost is the run's fixed rate on that
value, with any market impact. So a stored run keeps each trade's date,
name, side, quantity, reason and tags, and the fixed rate; the price, value,
cost and impact are worked out again from the closes it keeps, in the
arithmetic that first recorded them.

Wherever a value worked out this way is not the recorded one to the last
bit, the recorded value is kept in a patch and put back, so the trades come
back exactly. A column the patch would hold for most trades, such as the
costs of a run that modelled market impact, is kept whole instead.
"""
# BN-412. At 6,000 names over seven years the trades were 4.8 MB of a run's
# tables; price, value and cost were three quarters of it.
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..data.prices import converted
from ..portfolio.reasons import CORPORATE_ACTION, DIVIDEND_REINVESTMENT
from .packing import StoredSource

# The tables a run's packed trades are kept in.
TRADES_KEPT = "trades_kept"
TRADES_PATCH = "trades_patch"
PACKED_TRADES = (TRADES_KEPT, TRADES_PATCH)

# The trade columns worked out again, in the order each is worked out: each
# from those before it, with their recorded values put back.
PRICE, VALUE, COST, IMPACT = "PRICE", "VALUE", "COST", "IMPACT"
WORKED_OUT = (PRICE, VALUE, IMPACT, COST)

# The columns of the patch: the trade's position, the column and its value.
ROW, COLUMN, RECORDED = "ROW", "COLUMN", "RECORDED"

BASIS_POINTS = 10_000.0

# The fixed rate is read off the trades to this many decimals of a basis
# point; a rate that is not exactly this is put right by the patch.
RATE_DECIMALS = 6


def fixed_rate(trades: pd.DataFrame) -> float:
    """The fixed rate, in basis points, the run's rebalance trades paid: the
    middle of their cost over value, among those with no market impact."""
    plain = (trades["REASON"] != CORPORATE_ACTION) & (trades[IMPACT] == 0.0) & (trades[VALUE] > 0.0)

    if not plain.any():
        return 0.0

    rates = trades.loc[plain, COST] / trades.loc[plain, VALUE] * BASIS_POINTS

    return round(float(rates.median()), RATE_DECIMALS)


def pack_trades(trades: pd.DataFrame,
                tables: Mapping[str, pd.DataFrame],
                currency: str) -> tuple[dict[str, pd.DataFrame], float] | None:
    """*trades* as the tables that keep them, and the fixed rate; None when
    they would not come back exactly, so they are kept as they are.

    Args:
        trades: The run's trades (`beacon.backtest.breakdown.trades`).
        tables: The run's closes, currencies, rates and actions.
        currency: The book's currency.
    """
    rate = fixed_rate(trades)
    prices = book_prices(trades, tables, currency)
    whole, rows, columns, values = [], [], [], []
    frame = trades.copy()

    for column in WORKED_OUT:
        rebuilt = _worked_out(column, frame, prices, rate)
        differ = np.flatnonzero(~_same(rebuilt, frame[column].to_numpy(dtype=float)))

        # A column worked out wrong for most trades is kept whole.
        if len(differ) * 2 > len(frame):
            whole.append(column)
            continue

        rows.extend(differ.tolist())
        columns.extend([column] * len(differ))
        values.extend(frame[column].to_numpy(dtype=float)[differ].tolist())

    kept = trades[[name for name in trades.columns if name not in WORKED_OUT or name in whole]]
    positions = np.asarray(rows, dtype=np.int64)
    order = np.argsort(positions, kind="stable")
    patch = pd.DataFrame({ROW: positions[order],
                          COLUMN: np.asarray(columns, dtype=object)[order],
                          RECORDED: np.asarray(values, dtype=float)[order]},
                         index=pd.DatetimeIndex(trades.index[positions[order]],
                                                name=trades.index.name))
    packed = {TRADES_KEPT: kept, TRADES_PATCH: patch}

    # Checked as it is written: trades that would not come back exactly are
    # kept as they are.
    back = unpack_trades(packed, tables, currency, rate, list(trades.columns))

    return (packed, rate) if same_trades(back, trades) else None


def unpack_trades(packed: Mapping[str, pd.DataFrame],
                  tables: Mapping[str, pd.DataFrame],
                  currency: str,
                  rate: float,
                  order: list[str]) -> pd.DataFrame:
    """The trades `pack_trades` kept, exactly as they were, in the columns of
    *order*."""
    kept, patch = packed[TRADES_KEPT], packed[TRADES_PATCH]
    frame = kept.copy()
    prices = book_prices(frame, tables, currency)

    for column in WORKED_OUT:
        if column in kept.columns:
            continue

        values = _worked_out(column, frame, prices, rate)
        fixes = patch[patch[COLUMN] == column]
        values[fixes[ROW].to_numpy(dtype=np.int64)] = fixes[RECORDED].to_numpy(dtype=float)
        frame[column] = values

    return frame[order]


def book_prices(trades: pd.DataFrame,
                tables: Mapping[str, pd.DataFrame],
                currency: str) -> np.ndarray:
    """Each trade's name's close on its day, carried over gaps, in the book's
    currency, as the books priced it; missing where the run kept none."""
    if trades.empty:
        return np.array([], dtype=float)

    days = pd.DatetimeIndex(trades.index).unique().sort_values()
    stored = StoredSource(tables["currencies"], tables["rates"], tables["actions"], currency)
    prices = converted(stored, tables["closes"].ffill().reindex(days), currency)
    rows = prices.index.get_indexer(pd.DatetimeIndex(trades.index))
    columns = prices.columns.get_indexer(trades["IDENTIFIER"].astype(str))
    found = (rows >= 0) & (columns >= 0)
    grid = prices.to_numpy(dtype=float)

    return np.where(found, grid[rows, np.maximum(columns, 0)], np.nan)


def same_trades(ours: pd.DataFrame,
                theirs: pd.DataFrame) -> bool:
    """Whether two trade tables are the same: index, columns and dtypes, text
    equal and every number to the last bit."""
    if (list(ours.columns) != list(theirs.columns) or not ours.index.equals(theirs.index)
            or ours.index.name != theirs.index.name or not ours.dtypes.equals(theirs.dtypes)):
        return False

    for column in theirs.columns:
        if pd.api.types.is_float_dtype(theirs[column].dtype):
            if not _same(ours[column].to_numpy(), theirs[column].to_numpy()).all():
                return False
        elif not ours[column].equals(theirs[column]):
            return False

    return True


def _worked_out(column: str,
                frame: pd.DataFrame,
                prices: np.ndarray,
                rate: float) -> np.ndarray:
    """*column* worked out from the columns before it, as recorded."""
    if column == PRICE:
        return prices.copy()

    if column == VALUE:
        quantity = frame["QUANTITY"].to_numpy(dtype=float)

        return np.asarray(quantity * frame[PRICE].to_numpy(dtype=float), dtype=float)

    if column == IMPACT:
        return np.zeros(len(frame))

    # The fixed rate in each kind of trade's own order of operations: a
    # rebalance's notional times the basis points over 10,000, a dividend
    # reinvested at the rate as a fraction; a corporate action is booked at
    # no cost.
    value = frame[VALUE].to_numpy(dtype=float)
    reason = frame["REASON"].to_numpy(dtype=object)
    fixed = np.where(reason == DIVIDEND_REINVESTMENT, value * (rate / BASIS_POINTS),
                     value * rate / BASIS_POINTS)

    return np.where(reason == CORPORATE_ACTION, 0.0,
                    fixed + frame[IMPACT].to_numpy(dtype=float))


def _same(ours: np.ndarray,
          theirs: np.ndarray) -> np.ndarray:
    """Where two float arrays hold the same value to the last bit, or are
    both missing."""
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)

    return np.asarray((ours.view(np.int64) == theirs.view(np.int64))
                      | (np.isnan(ours) & np.isnan(theirs)), dtype=bool)
