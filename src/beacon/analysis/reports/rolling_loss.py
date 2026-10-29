# src/beacon/analysis/reports/rolling_loss.py
"""
Value at risk and expected shortfall over each trailing window, every window
at once: the windows stacked as rows, each method worked out along them, as
`beacon.analysis.risk.var` works it out for one window.
"""
# BN-398. Window by window this took 16 s at 6,000 names over seven years,
# almost all of it in building each window's tables.
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ..performance import formulas
from ..risk.var import (
    ACTIVE,
    CORNISH_FISHER,
    ES_METHODS,
    FUND,
    HISTORICAL,
    NORMAL,
    STANDARD,
    VAR_METHODS,
    risk_of_loss,
)
from .context import Context

VAR, ES = "var", "es"


def rolling_loss(measure: str) -> Callable[[Context], pd.DataFrame]:
    """Each day's value at risk (``var``) or expected shortfall (``es``) over
    the window ending that day, by book, method, level and horizon."""
    def compute(context: Context) -> pd.DataFrame:
        tables = context.kept(("rolling_loss",), lambda: _both(context))

        return tables[measure]

    return compute


def _both(context: Context) -> dict[str, pd.DataFrame]:
    options = context.options.value_at_risk
    window = context.options.rolling.window
    loss = context.loss
    found: dict[str, dict[str, pd.Series]] = {VAR: {}, ES: {}}

    for book in loss.books():
        if book not in (FUND, ACTIVE):
            continue

        returns = loss.returns_of(book)

        for measure, figures in _of(returns, window, options.levels, options.horizons).items():
            found[measure].update({f"{book}/{name}": series for name, series in figures.items()})

    days = context.performance.returns.index

    return {measure: pd.DataFrame(columns).reindex(days) for measure, columns in found.items()}


def _of(returns: pd.Series,
        window: int,
        levels: tuple[float, ...],
        horizons: tuple[int, ...]) -> dict[str, dict[str, pd.Series]]:
    """Every method's figures over each window of *returns*."""
    values = returns.to_numpy(dtype=float)
    found: dict[str, dict[str, pd.Series]] = {VAR: {}, ES: {}}

    if len(values) < window:
        return found

    rows = sliding_window_view(values, window)
    ends = returns.index[window - 1:]
    gaps = np.isnan(rows).any(axis=1)
    mean = rows.mean(axis=1)
    spread = rows.std(axis=1, ddof=1)
    leaning, kurtosis = _shape(rows, spread)

    for level in levels:
        z = STANDARD.inv_cdf(1.0 - level)
        cut = np.quantile(rows, 1.0 - level, axis=1)
        tail = np.where(rows <= cut[:, None], rows, np.nan)
        moved = (z + (z * z - 1.0) * leaning / 6.0 + (z ** 3 - 3.0 * z) * kurtosis / 24.0
                 - (2.0 * z ** 3 - 5.0 * z) * leaning * leaning / 36.0)
        one_day = {(VAR, HISTORICAL): -cut, (VAR, NORMAL): -(mean + z * spread),
                   (VAR, CORNISH_FISHER): -(mean + moved * spread),
                   (ES, HISTORICAL): -np.nanmean(tail, axis=1),
                   (ES, NORMAL): -mean + spread * STANDARD.pdf(z) / (1.0 - level)}

        for (measure, method), daily in one_day.items():
            for horizon in horizons:
                found[measure][f"{method}/{level}/{horizon}"] = pd.Series(
                    daily * math.sqrt(horizon), index=ends)

    # A window with a gap is worked out on its own, as the window's figures
    # drop the gap's day.
    for position in np.flatnonzero(gaps):
        _fill(found, returns.iloc[position:position + window], ends[position], levels, horizons)

    return found


def _shape(rows: np.ndarray,
           spread: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row's bias-corrected sample skew and excess kurtosis, 0 for a
    row with no spread, as `beacon.analysis.performance.distribution` gives
    them."""
    count = rows.shape[1]
    centred = rows - rows.mean(axis=1, keepdims=True)
    second = np.mean(centred ** 2, axis=1)
    third = np.mean(centred ** 3, axis=1)
    fourth = np.mean(centred ** 4, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        leaning = math.sqrt(count * (count - 1)) / (count - 2) * third / second ** 1.5
        excess = fourth / second ** 2 - 3.0
        kurtosis = ((count + 1) * excess + 6.0) * (count - 1) / ((count - 2) * (count - 3))

    flat = ~(spread > formulas.NEGLIGIBLE_SPREAD)

    return np.where(flat, 0.0, leaning), np.where(flat, 0.0, kurtosis)


def _fill(found: dict[str, dict[str, pd.Series]],
          returns: pd.Series,
          end: pd.Timestamp,
          levels: tuple[float, ...],
          horizons: tuple[int, ...]) -> None:
    table = risk_of_loss(returns, levels, horizons)

    for measure, methods in ((VAR, VAR_METHODS), (ES, ES_METHODS)):
        for method in methods:
            for level in levels:
                for horizon in horizons:
                    found[measure][f"{method}/{level}/{horizon}"].loc[end] = table.loc[
                        (measure, method), (level, horizon)]
