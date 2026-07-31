# src/beacon/plot/comparison.py
"""
Comparing several results on one axis.

A free function rather than an accessor because it is not about any one result:
`compare(a, b, c)` is a statement about the set, and hanging it off the first
argument would make that arbitrary.

The module is `comparison` rather than `compare` so it does not shadow the
function it exports. `from beacon.plot import compare` resolves a submodule
before it consults the package's lazy attribute hook, so a module of the same
name would hand back the module and every call site would fail on a
not-callable error.

## Aligned, not concatenated

Every series is clipped to the window they all share and rebased to 100 on the
first shared date. Two indices with different histories compared over different
periods differ for no reason but their spans, and the one with the shorter
history looks better or worse than it is. Rebasing on the shared start means
the lines begin together and the comparison is of shape.
"""
import logging
from typing import Any

import pandas as pd

from .._optional import require
from . import style as beacon_style

require("matplotlib", "Charting")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

# Metrics shown beneath the lines. Total return and volatility are what a
# comparison is usually about; the shared observation count is there because a
# reader should be able to see how much history the comparison actually rests
# on.
METRIC_LABELS = ("Total return", "Volatility", "Max drawdown")


def _level_of(result: Any) -> pd.Series:
    """The level series a result carries, whichever kind it is."""
    for attribute in ("index_levels", "portfolio_nav"):
        series = getattr(result, attribute, None)
        if series is not None and len(series):
            return pd.Series(series).astype(float)

    raise TypeError(
        f"{type(result).__name__} carries no level series to compare; "
        f"expected an IndexResult or a BacktestResult.")


def _label_of(result: Any,
              position: int) -> str:
    """A display name, falling back to the position when there is none."""
    for attribute in ("index_id", "portfolio_id"):
        name = getattr(result, attribute, None)
        if name:
            return str(name)

    return f"Series {position + 1}"


def compare(*results: Any,
            labels: list[str] | None = None,
            ax: Axes | None = None) -> Axes:
    """Plot several results on one rebased axis, with a metrics table.

    Args:
        *results: Two or more `IndexResult` or `BacktestResult` objects.
        labels: Display names. Defaults to each result's own identifier.
        ax: Axes to draw on. A new figure is created when absent.

    Returns:
        Axes: What was drawn on.

    Raises:
        ValueError: If fewer than two results are given, or they share no dates.
    """
    if len(results) < 2:
        raise ValueError(
            f"comparing needs at least two results, got {len(results)}.")

    beacon_style.register()

    series = [_level_of(result) for result in results]
    names = labels or [_label_of(result, position)
                       for position, result in enumerate(results)]

    window = series[0].index
    for other in series[1:]:
        window = window.intersection(other.index)

    if window.empty:
        raise ValueError(
            "these results share no dates, so there is no window to compare "
            "them over.")

    window = window.sort_values()

    if ax is None:
        _, ax = plt.subplots(figsize=beacon_style.FIGSIZE["compare"])

    rows = []
    for name, values in zip(names, series, strict=True):
        clipped = values.loc[window]
        rebased = clipped / float(clipped.iloc[0]) * 100.0

        ax.plot(rebased.index, rebased.to_numpy(), linewidth=1.5, label=name)
        rows.append((name, _metrics(rebased)))

    ax.legend(loc="upper left")
    ax.set_title("Comparison", loc="left", pad=12)
    ax.set_ylabel("Level")

    _annotate(ax, rows, len(window))

    return ax


def _metrics(level: pd.Series) -> tuple[float, float, float]:
    """Total return, annualised volatility and maximum drawdown."""
    returns = level.pct_change().dropna()

    total = float(level.iloc[-1] / level.iloc[0] - 1.0)
    volatility = float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else 0.0
    drawdown = float((level / level.cummax() - 1.0).min())

    return total, volatility, drawdown


def _annotate(ax: Axes,
              rows: list[tuple[str, tuple[float, float, float]]],
              observations: int) -> None:
    """Write the metrics table beneath the chart."""
    header = f"{'':<16}" + "".join(f"{label:>16}" for label in METRIC_LABELS)
    lines = [header]

    for name, (total, volatility, drawdown) in rows:
        lines.append(f"{name[:16]:<16}{total:>15.2%} {volatility:>15.2%} "
                     f"{drawdown:>15.2%}")

    lines.append(f"Aligned on {observations} shared observations.")

    ax.text(0.0, -0.16, "\n".join(lines), transform=ax.transAxes,
            fontsize=7, family="monospace", va="top",
            color=ax.yaxis.label.get_color())
