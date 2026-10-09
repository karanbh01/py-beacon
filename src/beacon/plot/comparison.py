# src/beacon/plot/comparison.py
"""
Comparing several results on one axis.

A free function rather than an accessor because it is not about any one result:
`compare(a, b, c)` is a statement about the set, and hanging it off the first
argument would make that arbitrary. Import it as
`from beacon.plot import compare`.

## Aligned, not concatenated

Every series is clipped to the window they all share and rebased to 100 on the
first shared date. Two indices with different histories compared over different
periods differ for no reason but their spans, and the one with the shorter
history looks better or worse than it is. Rebasing on the shared start means
the lines begin together and the comparison is of shape.
"""
# The module is `comparison` rather than `compare` so it does not shadow the
# function it exports. `from beacon.plot import compare` resolves a submodule
# before it consults the package's lazy attribute hook, so a module of the same
# name would hand back the module and every call site would fail on a
# not-callable error.
import logging
from typing import Any

import pandas as pd

from .._optional import require
from . import style as beacon_style
from .frame import REFERENCE_ID, finish, ink, new_figure

require("matplotlib", "Charting")

from matplotlib.axes import Axes  # noqa: E402

logger = logging.getLogger(__name__)

def _level_of(result: Any) -> pd.Series:
    """The level series a result carries, whichever kind it is: a
    backtest's performance (per unit when money flowed), or an index's
    levels."""
    performance = getattr(result, "performance_levels", None)

    if callable(performance):
        return pd.Series(performance()).astype(float)

    for attribute in ("index_levels", "trading_nav"):
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

    # A BacktestResult names its portfolio through the books (decision 15:
    # the id has one home, on the portfolio).
    portfolio = getattr(result, "portfolio", None)
    name = getattr(portfolio, "portfolio_id", None)
    if name:
        return str(name)

    return f"Series {position + 1}"


def compare(*results: Any,
            labels: list[str] | None = None,
            ax: Axes | None = None,
            title: str | None = None,
            subtitle: str | None = None,
            notes: str | None = None) -> Axes:
    """Plot several results on one rebased axis.

    Args:
        *results: Two or more `IndexResult` or `BacktestResult` objects.
        labels: Display names. Defaults to each result's own identifier
            (its index id or portfolio id), or "Series N" when it has none.
        ax: Axes to draw on. A new figure is created when absent.
        title: The heading. Defaults to "Comparison".
        subtitle: Beneath the heading. Defaults to what the lines are.
        notes: The notes line, after "Notes:". Defaults to how many shared
            observations the comparison rests on.

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
        ax = new_figure(beacon_style.FIGSIZE["compare"]).add_subplot()

    for name, values in zip(names, series, strict=True):
        clipped = values.loc[window]
        rebased = clipped / float(clipped.iloc[0]) * 100.0

        ax.plot(rebased.index, rebased.to_numpy(), linewidth=1.5, label=name)

    # A level chart's one line, at the shared start, in place of gridlines.
    ax.grid(visible=False)
    ax.axhline(100.0, color=ink(ax, "border"),
               linewidth=beacon_style.REFERENCE_WIDTH, zorder=1, gid=REFERENCE_ID)
    ax.legend(loc="upper left")

    shared = (f"Aligned on {len(window)} shared observations, rebased to 100 on "
              f"the first.")

    return finish(ax,
                  "Comparison" if title is None else title,
                  "Each result rebased to 100" if subtitle is None else subtitle,
                  "Level",
                  shared if notes is None else notes)
