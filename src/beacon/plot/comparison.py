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

from .charts import Line
from .charts.base import Chart
from .design import StyleArgument

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
            title: str | None = None,
            subtitle: str | None = None,
            notes: str | None = None,
            style: StyleArgument = None) -> Chart:
    """Plot several results on one rebased axis.

    Args:
        *results: Two or more `IndexResult` or `BacktestResult` objects.
        labels: Display names. Defaults to each result's own identifier
            (its index id or portfolio id), or "Series N" when it has none.
        title: The heading. Defaults to "Comparison".
        subtitle: Beneath the heading. Defaults to what the lines are.
        notes: The notes line, after "Notes:". Defaults to how many shared
            observations the comparison rests on.
        style: Settings for this chart alone: a Style, or settings by
            name such as {"mark": False, "scale": 1.2}. None draws in the
            current style (beacon.plot.configure).

    Returns:
        Chart: The chart, drawn when it is used.

    Raises:
        ValueError: If fewer than two results are given, or they share no dates.
    """
    if len(results) < 2:
        raise ValueError(
            f"comparing needs at least two results, got {len(results)}.")

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
    lines = pd.DataFrame({name: values.loc[window] / float(values.loc[window].iloc[0]) * 100.0
                          for name, values in zip(names, series, strict=True)})

    shared = (f"Aligned on {len(window)} shared observations, rebased to 100 on "
              f"the first.")

    return Line(lines, reference=100.0, ylabel="Level",
                title="Comparison" if title is None else title,
                subtitle="Each result rebased to 100" if subtitle is None else subtitle,
                notes=shared if notes is None else notes, style=style)
