# src/beacon/analysis/attribution/schedule.py
"""
Contributions of an index's names over a window, from its weights at each
rebalance and its names' prices, with what its cap and its costs took.

The weights drift from each rebalance with prices, so a rebalance before the
window still sets the weights inside it: *prices* should cover the whole
history, and the window is chosen afterwards from the drifted weights' days.
"""
# BN-401: moved from the server's attribution view, which now calls this.
import pandas as pd

from ...exceptions import DataNotFoundError
from .contribution import AttributionResult, attribute, cap_drag, cost_drag, drifted_weights


def attribute_schedule(capped: dict[pd.Timestamp, dict[str, float]],
                       uncapped: dict[pd.Timestamp, dict[str, float]],
                       prices: pd.DataFrame,
                       start: str | None,
                       end: str | None,
                       total_costs: float = 0.0,
                       initial_capital: float = 0.0) -> AttributionResult:
    """Each name's contribution to the index's return between *start* and
    *end*.

    Args:
        capped: The weights set at each rebalance, as applied.
        uncapped: The same before the cap; equal to *capped* when nothing
            was capped, and then no cap drag is given.
        prices: Every name's prices, names on the columns, in the index's
            currency.
        start: The window's first date; the first priced day when None.
        end: Its last; the last priced day when None.
        total_costs: Transaction costs paid; no cost drag is given at 0.
        initial_capital: What the costs are measured against.

    Raises:
        DataNotFoundError: If the window has no days. A total return of zero
            over an empty window would read as a flat index rather than as a
            question that could not be answered.
    """
    weights = drifted_weights(capped, prices)
    asset_returns = prices.pct_change().reindex(weights.index)
    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)
    window = _window_of(weights.index, start, end)

    return attribute(period_returns.loc[window],
                     weights.loc[window],
                     asset_returns.loc[window],
                     cap_drag=(None if capped == uncapped
                               else cap_drag(capped, uncapped, prices, window)),
                     cost_drag=(cost_drag(total_costs, initial_capital)
                                if total_costs > 0.0 and initial_capital > 0.0 else None))


def _window_of(index: pd.Index,
               start: str | None,
               end: str | None) -> pd.Index:
    """The dates of *index* from *start* to *end*."""
    selected = pd.DatetimeIndex(index)

    if start is not None:
        selected = selected[selected >= pd.Timestamp(start)]
    if end is not None:
        selected = selected[selected <= pd.Timestamp(end)]

    if selected.empty:
        raise DataNotFoundError(
            f"any index dates between {start or 'the start'} and "
            f"{end or 'the end'}",
            source="the run covers a different period")

    return selected
