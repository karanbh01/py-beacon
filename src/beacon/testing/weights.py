# src/beacon/testing/weights.py
"""
A minimal valid IndexResult built from a raw weight schedule.

The backtest engine's raw weight-dict mode was removed in BN-165:
``index_result`` is the sole schedule source. Tests, examples and notebooks
still want the cheap construction — "these weights on these dates, nothing
else" — so this helper turns that dict into an :class:`IndexResult` the engine
accepts, with every field a consumer reads filled consistently:

* ``weight_snapshots`` and ``constituent_snapshots`` carry the schedule
  verbatim, so the engine trades exactly the weights that were written down.
* ``index_levels`` is a flat series at *base_value* over the schedule's
  business-day span — a stand-in level, not a claim about performance.
* ``divisor_history`` is 1.0 throughout, and the daily weights panel prices
  every constituent at 1.0, so the divisor identity holds trivially:
  Σ amount × price ÷ divisor = base_value on every day.

Core-only: pandas, nothing optional.
"""
import pandas as pd

from ..index.result import IndexResult, daily_weights_frame


def index_result_from_weights(schedule: dict[pd.Timestamp, dict[str, float]],
                              index_id: str = "TEST",
                              base_value: float = 1000.0) -> IndexResult:
    """Build a minimal valid :class:`IndexResult` from a weight schedule.

    Args:
        schedule: Mapping of rebalance date (anything ``pd.Timestamp``
            accepts) to ``{asset_id: weight}``. An empty inner dict is a
            rebalance into cash.
        index_id: Identifier stamped on the result.
        base_value: The flat level the synthesised series holds.

    Returns:
        IndexResult: Snapshots straight from *schedule*, a flat level series
        over its business-day span, divisor 1.0, and a daily weights panel
        that forward-fills each snapshot until the next.

    Raises:
        ValueError: If *schedule* is empty — an engine with no rebalance
            schedule at all has nothing to simulate.
    """
    if not schedule:
        raise ValueError("schedule must contain at least one rebalance date.")

    snapshots = {pd.Timestamp(date): dict(weights)
                 for date, weights in schedule.items()}
    rebalances = sorted(snapshots)

    # The span covers every rebalance; union keeps a date that is not itself
    # a business day rather than silently dropping it.
    span = pd.bdate_range(start=rebalances[0], end=rebalances[-1],
                          freq="B").union(pd.DatetimeIndex(rebalances))

    levels = pd.Series(base_value, index=span, dtype=float)
    divisors = pd.Series(1.0, index=span)

    # Each day holds the latest snapshot on or before it; with every
    # constituent priced at 1.0, the amount that reproduces the level is
    # simply weight x base_value.
    records: list[dict[str, object]] = []
    effective: dict[str, float] = {}
    upcoming = list(rebalances)

    for day in span:
        while upcoming and upcoming[0] <= day:
            effective = snapshots[upcoming.pop(0)]

        for asset_id, weight in effective.items():
            records.append({"DATE": day,
                            "IDENTIFIER": asset_id,
                            "AMOUNT": weight * base_value,
                            "WEIGHT": weight})

    return IndexResult(
        index_id=index_id,
        index_levels=levels,
        divisor_history=divisors,
        constituent_snapshots={date: list(weights)
                               for date, weights in snapshots.items()},
        weight_snapshots=snapshots,
        daily_weights=daily_weights_frame(records),
    )
