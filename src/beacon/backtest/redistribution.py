# src/beacon/backtest/redistribution.py
"""
Moving weight from names that cannot hold it to names that can.

`redistributed` scales what is left back up to the total, or leaves it for
cash. `water_filled` cuts every name over its limit and spreads the excess
across the names under theirs, in proportion to their weights, until none is
over. The capacity caps and a vehicle's diversification limits both use
them.
"""
# Moved out of implementation.py (BN-268) when the vehicle's diversification
# limits came to need them too.

PRO_RATA = "pro_rata"
CASH = "cash"

# Below this a weight difference is rounding, not a breach of a limit.
TOLERANCE = 1e-12


def water_filled(weights: dict[str, float],
                 limits: dict[str, float],
                 rule: str,
                 capped: dict[str, float]) -> dict[str, float]:
    """Cut every name over its limit to it and spread the excess across the
    names under theirs, pro rata, until none is over; under the cash rule
    the excess stays in cash. Records each cut name in *capped*."""
    weights = dict(weights)

    for _ in range(len(weights) + 1):
        over = [name for name, weight in weights.items()
                if name in limits and weight > limits[name] + TOLERANCE]

        if not over:
            break

        excess = sum(weights[name] - limits[name] for name in over)

        for name in over:
            weights[name] = limits[name]
            capped[name] = limits[name]

        if rule == CASH:
            continue

        room = {name: weight for name, weight in weights.items()
                if name not in capped and weight > 0.0}
        spare = sum(room.values())

        # Every name is at its cap: what is left over stays in cash.
        if spare <= 0.0:
            break

        for name, weight in room.items():
            weights[name] = weight + excess * weight / spare

    return weights


def redistributed(kept: dict[str, float],
                  total: float,
                  rule: str) -> dict[str, float]:
    """*kept* scaled back up to *total*, or left as it is for cash."""
    remaining = sum(kept.values())

    if rule == CASH or remaining <= 0.0:
        return kept

    return {name: weight * total / remaining for name, weight in kept.items()}
