# src/beacon/backtest/implementation.py
"""
How a strategy is carried out at a fund's size: what it may hold, and where
the weight of what it may not goes.

    implementation = Implementation(
        screens=[LiquidityScreen(min_traded_value=5e7)],
        redistribution="pro_rata")

    Backtest(initial_capital=1e11, implementation=implementation)

Separate from the strategy (what to hold) and from the vehicle (how money is
held): the same index can be run under the limits of a small fund and of a
large one, and the difference is the cost of size. See
`decisions/0006-backtest-implementation-and-fund-vehicles.md`.

## The rebalance, as stages

1. **Target**: the strategy's weights, as published.
2. **Screens**: names the data says are stale are removed first, then each
   screen in order removes what it rejects.
3. **Redistribution**: removed weight is spread across the remaining names in
   proportion to their weights (``"pro_rata"``), or held as cash
   (``"cash"``).
4. **Trades**, generated from the result, then any modifiers.

Each rebalance's stages are recorded on the result as a `RebalanceStep`.
"""
# BN-264. Capacity caps (BN-265) and execution limits and market impact
# (BN-266) join this object as further stages.
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .screens import Screen, ScreenContext

PRO_RATA = "pro_rata"
CASH = "cash"
REDISTRIBUTIONS = (PRO_RATA, CASH)

# What a RebalanceStep names a name the data says has gone stale.
STALE = "stale price"


class Implementation:
    """A backtest's screens and redistribution rule.

    Args:
        screens: Applied in order at each rebalance; a name must pass all.
        redistribution: Where removed weight goes: ``"pro_rata"`` (the
            default) across the remaining names, or ``"cash"``.

    Raises:
        ValueError: If *redistribution* is not one of the two.
    """

    def __init__(self,
                 screens: Iterable[Screen] = (),
                 redistribution: str = PRO_RATA):
        if redistribution not in REDISTRIBUTIONS:
            raise ValueError(f"Unknown redistribution {redistribution!r}. "
                             f"Supported: {', '.join(REDISTRIBUTIONS)}.")

        self.screens: tuple[Screen, ...] = tuple(screens)
        self.redistribution = redistribution

    def __repr__(self) -> str:
        names = ", ".join(screen.name for screen in self.screens)

        return (f"Implementation(screens=[{names}], "
                f"redistribution={self.redistribution!r})")


@dataclass(frozen=True)
class RebalanceStep:
    """What each stage of one rebalance did.

    Attributes:
        date: The rebalance date.
        target: The strategy's weights, before any stage.
        removed: Each name taken out, and the screen that took it out
            (``"stale price"`` for one the data says has gone stale).
        weights: What the trades aimed at, after redistribution.
        cash_weight: The share of the book the weights leave in cash.
    """
    date: pd.Timestamp
    target: dict[str, float]
    removed: dict[str, str] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 0.0


def plan(implementation: Implementation,
         target: dict[str, float],
         date: pd.Timestamp,
         held: set[str],
         stale: set[str],
         context: ScreenContext) -> RebalanceStep:
    """Run the screening and redistribution stages for one rebalance.

    Args:
        implementation: The screens and redistribution rule.
        target: The strategy's weights.
        date: The rebalance date.
        held: Names the book holds now, for buffered screens.
        stale: Names the data says have gone stale.
        context: The run's data and book currency.

    Returns:
        RebalanceStep: The weights to trade to, and how they were reached.
    """
    removed: dict[str, str] = {name: STALE for name in target if name in stale}

    for name in target:
        if name in removed:
            continue

        for screen in implementation.screens:
            if not screen.admits(name, date, name in held, context):
                removed[name] = screen.name
                break

    kept = {name: weight for name, weight in target.items()
            if name not in removed}
    weights = _redistributed(kept, sum(target.values()),
                             implementation.redistribution)

    return RebalanceStep(date=date, target=dict(target), removed=removed,
                         weights=weights,
                         cash_weight=max(1.0 - sum(weights.values()), 0.0))


def _redistributed(kept: dict[str, float],
                   total: float,
                   rule: str) -> dict[str, float]:
    """*kept* scaled back up to *total*, or left as it is for cash."""
    remaining = sum(kept.values())

    if rule == CASH or remaining <= 0.0:
        return kept

    return {name: weight * total / remaining for name, weight in kept.items()}
