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
4. **Capacity**: each name is cut to its caps at the book's size, the excess
   spread across the names under their caps until none is over (or held as
   cash), and positions too small to keep are dropped and their weight
   redistributed. See `beacon.backtest.capacity`.
5. **Trades**, generated from the result and sized so they and their costs
   fit the cash, then any modifiers.
6. **Execution**: every trade at once, or as much as an execution limit
   allows each day, with market impact on top of the fixed cost. See
   `beacon.backtest.execution`.

Each rebalance's stages are recorded on the result as a `RebalanceStep`.

## Cash and flows

A `cash_buffer` keeps that share of the book in cash: each rebalance invests
the rest, and an outflow is paid from cash before anything is sold. Money
arriving between rebalances is invested the day it arrives, toward the last
rebalance's weights (``invest_flows="target"``, the default) or in proportion
to the holdings (``"holdings"``), keeping the buffer topped up first.
"""
# BN-264. Capacity caps (BN-265) and execution limits and market impact
# (BN-266) join this object as further stages; the cash buffer and how flows
# are invested are BN-267.
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from .capacity import CapacityCap, MinimumPosition
from .costs import ExecutionLimit, MarketImpact
from .screens import Screen, ScreenContext

PRO_RATA = "pro_rata"
CASH = "cash"
REDISTRIBUTIONS = (PRO_RATA, CASH)

# Where money arriving between rebalances goes.
TARGET = "target"
HOLDINGS = "holdings"
FLOW_INVESTMENTS = (TARGET, HOLDINGS)

# What a RebalanceStep names a name the data says has gone stale.
STALE = "stale price"

# What it names a position dropped as too small to keep.
TOO_SMALL = "MinimumPosition"

# Below this a weight difference is rounding, not a breach of a cap.
TOLERANCE = 1e-12


class Implementation:
    """A backtest's screens, capacity limits and redistribution rule.

    Args:
        screens: Applied in order at each rebalance; a name must pass all.
        caps: Capacity caps; a name is held at no more than the smallest.
        minimum_position: Positions too small to keep, or None to keep all.
        impact: Market impact charged on every trade, beside the fixed cost,
            or None for none.
        execution: How much of an order can trade in a day, or None to
            trade every order in full on the rebalance day.
        redistribution: Where removed or capped weight goes: ``"pro_rata"``
            (the default) across the remaining names, or ``"cash"``.
        cash_buffer: The share of the book kept in cash, such as 0.02 for 2%.
            Outflows are paid from cash first.
        invest_flows: What money arriving between rebalances buys: the last
            rebalance's weights (``"target"``, the default) or the holdings
            in proportion to their value (``"holdings"``).

    Raises:
        ValueError: If *redistribution* or *invest_flows* is not one of its
            choices, or *cash_buffer* is not from 0 up to 1.
    """

    def __init__(self,
                 screens: Iterable[Screen] = (),
                 caps: Iterable[CapacityCap] = (),
                 minimum_position: MinimumPosition | None = None,
                 impact: MarketImpact | None = None,
                 execution: ExecutionLimit | None = None,
                 redistribution: str = PRO_RATA,
                 cash_buffer: float = 0.0,
                 invest_flows: str = TARGET):
        if redistribution not in REDISTRIBUTIONS:
            raise ValueError(f"Unknown redistribution {redistribution!r}. "
                             f"Supported: {', '.join(REDISTRIBUTIONS)}.")

        if invest_flows not in FLOW_INVESTMENTS:
            raise ValueError(f"Unknown invest_flows {invest_flows!r}. "
                             f"Supported: {', '.join(FLOW_INVESTMENTS)}.")

        if not 0.0 <= cash_buffer < 1.0:
            raise ValueError(f"cash_buffer is a share of the book from 0 up "
                             f"to 1, got {cash_buffer!r}.")

        self.screens: tuple[Screen, ...] = tuple(screens)
        self.caps: tuple[CapacityCap, ...] = tuple(caps)
        self.minimum_position = minimum_position
        self.impact = impact
        self.execution = execution
        self.redistribution = redistribution
        self.cash_buffer = cash_buffer
        self.invest_flows = invest_flows

    def __repr__(self) -> str:
        screens = ", ".join(screen.name for screen in self.screens)
        caps = ", ".join(cap.name for cap in self.caps)

        return (f"Implementation(screens=[{screens}], caps=[{caps}], "
                f"redistribution={self.redistribution!r})")


@dataclass(frozen=True)
class RebalanceStep:
    """What each stage of one rebalance did.

    Attributes:
        date: The rebalance date.
        target: The strategy's weights, before any stage.
        removed: Each name taken out, and the screen that took it out
            (``"stale price"`` for one the data says has gone stale,
            ``"MinimumPosition"`` for one too small to keep).
        capped: Each name cut to its capacity, and the weight it was cut to.
        weights: What the trades aimed at, after redistribution.
        cash_weight: The share of the book the weights leave in cash.
    """
    date: pd.Timestamp
    target: dict[str, float]
    removed: dict[str, str] = field(default_factory=dict)
    capped: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 0.0


def plan(implementation: Implementation,
         target: dict[str, float],
         date: pd.Timestamp,
         held: set[str],
         stale: set[str],
         context: ScreenContext,
         book_value: float = 0.0) -> RebalanceStep:
    """Run the screening, redistribution and capacity stages for one
    rebalance.

    Args:
        implementation: The screens, caps and redistribution rule.
        target: The strategy's weights.
        date: The rebalance date.
        held: Names the book holds now, for buffered screens.
        stale: Names the data says have gone stale.
        context: The run's data and book currency.
        book_value: The book's value before the rebalance, which caps and
            minimum positions are measured against.

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

    total = sum(target.values())
    kept = {name: weight for name, weight in target.items()
            if name not in removed}
    weights = _redistributed(kept, total, implementation.redistribution)

    # BN-267: the buffer is set aside before the caps, so a name dropped as
    # too small is redistributed within what is invested.
    if implementation.cash_buffer > 0.0:
        invested = 1.0 - implementation.cash_buffer
        weights = {name: weight * invested for name, weight in weights.items()}
        total *= invested

    weights, capped = _within_capacity(implementation, weights, total, date,
                                       book_value, context, removed)

    return RebalanceStep(date=date, target=dict(target), removed=removed,
                         capped=capped, weights=weights,
                         cash_weight=max(1.0 - sum(weights.values()), 0.0))


def _within_capacity(implementation: Implementation,
                     weights: dict[str, float],
                     total: float,
                     date: pd.Timestamp,
                     book_value: float,
                     context: ScreenContext,
                     removed: dict[str, str]
                     ) -> tuple[dict[str, float], dict[str, float]]:
    """Cut each name to its caps and drop positions too small to keep,
    redistributing as the rule says, until neither changes anything.

    Names dropped as too small are added to *removed*. Returns the weights
    and the names capped, with the weight each was cut to.
    """
    minimum = implementation.minimum_position

    if (not implementation.caps and minimum is None) or book_value <= 0.0:
        return weights, {}

    limits = _limits(implementation.caps, weights, date, book_value, context)
    rule = implementation.redistribution
    capped: dict[str, float] = {}

    # Each pass either finishes or drops at least one name, so this ends.
    for _ in range(len(weights) + 1):
        weights = _water_filled(weights, limits, rule, capped)

        small = ([name for name, weight in weights.items()
                  if minimum.too_small(weight, book_value)]
                 if minimum is not None else [])

        if not small:
            break

        for name in small:
            removed[name] = TOO_SMALL
            capped.pop(name, None)
            del weights[name]

        weights = _redistributed(weights, total, rule)

    return weights, capped


def _limits(caps: tuple[CapacityCap, ...],
            weights: dict[str, float],
            date: pd.Timestamp,
            book_value: float,
            context: ScreenContext) -> dict[str, float]:
    """Each name's largest weight: its tightest cap over the book's value.
    A name no cap can value has no limit."""
    limits: dict[str, float] = {}

    for name in weights:
        values = [value for cap in caps
                  if (value := cap.max_value(name, date, book_value, context))
                  is not None]

        if values:
            limits[name] = max(min(values), 0.0) / book_value

    return limits


def _water_filled(weights: dict[str, float],
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


def _redistributed(kept: dict[str, float],
                   total: float,
                   rule: str) -> dict[str, float]:
    """*kept* scaled back up to *total*, or left as it is for cash."""
    remaining = sum(kept.values())

    if rule == CASH or remaining <= 0.0:
        return kept

    return {name: weight * total / remaining for name, weight in kept.items()}
