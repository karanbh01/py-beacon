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
to the holdings (``"holdings"``), keeping the buffer topped up first. A
`cash_band` around the buffer is checked every day: cash above it is
invested back down to the buffer, and cash below it raised back up to it
(see `beacon.backtest.cash_band`).
"""
# BN-264. Capacity caps (BN-265) and execution limits and market impact
# (BN-266) join this object as further stages; the cash buffer and how flows
# are invested are BN-267.
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from ..strategy.futures import IndexFutures
from .capacity import CapacityCap, MinimumPosition
from .costs import ExecutionLimit, MarketImpact
from .limits import DiversificationLimit
from .redistribution import CASH, PRO_RATA, redistributed, water_filled
from .screens import Screen, ScreenContext

REDISTRIBUTIONS = (PRO_RATA, CASH)

# Where money arriving between rebalances goes.
TARGET = "target"
HOLDINGS = "holdings"
FLOW_INVESTMENTS = (TARGET, HOLDINGS)

# What a RebalanceStep names a name the data says has gone stale.
STALE = "stale price"

# What it names a position dropped as too small to keep.
TOO_SMALL = "MinimumPosition"


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
        cash_band: The lowest and highest share of the book the cash may
            be before it is brought back to *cash_buffer*, checked every
            day; None (the default) leaves it to the next rebalance.
        invest_flows: What money arriving between rebalances buys: the last
            rebalance's weights (``"target"``, the default) or the holdings
            in proportion to their value (``"holdings"``).
        equitise: Index futures held on the cash the fund keeps uninvested,
            so its buffer earns the index's return, or None to leave the cash
            as cash. See :mod:`beacon.strategy.futures`.

    Raises:
        ValueError: If *redistribution* or *invest_flows* is not one of its
            choices, *cash_buffer* is not from 0 up to 1, or *cash_band* does
            not hold it.
    """

    def __init__(self,
                 screens: Iterable[Screen] = (),
                 caps: Iterable[CapacityCap] = (),
                 minimum_position: MinimumPosition | None = None,
                 impact: MarketImpact | None = None,
                 execution: ExecutionLimit | None = None,
                 redistribution: str = PRO_RATA,
                 cash_buffer: float = 0.0,
                 invest_flows: str = TARGET,
                 equitise: IndexFutures | None = None,
                 cash_band: tuple[float, float] | None = None):
        if redistribution not in REDISTRIBUTIONS:
            raise ValueError(f"Unknown redistribution {redistribution!r}. "
                             f"Supported: {', '.join(REDISTRIBUTIONS)}.")

        if invest_flows not in FLOW_INVESTMENTS:
            raise ValueError(f"Unknown invest_flows {invest_flows!r}. "
                             f"Supported: {', '.join(FLOW_INVESTMENTS)}.")

        if not 0.0 <= cash_buffer < 1.0:
            raise ValueError(f"cash_buffer is a share of the book from 0 up "
                             f"to 1, got {cash_buffer!r}.")

        if cash_band is not None:
            lower, upper = cash_band

            if not 0.0 <= lower <= cash_buffer <= upper < 1.0:
                raise ValueError(f"cash_band runs from a lower to an upper share of the "
                                 f"book around cash_buffer ({cash_buffer!r}), from 0 up "
                                 f"to 1, got {cash_band!r}.")

        self.screens: tuple[Screen, ...] = tuple(screens)
        self.caps: tuple[CapacityCap, ...] = tuple(caps)
        self.minimum_position = minimum_position
        self.impact = impact
        self.execution = execution
        self.redistribution = redistribution
        self.cash_buffer = cash_buffer
        self.invest_flows = invest_flows
        self.equitise = equitise
        # BN-358.
        self.cash_band = cash_band

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
        screened: The weights after the screens and redistribution, before
            the cash buffer and the caps.
        weights: What the trades aimed at, after redistribution.
        cash_weight: The share of the book the weights leave in cash.
    """
    date: pd.Timestamp
    target: dict[str, float]
    removed: dict[str, str] = field(default_factory=dict)
    capped: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    cash_weight: float = 0.0
    # BN-406: of the capped names, those the vehicle's diversification limits
    # cut, with the weight each was cut to, so a trade can say why.
    diversified: dict[str, float] = field(default_factory=dict)
    # BN-326: each name's weight between the screens and the caps.
    screened: dict[str, float] = field(default_factory=dict)


def buffered(weights: dict[str, float],
             cash_buffer: float) -> dict[str, float]:
    """*weights* with *cash_buffer*, a share of the book, set aside in cash:
    each scaled by the share left invested."""
    invested = 1.0 - cash_buffer

    return {name: weight * invested for name, weight in weights.items()}


def plan(implementation: Implementation,
         target: dict[str, float],
         date: pd.Timestamp,
         held: set[str],
         stale: set[str],
         context: ScreenContext,
         book_value: float = 0.0,
         limits: Iterable[DiversificationLimit] = ()) -> RebalanceStep:
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
        limits: The vehicle's diversification limits, applied with the caps.

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
    weights = redistributed(kept, total, implementation.redistribution)
    screened = dict(weights)

    # BN-267: the buffer is set aside before the caps, so a name dropped as
    # too small is redistributed within what is invested.
    if implementation.cash_buffer > 0.0:
        weights = buffered(weights, implementation.cash_buffer)
        total *= 1.0 - implementation.cash_buffer

    weights, capped, diversified = _within_capacity(implementation, weights, total, date,
                                                    book_value, context, removed,
                                                    tuple(limits))

    return RebalanceStep(date=date, target=dict(target), removed=removed,
                         capped=capped, weights=weights,
                         cash_weight=max(1.0 - sum(weights.values()), 0.0),
                         screened=screened, diversified=diversified)


def _within_capacity(implementation: Implementation,
                     weights: dict[str, float],
                     total: float,
                     date: pd.Timestamp,
                     book_value: float,
                     context: ScreenContext,
                     removed: dict[str, str],
                     diversification: tuple[DiversificationLimit, ...] = ()
                     ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Cut each name to its caps and the vehicle's diversification limits,
    and drop positions too small to keep, redistributing as the rule says,
    until none changes anything.

    Names dropped as too small are added to *removed*. Returns the weights,
    the names capped with the weight each was cut to, and of those the ones a
    diversification limit cut.
    """
    minimum = implementation.minimum_position

    caps = (*implementation.caps,
            *(cap for limit in diversification for cap in limit.caps()))

    if (not caps and minimum is None and not diversification) or book_value <= 0.0:
        return weights, {}, {}

    limits = _limits(caps, weights, date, book_value, context)
    rule = implementation.redistribution
    capped: dict[str, float] = {}
    diversified: dict[str, float] = {}

    # Each pass either finishes or drops at least one name, so this ends.
    for _ in range(len(weights) + 1):
        weights = water_filled(weights, limits, rule, capped)

        for limit in diversification:
            before = dict(capped)
            weights = limit.limited(weights, rule, capped)
            diversified.update({name: weight for name, weight in capped.items()
                                if before.get(name) != weight})

        small = ([name for name, weight in weights.items()
                  if minimum.too_small(weight, book_value)]
                 if minimum is not None else [])

        if not small:
            break

        for name in small:
            removed[name] = TOO_SMALL
            capped.pop(name, None)
            diversified.pop(name, None)
            del weights[name]

        weights = redistributed(weights, total, rule)

    return weights, capped, diversified


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
