# src/beacon/backtest/flows.py
"""
Money arriving in a backtest and leaving it, as a scenario.

    Backtest(initial_capital=1e8,
             flows=[PeriodicFlows(amount=5e6, frequency="MONTHLY"),
                    DatedFlows({"2024-06-03": -2e7})])

Flows are their own part of a run, separate from the strategy, the
implementation and the vehicle, so the same flows can go into different
vehicles. Each scenario says how much arrives (positive) or leaves (negative)
on a simulated day, in the book's currency; several are added together.

| Scenario | A flow day's amount |
| --- | --- |
| `DatedFlows` | The amounts dated on or before the day, since the last one |
| `PeriodicFlows` | A fixed amount, or a share of the fund's assets |
| `RandomFlows` | A share of the assets drawn from a normal distribution, seeded |
| `PerformanceChasingFlows` | A share of the assets that follows the trailing return |

A periodic scenario's flow days are the first simulated day of each period
(each week, month, quarter or year) after the first day of the run, which the
initial capital already funds.

Each flow creates or cancels units at that day's NAV per unit, so performance
is measured per unit and a flow is not mistaken for a return. See
`beacon.backtest.accounting` for how a run deals and invests them.
"""
# BN-267, phase 5 of decisions/0006 (amended 2026-10-05: flows are a fifth
# part of a run, not a vehicle setting).
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

DAILY = "DAILY"
WEEKLY = "WEEKLY"
MONTHLY = "MONTHLY"
QUARTERLY = "QUARTERLY"
ANNUALLY = "ANNUALLY"

# How many flow days a year each frequency has, to scale annual rates.
PERIODS_PER_YEAR = {DAILY: 252, WEEKLY: 52, MONTHLY: 12, QUARTERLY: 4,
                    ANNUALLY: 1}


@dataclass(frozen=True)
class FlowContext:
    """What a scenario is told on each simulated day.

    Attributes:
        previous: The last simulated day, or the eve of the run on the first.
        date: Today.
        aum: The fund's net assets before today's flows, in the book's
            currency.
        nav_per_unit: NAV per unit on each simulated day so far, today's
            before its flows included.
    """
    previous: pd.Timestamp
    date: pd.Timestamp
    aum: float
    nav_per_unit: pd.Series


@dataclass(frozen=True)
class FlowRecord:
    """One day's flow, as the run dealt it.

    Attributes:
        date: The day it was dealt.
        amount: What was paid in (positive) or out (negative), in the book's
            currency. A redemption larger than the fund is cut to the fund.
        nav_per_unit: The day's NAV per unit, before the flow.
        units: Units created (positive) or cancelled (negative).
        dealing_price: The price per unit they were dealt at: NAV per unit,
            or NAV moved by the vehicle's pricing method.
        adjustment: What the dealing investors paid into the fund, through
            the price or a levy, toward the trading their flow caused. 0
            under single pricing.
    """
    date: pd.Timestamp
    amount: float
    nav_per_unit: float
    units: float
    dealing_price: float
    adjustment: float = 0.0


class Flows(ABC):
    """A flow scenario: how much arrives or leaves on each simulated day."""

    def start(self,  # noqa: B027 -- an optional hook, not part of the interface
              first_day: pd.Timestamp) -> None:
        """Get ready for a run whose first simulated day is *first_day*."""

    @abstractmethod
    def amount(self,
               context: FlowContext) -> float:
        """Today's flow in the book's currency: positive in, negative out."""


class DatedFlows(Flows):
    """Amounts on given dates. One dated on a day the run does not simulate
    arrives on the next simulated day.

    Args:
        amounts: Date (YYYY-MM-DD or a timestamp) to amount, positive in and
            negative out.
    """

    def __init__(self,
                 amounts: Mapping[str | pd.Timestamp, float]):
        self.amounts: dict[pd.Timestamp, float] = {
            pd.Timestamp(date): float(value) for date, value in amounts.items()}

    def amount(self,
               context: FlowContext) -> float:
        return sum(value for date, value in self.amounts.items()
                   if context.previous < date <= context.date)


class _Periodic(Flows, ABC):
    """A scenario that flows once a period, after the run's first day."""

    def __init__(self,
                 frequency: str):
        if frequency not in PERIODS_PER_YEAR:
            raise ValueError(f"Unknown frequency {frequency!r}. Supported: "
                             f"{', '.join(PERIODS_PER_YEAR)}.")

        self.frequency = frequency
        self._first_day: pd.Timestamp | None = None

    def start(self,
              first_day: pd.Timestamp) -> None:
        self._first_day = first_day

    def flows_on(self,
                 context: FlowContext) -> bool:
        """Whether today is a flow day: the first of a new period."""
        if self._first_day is not None and context.date <= self._first_day:
            return False

        return _period(context.date, self.frequency) != _period(context.previous,
                                                                self.frequency)

    @property
    def periods_per_year(self) -> int:
        return PERIODS_PER_YEAR[self.frequency]


class PeriodicFlows(_Periodic):
    """A fixed amount, or a fixed share of the fund's assets, each period.

    Args:
        amount: In the book's currency, positive in and negative out.
        fraction: A share of the fund's assets, such as 0.01 for 1% in or
            -0.01 for 1% out.
        frequency: ``"DAILY"``, ``"WEEKLY"``, ``"MONTHLY"`` (the default),
            ``"QUARTERLY"`` or ``"ANNUALLY"``.

    Raises:
        ValueError: Unless exactly one of *amount* and *fraction* is given.
    """

    def __init__(self,
                 amount: float | None = None,
                 fraction: float | None = None,
                 frequency: str = MONTHLY):
        super().__init__(frequency)

        if (amount is None) == (fraction is None):
            raise ValueError("PeriodicFlows needs an amount or a fraction of "
                             "the assets, not both.")

        self.fixed = amount
        self.fraction = fraction

    def amount(self,
               context: FlowContext) -> float:
        if not self.flows_on(context):
            return 0.0

        if self.fixed is not None:
            return self.fixed

        return (self.fraction or 0.0) * context.aum


class RandomFlows(_Periodic):
    """Flows drawn at random as a share of the fund's assets, seeded so a run
    repeats exactly.

    Each flow day's share is normal, with the annual *drift* and *volatility*
    scaled to the frequency: a mean of drift / n and a standard deviation of
    volatility / sqrt(n), for n flow days a year.

    Args:
        drift: The expected net flow a year, as a share of the assets.
        volatility: Its annual standard deviation.
        frequency: How often money moves (``"DAILY"`` by default).
        seed: The random seed.

    Raises:
        ValueError: If *volatility* is negative.
    """

    def __init__(self,
                 drift: float = 0.0,
                 volatility: float = 0.1,
                 frequency: str = DAILY,
                 seed: int = 0):
        super().__init__(frequency)

        if volatility < 0:
            raise ValueError(f"volatility cannot be negative, got "
                             f"{volatility!r}.")

        self.drift = drift
        self.volatility = volatility
        self.seed = seed
        self._random = np.random.default_rng(seed)

    def start(self,
              first_day: pd.Timestamp) -> None:
        super().start(first_day)
        self._random = np.random.default_rng(self.seed)

    def amount(self,
               context: FlowContext) -> float:
        if not self.flows_on(context):
            return 0.0

        n = self.periods_per_year
        share = self._random.normal(self.drift / n,
                                    self.volatility / math.sqrt(n))

        return float(share * context.aum)


class PerformanceChasingFlows(_Periodic):
    """Flows that follow the fund's trailing return, as investors chase
    performance.

    Each flow day's share of the assets is ``base + sensitivity x trailing
    return``, the trailing return being the NAV per unit's change over the
    last *lookback_days* simulated days. No flow until the run has that much
    history.

    Args:
        sensitivity: The share of the assets that flows per unit of trailing
            return: 0.1 turns a 10% trailing return into a 1% inflow.
        base: A share of the assets that flows each flow day whatever the
            return.
        lookback_days: Simulated days the trailing return covers.
        frequency: How often money moves (``"MONTHLY"`` by default).

    Raises:
        ValueError: If *lookback_days* is below 1.
    """

    def __init__(self,
                 sensitivity: float = 0.1,
                 base: float = 0.0,
                 lookback_days: int = 63,
                 frequency: str = MONTHLY):
        super().__init__(frequency)

        if lookback_days < 1:
            raise ValueError(f"lookback_days must be at least 1, got "
                             f"{lookback_days!r}.")

        self.sensitivity = sensitivity
        self.base = base
        self.lookback_days = lookback_days

    def amount(self,
               context: FlowContext) -> float:
        history = context.nav_per_unit

        if not self.flows_on(context) or len(history) <= self.lookback_days:
            return 0.0

        trailing = float(history.iloc[-1] / history.iloc[-1 - self.lookback_days]
                         - 1.0)

        return (self.base + self.sensitivity * trailing) * context.aum


def money_weighted_return(flows: list[tuple[pd.Timestamp, float]]) -> float | None:
    """The annual rate that discounts an investor's cash flows to zero.

    Args:
        flows: (date, amount) from the investor's side: money put in is
            negative, money taken out (and the final value) positive.

    Returns:
        float or None: The internal rate of return a year, over calendar days
        (ACT/365), or None when the flows do not change sign, so no rate
        exists.
    """
    if not flows or min(amount for _, amount in flows) >= 0 \
            or max(amount for _, amount in flows) <= 0:
        return None

    start = min(date for date, _ in flows)
    years = np.array([(date - start).days / 365.0 for date, _ in flows])
    amounts = np.array([amount for _, amount in flows])

    def value(rate: float) -> float:
        return float(np.sum(amounts / (1.0 + rate) ** years))

    low, high = -0.9999, 1.0

    # Widen until the value changes sign; a rate above 1e6 a year is noise.
    while value(high) > 0 and high < 1e6:
        high *= 10.0

    if value(low) * value(high) > 0:
        return None

    for _ in range(200):
        middle = (low + high) / 2.0

        if value(low) * value(middle) <= 0:
            high = middle
        else:
            low = middle

    return (low + high) / 2.0


def _period(date: pd.Timestamp,
            frequency: str) -> tuple[int, ...]:
    """The period *date* falls in, at *frequency*."""
    if frequency == DAILY:
        return (date.year, date.month, date.day)

    if frequency == WEEKLY:
        iso = date.isocalendar()

        return (iso.year, iso.week)

    if frequency == MONTHLY:
        return (date.year, date.month)

    if frequency == QUARTERLY:
        return (date.year, (date.month - 1) // 3)

    return (date.year,)
