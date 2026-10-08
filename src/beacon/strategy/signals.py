# src/beacon/strategy/signals.py
"""
Signals: what an active strategy believes about each name, as a score.

    FieldSignal(data.features.fundamentals.earnings_yield)
    Momentum(lookback_days=252, skip_days=21)
    FunctionSignal(my_function, higher_is_better=False)

A signal gives each candidate a value at a rebalance. The values are turned
into **scores**: standardised across the candidates (mean 0, standard
deviation 1), capped at 3 either side so one outlier cannot dominate, and
negated when lower is better. A name with no value scores 0, so it is held
neither over nor under on the signal's account.
"""
# BN-280, phase 7 of decisions/0006.
import math
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import pandas as pd

from ..catalogue import SIGNAL, register
from ..data.fetcher import DataFetcher
from ..expressions.core import Field
from ..expressions.resolve import value_of
from .base import StrategyContext, trailing_returns

# How far either side of the mean a score may go, in standard deviations.
SCORE_LIMIT = 3.0


class Signal(ABC):
    """A value for each name at a rebalance.

    Args:
        higher_is_better: Whether a larger value is a stronger case to hold
            the name.
    """

    def __init__(self,
                 higher_is_better: bool = True):
        self.higher_is_better = higher_is_better

    @property
    def name(self) -> str:
        """How the run's record names this signal."""
        return type(self).__name__

    @abstractmethod
    def values(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float | None]:
        """Each name's value on *date*, None where it has none."""

    def scores(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float]:
        """Each name's standardised score on *date*."""
        scored = standardised(self.values(names, date, context))

        return scored if self.higher_is_better else {name: -score
                                                     for name, score in scored.items()}


@register(SIGNAL, "Field")
class FieldSignal(Signal):
    """A field's value: a market column, a reference value or a feature.

    Args:
        field: The field, such as ``data.features.fundamentals.earnings_yield``.
        higher_is_better: Whether a larger value is better.
    """

    def __init__(self,
                 field: Field,
                 higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.field = field

    def values(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float | None]:
        return {name: _number(value_of(self.field, name, date, context.fetcher,
                                       currency=context.currency))
                for name in names}

    @property
    def name(self) -> str:
        return f"FieldSignal({self.field.namespace}.{self.field.name})"


class FunctionSignal(Signal):
    """A value from a function of the name, the date and the data.

    Args:
        function: Called as ``function(name, date, fetcher)``; returns a
            number, or None when the name has no value.
        higher_is_better: Whether a larger value is better.
    """

    def __init__(self,
                 function: Callable[[str, pd.Timestamp, DataFetcher], float | None],
                 higher_is_better: bool = True):
        super().__init__(higher_is_better)
        self.function = function

    def values(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float | None]:
        return {name: _number(self.function(name, date, context.fetcher))
                for name in names}


@register(SIGNAL, "Momentum")
class Momentum(Signal):
    """Price momentum: the return over a lookback, leaving out the most
    recent days, where short-term reversal works against it.

    Args:
        lookback_days: Trading days the return is measured over.
        skip_days: The most recent trading days left out.

    Raises:
        ValueError: If *skip_days* is not below *lookback_days*.
    """

    def __init__(self,
                 lookback_days: int = 252,
                 skip_days: int = 21):
        if not 0 <= skip_days < lookback_days:
            raise ValueError("Momentum needs 0 <= skip_days < lookback_days.")

        super().__init__(higher_is_better=True)
        self.lookback_days = lookback_days
        self.skip_days = skip_days

    def values(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float | None]:
        returns = trailing_returns(names, date, context, self.lookback_days)
        window = returns.iloc[:len(returns) - self.skip_days] if self.skip_days else returns
        values: dict[str, float | None] = {}

        for name in names:
            series = window[name].dropna() if name in window else pd.Series(dtype=float)
            # Most of the window must be there for the return to mean anything.
            enough = len(series) >= (self.lookback_days - self.skip_days) // 2
            values[name] = float((1.0 + series).prod() - 1.0) if enough else None

        return values


def standardised(values: dict[str, float | None]) -> dict[str, float]:
    """Values as scores: standardised across the names that have one, capped
    at three standard deviations, and 0 for a name with none."""
    known = {name: value for name, value in values.items() if value is not None}
    numbers = np.array(list(known.values()), dtype=float)

    if len(numbers) < 2 or float(numbers.std()) == 0.0:
        return dict.fromkeys(values, 0.0)

    mean, spread = float(numbers.mean()), float(numbers.std())

    return {name: (float(np.clip((known[name] - mean) / spread, -SCORE_LIMIT, SCORE_LIMIT))
                   if name in known else 0.0)
            for name in values}


def _number(value: object) -> float | None:
    """A finite number, or None."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        return None

    number = float(value)

    return number if math.isfinite(number) else None
