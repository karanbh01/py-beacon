# src/beacon/strategy/factors.py
"""
Factors: what a name is exposed to, such as size, value or momentum, as a
standardised score built from one or more signals.

    Factor("size", FieldSignal(data.market.market_cap), transform="log",
           higher_is_better=False)
    Factor("quality", [FieldSignal(data.features.fundamentals.roe),
                       FieldSignal(data.features.fundamentals.debt_to_equity,
                                   higher_is_better=False)])
    Factor("vendor_quality", FieldSignal(data.features.vendor.quality),
           standardise=False)

A factor is a signal, so it serves wherever one does: as a factor tilt's
factor, or as an active strategy's signal. It differs from a plain signal in
how its scores are standardised.

**Standardising.** Each part's values (made larger-is-better, then
transformed) are centred and scaled across the *universe*: centred on its
cap-weighted mean by default (``centre="cap_weighted"``), so the market
portfolio has no exposure and a book's exposure reads as its tilt against the
market, or on its equally weighted mean (``"equal"``); and scaled by the
standard deviation of the centred values, then capped at 3 either side. With
*centre_by* naming grouping fields (``"sector"``, or ``["region", "sector"]``)
each name is centred on its own group's mean instead, so its score says how
it stands within its group; the scale stays universe-wide unless
*scale_within_groups*. A group of fewer than 5 names in the universe is
centred on the group one level up (the region alone, for a region and
sector), and so on to the whole universe. A name outside the universe is
scored on the universe's figures.

**Composites.** A factor of several parts averages the parts' scores, by
*weights* (equal unless given), over the parts a name has a value for, so a
name missing one part is scored on the rest rather than pulled toward 0; the
average is then standardised again. A part may itself be a factor, with its
own transform and centring.

**Imported exposures.** ``standardise=False`` takes the values as they are,
for exposures a vendor has already standardised and loaded as a feature.

A name with no value scores 0. *returns* names a series in the data whose
closes are the factor's own returns, as a level (a vendor's factor index),
for analyses to use in place of returns estimated from the exposures.
"""
# BN-387. Agreed with Karan on 2026-10-27.
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from ..catalogue import SIGNAL, Display, register
from ..expressions.namespaces import data
from ..expressions.resolve import value_of
from .base import StrategyContext
from .signals import SCORE_LIMIT, FieldSignal, Momentum, Signal

CAP_WEIGHTED = "cap_weighted"
EQUAL = "equal"
CENTRES = (CAP_WEIGHTED, EQUAL)

LOG = "log"
INVERSE = "inverse"
TRANSFORMS = (LOG, INVERSE)

# The fewest names a group needs to be centred on itself.
MINIMUM_GROUP = 5

MARKET_CAP = data.market.market_cap

Cell = tuple[str, ...]


@dataclass
class FactorInputs:
    """What scoring factors on one day reads once and shares: each name's
    market cap and groups, and the groups that fell back to a wider one.

    Attributes:
        context: The run's data and book currency.
        date: The day.
        names: Every name scored.
        fallbacks: For each factor by name, the groups too small to centre
            on, and the wider group each was centred on instead.
    """
    context: StrategyContext
    date: pd.Timestamp
    names: list[str]
    fallbacks: dict[str, dict[Cell, Cell]] = field(default_factory=dict)
    _caps: dict[str, float] | None = None
    _raw_caps: dict[str, float | None] | None = None
    _groups: dict[str, dict[str, str | None]] = field(default_factory=dict)

    def raw_caps(self) -> dict[str, float | None]:
        """Each name's market cap in the book's currency, None when unknown,
        read once for every factor that weights or scores by it."""
        if self._raw_caps is None:
            self._raw_caps = FieldSignal(MARKET_CAP).values(self.names, self.date, self.context)

        return self._raw_caps

    def caps(self) -> dict[str, float]:
        """Each name's market cap in the book's currency, 0 when unknown."""
        if self._caps is None:
            self._caps = {name: value if value is not None and value > 0 else 0.0
                          for name, value in self.raw_caps().items()}

        return self._caps

    def group(self,
              grouping: str) -> dict[str, str | None]:
        """Each name's value of the reference field *grouping*."""
        if grouping not in self._groups:
            reference = getattr(data.reference, grouping)
            self._groups[grouping] = {
                name: _text(value_of(reference, name, self.date, self.context.fetcher))
                for name in self.names}

        return self._groups[grouping]


@register(SIGNAL, "Factor", fields={
    "name": Display("Name"),
    "signals": Display("Signals", help="One signal, or several averaged."),
    "weights": Display("Weights", help="One for each signal. Leave blank for equal."),
    "transform": Display("Transform", choices=TRANSFORMS,
                         help="Applied to each signal's values before standardising."),
    "higher_is_better": Display("Higher is more exposed"),
    "standardise": Display("Standardise",
                           help="Off for exposures already standardised, such as a "
                                "vendor's."),
    "centre": Display("Centre on", choices=CENTRES),
    "centre_by": Display("Centre within", help="Grouping fields, such as sector, or "
                         "region and sector. Leave blank for the whole universe."),
    "scale_within_groups": Display("Scale within groups"),
    "returns": Display("Factor returns", help="A series in the data whose closes are "
                       "the factor's level. Leave blank to estimate its returns."),
})
class Factor(Signal):
    """A factor: a standardised score from one or more signals.

    Args:
        name: What the factor is called, such as ``"value"``.
        signals: One signal, or several whose scores are averaged.
        weights: Each signal's weight in the average; equal when None.
        transform: ``"log"`` or ``"inverse"``, applied to each signal's
            values (made larger-is-better first) before standardising; a
            value it cannot take (a log of 0 or less) is missing.
        higher_is_better: Whether a larger score is more exposed; False
            negates it.
        standardise: Whether the values are centred and scaled; False takes
            them as they are.
        centre: ``"cap_weighted"`` or ``"equal"``: the mean they are centred
            on.
        centre_by: Grouping fields to centre within, such as ``"sector"`` or
            ``["region", "sector"]``.
        scale_within_groups: Scale by each group's spread rather than the
            universe's.
        returns: A series in the data whose closes are the factor's level.

    Raises:
        ValueError: If the name is empty, there are no signals, the weights
            do not match them or are negative or all 0, or a choice is
            unknown.
    """

    # The shapes the editor renders these in, and the family the signals'
    # types come from (BN-330).
    PARAM_SCHEMAS: ClassVar[dict[str, str]] = {"signals": "TypedSpecs",
                                                "weights": "SignalWeights",
                                                "centre_by": "GroupingFields"}
    PARAM_KINDS: ClassVar[dict[str, str]] = {"signals": SIGNAL}

    def __init__(self,
                 name: str,
                 signals: Signal | Sequence[Signal],
                 weights: Sequence[float] | None = None,
                 transform: str | None = None,
                 higher_is_better: bool = True,
                 standardise: bool = True,
                 centre: str = CAP_WEIGHTED,
                 centre_by: str | Sequence[str] | None = None,
                 scale_within_groups: bool = False,
                 returns: str | None = None):
        super().__init__(higher_is_better)
        parts = [signals] if isinstance(signals, Signal) else list(signals)

        if not name:
            raise ValueError("A factor needs a name.")

        if not parts or not all(isinstance(part, Signal) for part in parts):
            raise ValueError("A factor needs one or more signals.")

        self._name = name
        self.signals = parts
        self.weights = _weights(weights, len(parts))

        if transform is not None and transform not in TRANSFORMS:
            raise ValueError(f"Unknown transform {transform!r}. Known: {', '.join(TRANSFORMS)}.")

        if centre not in CENTRES:
            raise ValueError(f"Unknown centre {centre!r}. Known: {', '.join(CENTRES)}.")

        self.transform = transform
        self.standardise = standardise
        self.centre = centre
        self.centre_by = ([centre_by] if isinstance(centre_by, str)
                          else list(centre_by or []))
        self.scale_within_groups = scale_within_groups
        self.returns = returns

    @property
    def name(self) -> str:
        """The factor's name."""
        return self._name

    def values(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float | None]:
        """Each name's score, None where it has none."""
        inputs = FactorInputs(context, date, list(names))

        return self.scored(names, names, inputs)

    def scores(self,
               names: list[str],
               date: pd.Timestamp,
               context: StrategyContext) -> dict[str, float]:
        """Each name's score, standardised across *names*; 0 where it has
        none."""
        return self.exposures(names, date, context)

    def exposures(self,
                  names: list[str],
                  date: pd.Timestamp,
                  context: StrategyContext,
                  universe: list[str] | None = None,
                  inputs: FactorInputs | None = None) -> dict[str, float]:
        """Each of *names*' score, standardised across *universe* (*names*
        when None); 0 where it has none."""
        universe = list(universe if universe is not None else names)
        everyone = sorted(set(names) | set(universe))
        inputs = inputs or FactorInputs(context, date, everyone)
        found = self.scored(everyone, universe, inputs)

        return {name: found.get(name) or 0.0 for name in names}

    def scored(self,
               names: list[str],
               universe: list[str],
               inputs: FactorInputs) -> dict[str, float | None]:
        """Each name's score before missing values become 0."""
        parts = [self._part(part, names, universe, inputs) for part in self.signals]
        combined = _averaged(parts, self.weights, names)

        if len(parts) > 1 and self.standardise:
            combined = self._standardised(combined, universe, inputs)

        if not self.higher_is_better:
            combined = {name: None if value is None else -value
                        for name, value in combined.items()}

        return combined

    def _part(self,
              part: Signal,
              names: list[str],
              universe: list[str],
              inputs: FactorInputs) -> dict[str, float | None]:
        """One part's scores: a factor's own, or a signal's values made
        larger-is-better, transformed and standardised."""
        if isinstance(part, Factor):
            return part.scored(names, universe, inputs)

        raw = (_subset(inputs.raw_caps(), names) if _reads_market_cap(part)
               else part.values(names, inputs.date, inputs.context))
        sign = 1.0 if part.higher_is_better else -1.0
        values = {name: _transformed(None if value is None else sign * value, self.transform)
                  for name, value in raw.items()}

        return self._standardised(values, universe, inputs) if self.standardise else values

    def _standardised(self,
                      values: dict[str, float | None],
                      universe: list[str],
                      inputs: FactorInputs) -> dict[str, float | None]:
        """*values* centred and scaled on the universe's, capped at 3."""
        known = [name for name in universe if values.get(name) is not None]

        if len(known) < 2:
            return dict.fromkeys(values)

        cells = self._cells(list(values), inputs)
        weights = (inputs.caps() if self.centre == CAP_WEIGHTED
                   else dict.fromkeys(values, 1.0))
        members = self._members(known, cells)
        chosen = {cell: self._fallback(cell, members, inputs) for cell in set(cells.values())}
        means = {group: _mean([values[name] for name in members[group]],
                              [weights.get(name, 0.0) for name in members[group]])
                 for group in set(chosen.values())}
        centred = {name: None if value is None else value - means[chosen[cells[name]]]
                   for name, value in values.items()}
        spreads = self._spreads(centred, known, cells, chosen, members)
        scored: dict[str, float | None] = {}

        for name, value in centred.items():
            spread = spreads[chosen[cells[name]]] if self.scale_within_groups else spreads[()]
            # min and max on one float, as np.clip gives it (BN-411: np.clip a
            # name at a time was 0.8 s of 3.5 at 6,000 names).
            scored[name] = (None if value is None
                            else 0.0 if spread <= 0.0
                            else float(min(max(value / spread, -SCORE_LIMIT), SCORE_LIMIT)))

        return scored

    def _cells(self,
               names: list[str],
               inputs: FactorInputs) -> dict[str, Cell]:
        """Each name's group under *centre_by*: a value per field."""
        if not self.centre_by:
            return dict.fromkeys(names, ())

        groups = [inputs.group(grouping) for grouping in self.centre_by]

        return {name: tuple(str(found.get(name)) for found in groups) for name in names}

    @staticmethod
    def _members(known: list[str],
                 cells: dict[str, Cell]) -> dict[Cell, list[str]]:
        """The universe's names with a value, by every group and each wider
        one they belong to, down to the whole universe, ``()``."""
        members: dict[Cell, list[str]] = {}

        for name in known:
            cell = cells[name]

            for size in range(len(cell) + 1):
                members.setdefault(cell[:size], []).append(name)

        return members

    def _fallback(self,
                  cell: Cell,
                  members: dict[Cell, list[str]],
                  inputs: FactorInputs) -> Cell:
        """*cell*, or the narrowest wider group with enough names."""
        for size in range(len(cell), -1, -1):
            wider = cell[:size]

            if len(members.get(wider, [])) >= MINIMUM_GROUP or size == 0:
                if wider != cell:
                    inputs.fallbacks.setdefault(self.name, {})[cell] = wider

                return wider

        return ()

    @staticmethod
    def _spreads(centred: dict[str, float | None],
                 known: list[str],
                 cells: dict[str, Cell],
                 chosen: dict[Cell, Cell],
                 members: dict[Cell, list[str]]) -> dict[Cell, float]:
        """The standard deviation of the universe's centred values, over all
        of it and over each group centred on."""
        def spread(names: list[str]) -> float:
            return float(np.std(np.array([centred[name] for name in names], dtype=float)))

        spreads: dict[Cell, float] = {(): spread(known)}

        for group in set(chosen.values()):
            spreads[group] = spread(members[group])

        return spreads

    def __repr__(self) -> str:
        return (f"Factor(name={self.name!r}, signals={self.signals!r}, "
                f"weights={self.weights!r}, transform={self.transform!r}, "
                f"higher_is_better={self.higher_is_better!r}, "
                f"standardise={self.standardise!r}, centre={self.centre!r}, "
                f"centre_by={self.centre_by!r}, "
                f"scale_within_groups={self.scale_within_groups!r}, returns={self.returns!r})")


def default_factors() -> list[Factor]:
    """Size (minus log market cap), value (book to price, one over P/B) and
    momentum (the twelve-month return leaving out the last month)."""
    return [Factor("size", FieldSignal(MARKET_CAP), transform=LOG, higher_is_better=False),
            Factor("value", FieldSignal(data.features.fundamentals.pb_ratio),
                   transform=INVERSE),
            Factor("momentum", Momentum(lookback_days=252, skip_days=21))]


def _reads_market_cap(part: Signal) -> bool:
    """Whether *part* is a field signal on the market cap, which the day's
    inputs have read already."""
    field = getattr(part, "field", None)

    return (isinstance(part, FieldSignal) and field is not None
            and (field.namespace, field.name) == (MARKET_CAP.namespace, MARKET_CAP.name))


def _subset(values: dict[str, float | None],
            names: list[str]) -> dict[str, float | None]:
    """*values* for *names*, reading any they lack as missing."""
    return {name: values.get(name) for name in names}


def _weights(weights: Sequence[float] | None,
             count: int) -> list[float]:
    """The parts' weights, checked; equal when None."""
    if weights is None:
        return [1.0 / count] * count

    given = [float(weight) for weight in weights]

    if len(given) != count:
        raise ValueError(f"A factor of {count} signals needs {count} weights, "
                         f"got {len(given)}.")

    if any(not math.isfinite(weight) or weight < 0.0 for weight in given) or sum(given) <= 0.0:
        raise ValueError(f"A factor's weights are zero or more and not all zero, got {given}.")

    return given


def _transformed(value: float | None,
                 transform: str | None) -> float | None:
    """*value* under *transform*; None where it cannot be taken."""
    if value is None or not math.isfinite(value):
        return None

    if transform == LOG:
        return math.log(value) if value > 0.0 else None

    if transform == INVERSE:
        return 1.0 / value if value != 0.0 else None

    return value


def _averaged(parts: list[dict[str, float | None]],
              weights: list[float],
              names: list[str]) -> dict[str, float | None]:
    """Each name's weighted average over the parts it has a value for."""
    if len(parts) == 1:
        return parts[0]

    averaged: dict[str, float | None] = {}

    for name in names:
        pairs = [(weight, part[name]) for weight, part in zip(weights, parts, strict=True)
                 if part.get(name) is not None]
        total = sum(weight for weight, _ in pairs)
        averaged[name] = (sum(weight * value for weight, value in pairs if value is not None)
                          / total if total > 0.0 else None)

    return averaged


def _mean(values: list[float | None],
          weights: list[float]) -> float:
    """The weighted mean of *values*, equal-weighted when no weight is
    positive."""
    numbers = np.array(values, dtype=float)
    given = np.array(weights, dtype=float)

    if given.sum() <= 0.0:
        return float(numbers.mean())

    return float((numbers * given).sum() / given.sum())


def _text(value: object) -> str | None:
    """A group's value as text; None when the name has none."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None

    return str(value)
