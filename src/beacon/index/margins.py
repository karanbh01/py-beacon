# src/beacon/index/margins.py
"""
How far a name was from each threshold an eligibility rule tested it
against.

An eligibility rule answers whether a name may join an index. Asked to
`assess` a name, it gives the same answer with what it measured: for each
threshold it tested, the quantity, the name's value, the comparison, the
threshold and whether the name passed it. A name's ``margin`` on one is how
far it was from failing, as a share of the threshold and signed so a
positive margin passes:

    at least $5bn, a $6bn name:    (6 - 5) / 5   =  0.2
    at most a P/E of 20, P/E 25:   (20 - 25) / 20 = -0.25

Against a threshold of 0 the margin is the plain difference, and a
comparison that does not order (equal to, one of, between) has none.
"""
# BN-474 (Karan, 2026-11-26): the margin as a share of the threshold; each
# comparison of an expression measured; ranks left until a rule ranks.
import math
import numbers
from dataclasses import dataclass
from typing import Any

# The comparisons a margin is measured for: those above and those below.
AT_LEAST, AT_MOST = ("ge", "gt"), ("le", "lt")


@dataclass(frozen=True, slots=True)
class Measure:
    """One threshold a rule tested a name against.

    Attributes:
        quantity: What was measured, such as ``market_cap`` or a feature's
            name; an expression's comparison by its field's path.
        value: The name's value; None where it had none to test, or the
            value is not a number.
        comparison: How it was tested: ``ge``, ``gt``, ``le``, ``lt``,
            ``eq``, ``ne``, ``in`` or ``between``.
        threshold: What it was tested against; None where that is not a
            number.
        passed: Whether the name passed it.
    """
    quantity: str
    value: float | None
    comparison: str
    threshold: float | None
    passed: bool

    @property
    def margin(self) -> float | None:
        """How far the name was from failing (see the module docstring)."""
        return margin(self.value, self.comparison, self.threshold)


@dataclass(frozen=True, slots=True)
class Assessment:
    """A rule's answer for one name, and what it measured to give it."""
    eligible: bool
    measures: tuple[Measure, ...] = ()


def margin(value: float | None,
           comparison: str,
           threshold: float | None) -> float | None:
    """How far *value* was from failing *comparison* against *threshold*:
    as a share of the threshold, or the plain difference against 0,
    positive when it passes; None when either is missing or the comparison
    does not order."""
    if value is None or threshold is None or not math.isfinite(value):
        return None

    if comparison in AT_LEAST:
        gap = value - threshold
    elif comparison in AT_MOST:
        gap = threshold - value
    else:
        return None

    return gap / abs(threshold) if threshold != 0.0 else gap


def number(value: Any) -> float | None:
    """*value* as a float, or None when it is not a real number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None

    found = float(value)

    return found if math.isfinite(found) else None


__all__ = ["Assessment", "Measure", "margin", "number"]
