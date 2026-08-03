# src/beacon/risk/contribution.py
"""
Which holdings actually drive an index's risk.

A weights table says what the index owns. It does not say what the index is
*exposed to*, and the two differ enough to matter: a name at 8% of a quiet
utility might account for 3% of volatility, while a name at 4% of something
volatile that moves with everything else accounts for 9%. A weights table
without this column looks like a risk view and is not one.

## The decomposition, and why it is exact

For weights ``w`` and an annualised covariance ``S``, portfolio volatility is
``sigma = sqrt(w' S w)``. Differentiating gives each name's *marginal*
contribution — how much volatility changes per unit of additional weight:

    marginal = (S w) / sigma

and its *component* contribution is its weight times that:

    contribution[i] = w[i] x marginal[i]

These sum to ``sigma`` **exactly**, not approximately, because ``sigma`` is
homogeneous of degree one in the weights and Euler's theorem applies. That is
worth knowing because it makes the acceptance test a real one: if the parts do
not add to the whole, something is wrong and there is no tolerance to hide
behind.

## Names the model does not cover

A constituent added last week has too little history to estimate against. Three
ways to handle it, and only one of them is honest:

* drop it and renormalise the rest — this claims the index holds more of the
  covered names than it does, and silently restates the portfolio
* fail the whole request — one new name blanks the column for 499 others
* compute over the covered names **at their actual weights**, and report what
  fraction of the index that was

The third is what happens here. The reported volatility is then genuinely the
volatility of the covered part *as held*, the identity still holds exactly over
that part, and `covered_weight` says how much of the index the figure speaks
for. A number that describes 94% of an index and says so is more useful than
one that describes 100% of a portfolio nobody holds.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskContributions:
    """How a portfolio's volatility divides among its holdings.

    Used for both total and active risk: the arithmetic is identical, only the
    weight vector differs.

    Attributes:
        volatility: Annualised volatility of the covered holdings, at the
            weights they are actually held. For an active decomposition this is
            the tracking error.
        marginal: Per name, the change in volatility per unit of extra weight.
        contribution: Per name, its share of `volatility`. Sums to it exactly.
            **Can be negative for active risk**, where an underweight that
            hedges an overweight genuinely reduces tracking error.
        covered_weight: Fraction the estimate speaks for. 1.0 when the model
            covers everything. For active risk this is a share of *gross*
            active weight, since active weights sum to roughly zero.
        uncovered: Names the covariance has no row for, so a reader can see
            which are missing rather than only how much weight is.
    """
    volatility: float
    marginal: dict[str, float] = field(default_factory=dict)
    contribution: dict[str, float] = field(default_factory=dict)
    covered_weight: float = 0.0
    uncovered: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        """Whether the model covered the whole index."""
        return not self.uncovered


def _decompose(weights: dict[str, float],
               covariance: pd.DataFrame,
               coverage: Callable[[list[str], tuple[str, ...]], float],
               subject: str) -> RiskContributions:
    """The shared decomposition, over whatever weight vector it is handed.

    Total and active risk are the same arithmetic on different vectors: one on
    the holdings, one on the holdings minus the benchmark. Writing it twice
    would let the two drift, and the identity is the property most worth not
    breaking.
    """
    if not weights or covariance.empty:
        return RiskContributions(volatility=0.0)

    available = [name for name in weights if name in covariance.index]
    missing = tuple(sorted(name for name in weights if name not in covariance.index))

    if missing:
        logger.warning(
            "The risk model covers %d of %d name(s) for the %s decomposition; "
            "the rest are excluded.", len(available), len(weights), subject)

    if not available:
        return RiskContributions(volatility=0.0, uncovered=missing)

    vector = np.array([weights[name] for name in available], dtype=float)
    matrix = covariance.loc[available, available].to_numpy(dtype=float)

    variance = float(vector @ matrix @ vector)
    if variance <= 0.0:
        # A degenerate covariance — every asset with zero variance, or a matrix
        # that is not positive semi-definite. It is also the ordinary answer
        # for active risk when the portfolio *is* the benchmark. Reporting zero
        # contributions beats dividing by a volatility of zero.
        logger.warning("%s variance is %.3e; no decomposition is possible.",
                       subject.capitalize(), variance)

        return RiskContributions(volatility=0.0, uncovered=missing,
                                 covered_weight=coverage(available, missing))

    volatility = float(np.sqrt(variance))
    marginal = (matrix @ vector) / volatility

    return RiskContributions(
        volatility=volatility,
        marginal={name: float(value)
                  for name, value in zip(available, marginal, strict=True)},
        contribution={name: float(weight * value)
                      for name, weight, value
                      in zip(available, vector, marginal, strict=True)},
        covered_weight=coverage(available, missing),
        uncovered=missing)


def risk_contributions(weights: dict[str, float],
                       covariance: pd.DataFrame) -> RiskContributions:
    """Decompose portfolio volatility across its holdings.

    Args:
        weights: Holdings, identifier to weight. Need not sum to one — they
            will not when part of the index is uncovered, and renormalising
            would restate the portfolio.
        covariance: Annualised covariance, indexed and columned by identifier.

    Returns:
        RiskContributions: The decomposition, with contributions summing to the
        reported volatility exactly.
    """
    def covered(available: list[str],
                _missing: tuple[str, ...]) -> float:
        return float(sum(weights[name] for name in available))

    return _decompose(weights, covariance, covered, "portfolio")


def active_weights(weights: dict[str, float],
                   benchmark: dict[str, float]) -> dict[str, float]:
    """Holdings minus benchmark, over the union of both.

    A name held and not in the benchmark is an overweight; one in the benchmark
    and not held is an underweight of its full benchmark weight. Taking the
    union rather than the intersection is what makes the second case visible —
    an omitted constituent is usually the largest active position a portfolio
    has, and intersecting would silently drop it.
    """
    names = sorted(set(weights) | set(benchmark))

    return {name: weights.get(name, 0.0) - benchmark.get(name, 0.0)
            for name in names}


def active_risk_contributions(weights: dict[str, float],
                              benchmark: dict[str, float],
                              covariance: pd.DataFrame) -> RiskContributions:
    """Decompose tracking error across active positions.

    The same arithmetic as :func:`risk_contributions`, on active weights
    ``w - b`` instead of ``w``. The reported volatility is then the annualised
    tracking error against that benchmark, and contributions sum to it exactly.

    **Contributions here can be negative, and that is the point.** An active
    weight is signed, so an underweight in something correlated with what the
    portfolio is overweight genuinely *reduces* tracking error — it hedges.
    Taking an absolute value would hide the position doing the most useful
    thing in the book.

    Args:
        weights: The portfolio's holdings.
        benchmark: What it is measured against.
        covariance: Annualised covariance over the union of both.

    Returns:
        RiskContributions: `volatility` is the tracking error; `covered_weight`
        is the share of *gross* active weight the model covers, since active
        weights sum to roughly zero and a plain sum would say nothing.
    """
    active = active_weights(weights, benchmark)
    gross = sum(abs(value) for value in active.values())

    def covered(available: list[str],
                _missing: tuple[str, ...]) -> float:
        if gross <= 0.0:
            return 0.0

        return float(sum(abs(active[name]) for name in available) / gross)

    return _decompose(active, covariance, covered, "active")
