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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskContributions:
    """How a portfolio's volatility divides among its holdings.

    Attributes:
        volatility: Annualised volatility of the covered holdings, at the
            weights they are actually held.
        marginal: Per name, the change in volatility per unit of extra weight.
        contribution: Per name, its share of `volatility`. Sums to it exactly.
        covered_weight: Fraction of the index the estimate speaks for. 1.0 when
            the model covers every constituent.
        uncovered: Constituents the covariance has no row for, so a reader can
            see which names are missing rather than only how much weight is.
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
    if not weights or covariance.empty:
        return RiskContributions(volatility=0.0)

    available = [name for name in weights if name in covariance.index]
    missing = tuple(sorted(name for name in weights if name not in covariance.index))

    if missing:
        logger.warning(
            "The risk model covers %d of %d constituent(s); %.1f%% of the "
            "index is excluded from the decomposition.",
            len(available), len(weights),
            100.0 * sum(weights[name] for name in missing))

    if not available:
        return RiskContributions(volatility=0.0, uncovered=missing)

    vector = np.array([weights[name] for name in available], dtype=float)
    matrix = covariance.loc[available, available].to_numpy(dtype=float)

    variance = float(vector @ matrix @ vector)
    if variance <= 0.0:
        # A degenerate covariance — every asset with zero variance, or a matrix
        # that is not positive semi-definite. Reporting zero contributions
        # beats dividing by a volatility of zero.
        logger.warning("Portfolio variance is %.3e; no decomposition is "
                       "possible.", variance)

        return RiskContributions(volatility=0.0, uncovered=missing,
                                 covered_weight=float(vector.sum()))

    volatility = float(np.sqrt(variance))
    marginal = (matrix @ vector) / volatility

    return RiskContributions(
        volatility=volatility,
        marginal={name: float(value)
                  for name, value in zip(available, marginal, strict=True)},
        contribution={name: float(weight * value)
                      for name, weight, value
                      in zip(available, vector, marginal, strict=True)},
        covered_weight=float(vector.sum()),
        uncovered=missing)
