# src/beacon/index/capping.py
"""
Weight capping.

Capping limits how much of an index any single constituent may represent —
UCITS 5/10/40, concentration control, investability. It is orthogonal to how
the base weights were derived, so it lives here rather than inside any one
weighting scheme, and composes with all of them.

Capping is iterative, not a single pass: reducing a breaching name to the cap
redistributes its excess across the others, which can push a previously
compliant name above the cap.
"""
import logging
from dataclasses import dataclass, field

from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

# Weights are compared against the cap with this relative slack. The
# redistribution loop accumulates float error across passes, so an exact
# comparison would re-flag a name that is already at the cap and never settle.
TOLERANCE = 1e-9

# The loop provably shrinks the uncapped set each pass, so it terminates in at
# most n passes. The bound is a backstop against a bug in that reasoning: it
# raises rather than spinning forever.
MAX_PASSES = 1_000


@dataclass
class CapReport:
    """What capping did, for the layers that need to show it.

    Attributes:
        cap: The maximum weight applied, or None when no cap was requested.
        capped: Identifiers held at the cap, mapped to the weight each would
            have had uncapped. The difference is what was redistributed.
        redistributed: Total weight moved off capped names and onto the rest.
            This is the quantity a report calls "cap drag".
        passes: Iterations the loop took to settle. 0 means nothing breached.
    """
    cap: float | None = None
    capped: dict[str, float] = field(default_factory=dict)
    redistributed: float = 0.0
    passes: int = 0

    @property
    def was_capped(self) -> bool:
        """Whether any constituent was held at the cap."""
        return bool(self.capped)


def minimum_feasible_cap(count: int) -> float:
    """The smallest cap that can still distribute a full unit of weight.

    Args:
        count: Number of constituents.

    Returns:
        float: ``1 / count``. Any cap below this is impossible to satisfy —
        every name would sit at the cap and the total would still fall short
        of 1.0.
    """
    if count <= 0:
        raise ValueError("count must be positive.")

    return 1.0 / count


def apply_cap(weights: dict[str, float],
              cap: float | None) -> tuple[dict[str, float], CapReport]:
    """Cap constituent weights, redistributing the excess pro rata.

    Args:
        weights: Base weights, expected to sum to 1.0. Keys are identifiers.
        cap: Maximum weight for any one constituent, or None for no capping.
            A cap of 1.0 or above is a no-op.

    Returns:
        tuple: The capped weights (summing to 1.0) and a CapReport describing
        what happened.

    Raises:
        ValueError: If *cap* is not in (0, 1].
        CalculationError: If the cap is infeasible for this many constituents,
            or if the iteration fails to settle within MAX_PASSES.
    """
    if cap is None or not weights:
        return dict(weights), CapReport(cap=cap)

    if not 0.0 < cap <= 1.0:
        raise ValueError(f"cap must be in (0, 1], got {cap}.")

    # A cap at or above 1.0 cannot bind, and neither can one at or above the
    # largest weight. Skip the loop rather than iterating to a no-op.
    if cap >= 1.0 or max(weights.values()) <= cap * (1 + TOLERANCE):
        return dict(weights), CapReport(cap=cap)

    _reject_infeasible_cap(weights, cap)

    return _redistribute(weights, cap)


def _reject_infeasible_cap(weights: dict[str, float],
                           cap: float) -> None:
    """Raise if no allocation could satisfy this cap.

    With *n* constituents each limited to *cap*, the most weight that can be
    distributed is ``n * cap``. Below 1.0 the request is impossible, and
    returning breaching weights would be worse than refusing.
    """
    count = len(weights)
    if cap * count >= 1.0 - TOLERANCE:
        return

    minimum = minimum_feasible_cap(count)
    raise CalculationError(
        "WeightCapping",
        f"a cap of {cap:.4%} cannot be satisfied by {count} constituents: "
        f"the total would reach at most {cap * count:.4%}. The smallest "
        f"feasible cap here is {minimum:.4%}.")


def _redistribute(weights: dict[str, float],
                  cap: float) -> tuple[dict[str, float], CapReport]:
    """Run the capping loop until no constituent breaches the cap."""
    original = dict(weights)
    current = dict(weights)
    capped: set[str] = set()

    for iteration in range(1, MAX_PASSES + 1):
        breaching = [asset_id for asset_id, weight in current.items()
                     if weight > cap * (1 + TOLERANCE) and asset_id not in capped]

        if not breaching:
            return _finalise(original, current, capped, cap, iteration - 1)

        capped.update(breaching)
        for asset_id in breaching:
            current[asset_id] = cap

        _spread_excess(current, capped, cap)

    raise CalculationError(
        "WeightCapping",
        f"capping did not settle within {MAX_PASSES} passes for a cap of "
        f"{cap:.4%} across {len(weights)} constituents.")


def _spread_excess(current: dict[str, float],
                   capped: set[str],
                   cap: float) -> None:
    """Push the freed weight onto the uncapped names, pro rata. Mutates.

    When every name is capped there is nowhere to put the excess; that can
    only happen when the cap is exactly feasible, in which case the weights
    already sum to 1.0 and there is nothing to spread.
    """
    excess = 1.0 - sum(current.values())
    if abs(excess) <= TOLERANCE:
        return

    uncapped = {asset_id: weight for asset_id, weight in current.items()
                if asset_id not in capped}
    remaining = sum(uncapped.values())

    if not uncapped or remaining <= 0.0:
        return

    for asset_id, weight in uncapped.items():
        current[asset_id] = weight + excess * (weight / remaining)


def _finalise(original: dict[str, float],
              current: dict[str, float],
              capped: set[str],
              cap: float,
              passes: int) -> tuple[dict[str, float], CapReport]:
    """Build the result and its report."""
    redistributed = sum(original[asset_id] - cap for asset_id in capped)

    report = CapReport(cap=cap,
                       capped={asset_id: original[asset_id] for asset_id in sorted(capped)},
                       redistributed=redistributed,
                       passes=passes)

    if capped:
        logger.info(
            f"Capping at {cap:.4%} held {len(capped)} constituent(s) at the cap "
            f"after {passes} pass(es), redistributing {redistributed:.4%}.")

    return current, report
