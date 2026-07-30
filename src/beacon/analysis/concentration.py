# src/beacon/analysis/concentration.py
"""
Concentration and drift measures for a set of index or portfolio weights.

Two questions an index owner asks constantly: how concentrated is this thing,
and how far has it wandered from its targets. Both are a few lines of
arithmetic, which is exactly why they belong in one tested place rather than
being recomputed inside whichever endpoint or report needs them.
"""
import logging
from dataclasses import dataclass

from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

# Weights are expected to sum to 1. A larger deviation than this is worth a
# warning, because it changes how the concentration figures should be read.
SUM_TOLERANCE = 1e-6

# Drifts within this relative distance of the largest count as tied, so which
# asset is reported as the worst drifter does not turn on the last bit of a
# subtraction.
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ConcentrationMetrics:
    """How concentrated a set of weights is.

    Attributes:
        assets: Number of weighted positions.
        herfindahl_index: Sum of squared weights. For weights summing to 1 this
            runs from 1/n (perfectly equal) to 1 (everything in one name).
        effective_assets: ``1 / herfindahl_index`` — the number of equally
            weighted positions that would be this concentrated. Reads more
            naturally than the index itself: "this 100-stock index behaves like
            23 equal positions".
        largest_weight: The biggest single weight.
        largest_asset_id: Which asset holds it, or None when there are no
            positions.
    """
    assets: int
    herfindahl_index: float
    effective_assets: float
    largest_weight: float
    largest_asset_id: str | None


@dataclass(frozen=True)
class DriftMetrics:
    """How far current weights have moved from their targets.

    Attributes:
        per_asset: Current minus target for every asset in either set.
            Positive means overweight.
        max_absolute: Largest absolute drift across assets.
        max_absolute_asset_id: Which asset drifted most, or None when there is
            nothing to compare.
        total_absolute: Sum of absolute drifts.
        turnover: Half of *total_absolute* — the one-way trading needed to
            return to target, since every overweight funds an underweight.
    """
    per_asset: dict[str, float]
    max_absolute: float
    max_absolute_asset_id: str | None
    total_absolute: float
    turnover: float


def herfindahl_index(weights: dict[str, float]) -> float:
    """Sum of squared weights.

    Args:
        weights: Mapping of asset id to weight, expected to sum to 1.

    Returns:
        float: The index. 0.0 for no positions. Note that squaring makes this
        blind to sign, so a short position concentrates the measure exactly as
        a long one of the same size would.
    """
    if not weights:
        return 0.0

    _warn_if_not_fully_invested(weights)

    return float(sum(weight * weight for weight in weights.values()))


def effective_number_of_assets(weights: dict[str, float]) -> float:
    """Number of equally weighted positions with the same concentration.

    Args:
        weights: Mapping of asset id to weight.

    Returns:
        float: ``1 / HHI``. 0.0 when there are no positions, or when every
        weight is zero — neither has a meaningful effective count, and
        returning 0.0 keeps callers from having to guard against a division by
        zero they cannot act on.
    """
    index = herfindahl_index(weights)

    if index <= 0.0:
        return 0.0

    return 1.0 / index


def concentration(weights: dict[str, float]) -> ConcentrationMetrics:
    """Summarise how concentrated a set of weights is.

    Args:
        weights: Mapping of asset id to weight.

    Returns:
        ConcentrationMetrics: The summary. An empty mapping yields zeros and a
        None largest asset rather than raising, since "no positions" is a
        legitimate state for an index between its inception and base date.
    """
    if not weights:
        return ConcentrationMetrics(assets=0,
                                    herfindahl_index=0.0,
                                    effective_assets=0.0,
                                    largest_weight=0.0,
                                    largest_asset_id=None)

    largest_asset_id = max(weights, key=lambda asset_id: weights[asset_id])

    return ConcentrationMetrics(
        assets=len(weights),
        herfindahl_index=herfindahl_index(weights),
        effective_assets=effective_number_of_assets(weights),
        largest_weight=float(weights[largest_asset_id]),
        largest_asset_id=largest_asset_id)


def drift_from_target(current: dict[str, float],
                      target: dict[str, float]) -> DriftMetrics:
    """Compare held weights against their targets.

    Every asset appearing in either mapping is included, treating absence as a
    zero weight — a position that has been fully sold, or one the target wants
    but the portfolio does not hold, is precisely the drift worth seeing.

    Args:
        current: Held weights.
        target: Target weights.

    Returns:
        DriftMetrics: The comparison. Empty inputs give zeros and a None asset
        rather than raising.
    """
    assets = set(current) | set(target)

    if not assets:
        return DriftMetrics(per_asset={},
                            max_absolute=0.0,
                            max_absolute_asset_id=None,
                            total_absolute=0.0,
                            turnover=0.0)

    per_asset = {
        asset_id: float(current.get(asset_id, 0.0) - target.get(asset_id, 0.0))
        for asset_id in sorted(assets)
    }

    worst = _worst_drifter(per_asset)
    total = float(sum(abs(value) for value in per_asset.values()))

    return DriftMetrics(per_asset=per_asset,
                        max_absolute=abs(per_asset[worst]),
                        max_absolute_asset_id=worst,
                        total_absolute=total,
                        turnover=total / 2.0)


def drift_history(weight_history: dict[str, dict[str, float]],
                  target: dict[str, float]) -> dict[str, DriftMetrics]:
    """Drift at each of several snapshots against one set of targets.

    Args:
        weight_history: Mapping of snapshot label (typically an ISO date) to
            the weights held at that point.
        target: The target weights to compare each snapshot against.

    Returns:
        dict: Snapshot label -> DriftMetrics, in the order the labels sort.
    """
    return {label: drift_from_target(weights, target)
            for label, weights in sorted(weight_history.items())}


def top_n_weight(weights: dict[str, float],
                 count: int) -> float:
    """Combined weight of the *count* largest positions.

    The measure a concentration limit is usually written against — "no more
    than 40% in the top five" — and not derivable from the Herfindahl index.

    Args:
        weights: Mapping of asset id to weight.
        count: How many of the largest positions to sum. Larger than the number
            of positions sums all of them.

    Returns:
        float: The combined weight.

    Raises:
        CalculationError: If *count* is not positive.
    """
    if count <= 0:
        raise CalculationError("TopNWeight", f"count must be positive, got {count}.")

    largest = sorted(weights.values(), reverse=True)[:count]

    return float(sum(largest))


def _worst_drifter(per_asset: dict[str, float]) -> str:
    """The most drifted asset, with ties broken deterministically.

    Taking a plain maximum would let floating-point noise decide. An overweight
    of ``0.35 - 0.25`` and an underweight of ``0.15 - 0.25`` are the same size
    in decimal but differ in the last bit, so a naive ``max`` picks whichever
    happens to be larger — and could pick differently on another platform or
    after an unrelated change upstream. A UI highlighting "the worst drifter"
    would flip for no visible reason.

    So: find the largest absolute drift, then take the first asset in sort
    order among those effectively tied with it.
    """
    magnitudes = {asset_id: abs(value) for asset_id, value in per_asset.items()}
    largest = max(magnitudes.values())

    tied = [asset_id for asset_id, magnitude in magnitudes.items()
            if largest - magnitude <= TIE_TOLERANCE * max(largest, 1.0)]

    return min(tied)


def _warn_if_not_fully_invested(weights: dict[str, float]) -> None:
    """Warn when weights do not sum to 1, which rescales the measures.

    Not an error: a partially invested portfolio is a real state. But HHI is
    only bounded by [1/n, 1] when the weights sum to 1, so a caller comparing
    it against a limit needs to know.
    """
    total = sum(weights.values())

    if abs(total - 1.0) > SUM_TOLERANCE:
        logger.warning(
            f"Weights sum to {total:.6f}, not 1.0. Concentration measures scale "
            f"with the square of that sum, so read them accordingly.")
