# src/beacon/risk/covariance.py
"""
Covariance estimation and the linear algebra it needs.

A sample covariance matrix estimated from a short history is a poor risk
forecast: with fewer observations than assets it is singular, and even with
more it overstates the dispersion of eigenvalues, so the minimum-variance
direction it suggests is largely noise. Shrinking it toward a structured
target trades a little bias for a large reduction in estimation error.

Everything here works on plain numpy arrays so the numerical layer stays
separate from the pandas-shaped result object in `model.py`.
"""
import numpy as np
from numpy.typing import NDArray

from ..exceptions import CalculationError

# Every matrix and panel here is float64. Aliased so the signatures stay
# readable — strict mypy requires ndarray to be parameterised.
Matrix = NDArray[np.float64]

# Eigenvalues below this (relative to the largest) count as numerically zero
# rather than negative. A singular-but-valid matrix — which any panel with
# fewer observations than assets produces — has exact zeros in theory and tiny
# values of either sign in floating point.
PSD_TOLERANCE = 1e-10

# Trading periods per year, used to annualise a covariance estimated from
# daily returns. Covariance scales linearly with the horizon.
PERIODS_PER_YEAR = 252


def sample_covariance(returns: Matrix) -> Matrix:
    """Unbiased sample covariance of a returns panel.

    Args:
        returns: Observations × assets array of period returns.

    Returns:
        Matrix: Assets × assets covariance, symmetric and positive
        semi-definite by construction.

    Raises:
        CalculationError: If there are fewer than two observations, which
            leaves the unbiased estimator undefined.
    """
    observations, _ = _validate_panel(returns)

    if observations < 2:
        raise CalculationError(
            "SampleCovariance",
            f"at least 2 observations are required, got {observations}.")

    deviations = returns - returns.mean(axis=0)
    covariance = deviations.T @ deviations / (observations - 1)

    # Symmetrise: the product above is symmetric in exact arithmetic but can
    # differ in the last bits, and downstream eigen-decomposition assumes
    # exact symmetry.
    return _symmetrise(covariance)


def scaled_identity_target(covariance: Matrix) -> Matrix:
    """Shrinkage target assuming equal variances and zero correlation.

    The average variance on the diagonal, zero elsewhere. Maximally
    structured: it discards every estimated relationship, which makes it a
    strong anchor when the panel is very short.

    Args:
        covariance: Sample covariance.

    Returns:
        Matrix: The target, positive definite whenever the average
        variance is positive.
    """
    assets = covariance.shape[0]
    average_variance = float(np.trace(covariance)) / assets

    return _as_matrix(np.eye(assets) * average_variance)


def constant_correlation_target(covariance: Matrix) -> Matrix:
    """Shrinkage target keeping sample variances but one common correlation.

    Each asset keeps its own estimated variance; every pair is assigned the
    average sample correlation. This retains the part of the sample estimate
    that is measured comparatively well — the individual variances — while
    replacing the part that is not, the O(n²) pairwise correlations.

    Args:
        covariance: Sample covariance.

    Returns:
        Matrix: The target. Positive semi-definite whenever *covariance*
        is, because the average correlation of a PSD correlation matrix
        cannot fall below -1/(n-1).
    """
    volatilities = np.sqrt(np.diag(covariance))
    correlation = correlation_from_covariance(covariance)
    average = average_pairwise_correlation(correlation)

    assets = covariance.shape[0]
    target_correlation = np.full((assets, assets), average)
    np.fill_diagonal(target_correlation, 1.0)

    outer = np.outer(volatilities, volatilities)

    return _symmetrise(target_correlation * outer)


def heuristic_intensity(observations: int,
                        assets: int) -> float:
    """A transparent shrinkage intensity based on the panel's shape.

    ``assets / (assets + observations)``: shrink hard when assets outnumber
    observations and the sample estimate is barely identified, and lightly
    when the history is long relative to the cross-section.

    This is **not** the Ledoit-Wolf closed-form optimal intensity. That
    estimator minimises expected squared error under stated assumptions and
    needs careful derivation to implement correctly; guessing at it would
    produce a plausible number with no such guarantee. This rule is stated,
    monotone and testable, and any caller who has computed an optimal
    intensity elsewhere can pass it explicitly instead.

    Args:
        observations: Number of periods in the panel.
        assets: Number of assets.

    Returns:
        float: Intensity in (0, 1).

    Raises:
        CalculationError: If either count is not positive.
    """
    if observations <= 0 or assets <= 0:
        raise CalculationError(
            "ShrinkageIntensity",
            f"observations and assets must both be positive, got "
            f"{observations} and {assets}.")

    return assets / (assets + observations)


def shrink_covariance(sample: Matrix,
                      target: Matrix,
                      intensity: float) -> Matrix:
    """Blend a sample covariance toward a structured target.

    ``(1 - intensity) * sample + intensity * target``. Because this is a
    convex combination and both inputs are positive semi-definite, the result
    is too — shrinkage cannot introduce a negative-variance direction.

    Args:
        sample: Sample covariance.
        target: Structured target of the same shape.
        intensity: Weight on the target, in [0, 1]. 0 returns the sample
            unchanged; 1 returns the target.

    Returns:
        Matrix: The shrunk covariance.

    Raises:
        CalculationError: If *intensity* is outside [0, 1] or the shapes
            disagree.
    """
    if not 0.0 <= intensity <= 1.0:
        raise CalculationError(
            "ShrinkageIntensity", f"intensity must be in [0, 1], got {intensity}.")

    if sample.shape != target.shape:
        raise CalculationError(
            "ShrinkageIntensity",
            f"sample shape {sample.shape} does not match target {target.shape}.")

    return _symmetrise((1.0 - intensity) * sample + intensity * target)


def correlation_from_covariance(covariance: Matrix) -> Matrix:
    """Derive the correlation matrix from a covariance matrix.

    Args:
        covariance: Covariance matrix.

    Returns:
        Matrix: Correlation matrix with an exact unit diagonal. A
        zero-variance asset yields zero correlation with everything rather
        than a division by zero — it has no variation to correlate.
    """
    volatilities = np.sqrt(np.diag(covariance))
    safe = np.where(volatilities > 0.0, volatilities, 1.0)

    correlation = covariance / np.outer(safe, safe)

    # Zero-variance assets get a zero row and column, then a unit diagonal, so
    # the matrix stays a valid correlation matrix in shape if not in content.
    degenerate = volatilities <= 0.0
    if degenerate.any():
        correlation[degenerate, :] = 0.0
        correlation[:, degenerate] = 0.0

    correlation = _symmetrise(correlation)
    np.fill_diagonal(correlation, 1.0)

    return correlation


def average_pairwise_correlation(correlation: Matrix) -> float:
    """Mean of the off-diagonal correlations.

    Args:
        correlation: Correlation matrix.

    Returns:
        float: The average. 0.0 for a single asset, which has no pairs.
    """
    assets = correlation.shape[0]
    if assets < 2:
        return 0.0

    off_diagonal = correlation[~np.eye(assets, dtype=bool)]

    return float(off_diagonal.mean())


def eigenvalues(matrix: Matrix) -> Matrix:
    """Ascending eigenvalues of a symmetric matrix."""
    return np.asarray(np.linalg.eigvalsh(_symmetrise(matrix)), dtype=np.float64)


def is_positive_semi_definite(matrix: Matrix,
                              tolerance: float = PSD_TOLERANCE) -> bool:
    """Whether every eigenvalue is non-negative within tolerance.

    The tolerance is relative to the largest eigenvalue, so the answer does
    not change when the matrix is rescaled — an annualised covariance and its
    daily counterpart must agree.

    Args:
        matrix: Symmetric matrix to test.
        tolerance: Relative slack for eigenvalues that are zero in theory.

    Returns:
        bool: True when the matrix is PSD to within tolerance.
    """
    values = eigenvalues(matrix)
    scale = max(abs(float(values[-1])), 1.0)

    return bool(values[0] >= -tolerance * scale)


def condition_number(matrix: Matrix) -> float:
    """Ratio of largest to smallest eigenvalue.

    A large value means the matrix is near-singular, so its inverse — which
    any optimiser will want — amplifies estimation error. Shrinkage exists
    largely to bring this down.

    Args:
        matrix: Symmetric matrix.

    Returns:
        float: The condition number, or infinity when the smallest
        eigenvalue is zero or negative.
    """
    values = eigenvalues(matrix)
    smallest = float(values[0])

    if smallest <= 0.0:
        return float("inf")

    return float(values[-1]) / smallest


def nearest_positive_semi_definite(matrix: Matrix,
                                   minimum_eigenvalue: float = 0.0) -> Matrix:
    """Repair a matrix by clipping its negative eigenvalues.

    Decompose, floor the eigenvalues, rebuild. The result is the closest PSD
    matrix in the Frobenius sense.

    Note that clipping changes the diagonal: the variances of the repaired
    matrix differ from the original by the total clipped mass. That is the
    honest cost of the repair, and it is why this is a fallback rather than
    something to apply routinely — shrinkage keeps the estimate PSD without
    it.

    Args:
        matrix: Symmetric matrix to repair.
        minimum_eigenvalue: Floor applied to each eigenvalue. Zero yields the
            nearest PSD matrix; a small positive value yields a positive
            definite one, which is what an optimiser needing an inverse
            wants.

    Returns:
        Matrix: The repaired matrix.
    """
    values, vectors = np.linalg.eigh(_symmetrise(matrix))
    clipped = np.clip(values, minimum_eigenvalue, None)

    return _symmetrise(vectors @ np.diag(clipped) @ vectors.T)


def annualise(covariance: Matrix,
              periods_per_year: int = PERIODS_PER_YEAR) -> Matrix:
    """Scale a per-period covariance to an annual one.

    Covariance scales linearly with the horizon, so this is a multiplication —
    volatilities, being square roots, scale with its square root.

    Args:
        covariance: Per-period covariance.
        periods_per_year: Periods in a year; 252 for daily data.

    Returns:
        Matrix: The annualised covariance.

    Raises:
        CalculationError: If *periods_per_year* is not positive.
    """
    if periods_per_year <= 0:
        raise CalculationError(
            "Annualisation",
            f"periods_per_year must be positive, got {periods_per_year}.")

    return _as_matrix(covariance * periods_per_year)


def _validate_panel(returns: Matrix) -> tuple[int, int]:
    """Check a returns panel and return its shape."""
    if returns.ndim != 2:
        raise CalculationError(
            "SampleCovariance",
            f"returns must be a 2-D array of observations by assets, got "
            f"{returns.ndim} dimension(s).")

    observations, assets = returns.shape
    if assets < 1:
        raise CalculationError("SampleCovariance", "at least one asset is required.")

    if not np.isfinite(returns).all():
        raise CalculationError(
            "SampleCovariance",
            "returns contain NaN or infinite values; drop or fill them first.")

    return observations, assets


def _symmetrise(matrix: Matrix) -> Matrix:
    """Average a matrix with its transpose to remove float asymmetry."""
    return _as_matrix((matrix + matrix.T) / 2.0)


def _as_matrix(values: object) -> Matrix:
    """Pin an array expression to float64.

    Numpy's stubs type some expressions — ``np.eye(n) * scalar`` among them —
    loosely enough that strict mypy sees Any, and which expressions those are
    changes between numpy releases. Routing every return through here means a
    stub change cannot reintroduce that error one function at a time.
    """
    return np.asarray(values, dtype=np.float64)
