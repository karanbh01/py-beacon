# src/beacon/risk/model.py
"""
RiskModel — output of a covariance estimation.

Follows the same shape as `IndexResult` and `BacktestResult`: a dataclass
carrying pandas structures, with opt-in data binding via `.with_data()` and
accessors that answer the questions callers actually have — what is this
portfolio's volatility, how much tracking error does this active position
carry — rather than making every caller do the matrix algebra.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from .covariance import (
    PERIODS_PER_YEAR,
    Matrix,
    annualise,
    average_pairwise_correlation,
    condition_number,
    constant_correlation_target,
    correlation_from_covariance,
    eigenvalues,
    heuristic_intensity,
    is_positive_semi_definite,
    nearest_positive_semi_definite,
    sample_covariance,
    scaled_identity_target,
    shrink_covariance,
)

logger = logging.getLogger(__name__)

CONSTANT_CORRELATION = "constant_correlation"
SCALED_IDENTITY = "scaled_identity"
TARGETS = (CONSTANT_CORRELATION, SCALED_IDENTITY)

_TARGET_BUILDERS = {
    CONSTANT_CORRELATION: constant_correlation_target,
    SCALED_IDENTITY: scaled_identity_target,
}


@dataclass(frozen=True)
class RiskDiagnostics:
    """What the estimation did, and how trustworthy the result is.

    Attributes:
        observations: Periods used, after dropping incomplete rows.
        assets: Size of the cross-section.
        target: Which structured target was shrunk toward.
        intensity: Weight placed on that target, in [0, 1]. 0 means the
            estimate is the raw sample covariance.
        average_correlation: Mean off-diagonal correlation of the result.
        condition_number: Largest eigenvalue over smallest. Large values mean
            the matrix is near-singular and its inverse amplifies noise.
        smallest_eigenvalue: The most negative (or least positive) eigenvalue,
            which is what the PSD flag turns on.
        positive_semi_definite: Whether every eigenvalue is non-negative to
            within tolerance. Truthful, not asserted — a caller inverting this
            matrix needs to know.
        repaired: Whether eigenvalue clipping was applied to make it PSD.
    """
    observations: int
    assets: int
    target: str
    intensity: float
    average_correlation: float
    condition_number: float
    smallest_eigenvalue: float
    positive_semi_definite: bool
    repaired: bool = False


@dataclass
class RiskModel:
    """A covariance and correlation estimate for a set of assets.

    Attributes:
        covariance: Annualised covariance, indexed and columned by asset id.
        correlation: Correlation derived from *covariance*, unit diagonal.
        diagnostics: How the estimate was produced and how well conditioned
            it is.
        periods_per_year: The annualisation factor applied.
    """
    covariance: pd.DataFrame
    correlation: pd.DataFrame
    diagnostics: RiskDiagnostics
    periods_per_year: int = PERIODS_PER_YEAR
    _data_fetcher: DataFetcher | None = field(default=None, repr=False, compare=False)

    def with_data(self,
                  data_fetcher: DataFetcher) -> "RiskModel":
        """Bind a DataFetcher for asset-level queries. Returns self."""
        self._data_fetcher = data_fetcher
        return self

    @property
    def asset_ids(self) -> list[str]:
        """Assets covered, in matrix order."""
        return list(self.covariance.index)

    def volatilities(self) -> pd.Series:
        """Annualised volatility of each asset.

        Returns:
            pd.Series: Standard deviations, the square root of the covariance
            diagonal, indexed by asset id.
        """
        return pd.Series(np.sqrt(np.diag(self.covariance.to_numpy())),
                         index=self.covariance.index,
                         name="volatility")

    def _weight_vector(self,
                       weights: dict[str, float]) -> Matrix:
        """Align a weight mapping to the matrix's asset order.

        Assets absent from *weights* are held at zero; a weight naming an
        asset the model does not cover is an error rather than a silent drop,
        because the resulting risk number would understate the position.
        """
        unknown = sorted(set(weights) - set(self.covariance.index))
        if unknown:
            raise CalculationError(
                "PortfolioRisk",
                f"weights reference assets absent from the risk model: {unknown}.")

        return np.array([weights.get(asset_id, 0.0)
                         for asset_id in self.covariance.index])

    def portfolio_variance(self,
                           weights: dict[str, float]) -> float:
        """Annualised variance of a weighted portfolio.

        Args:
            weights: Mapping of asset id to weight. Missing assets count as
                zero.

        Returns:
            float: ``wᵀ Σ w``. Clamped at zero: a PSD covariance cannot
            produce a negative variance, so any negative value is float noise
            on a near-zero result and returning it would be nonsense.
        """
        vector = self._weight_vector(weights)
        variance = float(vector @ self.covariance.to_numpy() @ vector)

        return max(variance, 0.0)

    def portfolio_volatility(self,
                             weights: dict[str, float]) -> float:
        """Annualised volatility of a weighted portfolio.

        Args:
            weights: Mapping of asset id to weight.

        Returns:
            float: Square root of the portfolio variance.
        """
        return float(np.sqrt(self.portfolio_variance(weights)))

    def tracking_error(self,
                       portfolio_weights: dict[str, float],
                       benchmark_weights: dict[str, float]) -> float:
        """Annualised tracking error of a portfolio against a benchmark.

        The volatility of the active position — the weight differences — which
        is the quantity an index-tracking mandate is measured on.

        Args:
            portfolio_weights: Held weights.
            benchmark_weights: Target weights.

        Returns:
            float: Annualised tracking error.
        """
        active = {
            asset_id: portfolio_weights.get(asset_id, 0.0)
            - benchmark_weights.get(asset_id, 0.0)
            for asset_id in set(portfolio_weights) | set(benchmark_weights)
        }

        return self.portfolio_volatility(active)

    def eigenvalues(self) -> Matrix:
        """Ascending eigenvalues of the covariance matrix."""
        return eigenvalues(self.covariance.to_numpy())


def estimate_risk_model(returns: pd.DataFrame,
                        target: str = CONSTANT_CORRELATION,
                        intensity: float | None = None,
                        periods_per_year: int = PERIODS_PER_YEAR,
                        repair: bool = False) -> RiskModel:
    """Estimate a shrunk covariance from a returns panel.

    Args:
        returns: DataFrame of period returns, dates on the index and assets on
            the columns. Rows with any missing value are dropped, so every
            covariance entry is estimated over the same periods — pairwise
            deletion would give a matrix that need not be PSD at all.
        target: Structured target to shrink toward; one of TARGETS.
        intensity: Weight on the target in [0, 1]. None uses the heuristic
            from the panel's shape. Pass 0.0 for the raw sample covariance.
        periods_per_year: Annualisation factor; 252 for daily returns.
        repair: Apply eigenvalue clipping if the result is not PSD. Off by
            default because shrinkage should make it unnecessary, and clipping
            silently shifts the variances.

    Returns:
        RiskModel: The estimate, annualised, with diagnostics.

    Raises:
        CalculationError: If *target* is unknown, or the panel is too small.
    """
    if target not in _TARGET_BUILDERS:
        raise CalculationError(
            "RiskModel",
            f"unknown target '{target}'. Available: {', '.join(TARGETS)}.")

    complete = returns.dropna(how="any")
    if complete.empty:
        raise CalculationError(
            "RiskModel", "no complete observations remain after dropping missing rows.")

    dropped = len(returns) - len(complete)
    if dropped:
        logger.warning(
            f"Dropped {dropped} of {len(returns)} observations with missing values "
            f"before estimating the risk model.")

    panel = complete.to_numpy(dtype=float)
    observations, assets = panel.shape

    sample = sample_covariance(panel)
    structured = _TARGET_BUILDERS[target](sample)

    weight = (heuristic_intensity(observations, assets)
              if intensity is None else intensity)
    estimate = shrink_covariance(sample, structured, weight)

    estimate, repaired = _repair_if_needed(estimate, repair)
    annualised = annualise(estimate, periods_per_year)
    correlation = correlation_from_covariance(annualised)

    diagnostics = RiskDiagnostics(
        observations=observations,
        assets=assets,
        target=target,
        intensity=float(weight),
        average_correlation=average_pairwise_correlation(correlation),
        condition_number=condition_number(annualised),
        smallest_eigenvalue=float(eigenvalues(annualised)[0]),
        positive_semi_definite=is_positive_semi_definite(annualised),
        repaired=repaired)

    logger.info(
        f"Estimated risk model over {observations} observations and {assets} "
        f"assets: target={target}, intensity={weight:.4f}, "
        f"condition={diagnostics.condition_number:.1f}, "
        f"psd={diagnostics.positive_semi_definite}.")

    labels = list(complete.columns)

    return RiskModel(
        covariance=pd.DataFrame(annualised, index=labels, columns=labels),
        correlation=pd.DataFrame(correlation, index=labels, columns=labels),
        diagnostics=diagnostics,
        periods_per_year=periods_per_year)


def _repair_if_needed(covariance: Matrix,
                      repair: bool) -> tuple[Matrix, bool]:
    """Clip negative eigenvalues when asked and needed."""
    if not repair or is_positive_semi_definite(covariance):
        return covariance, False

    logger.warning(
        "Shrunk covariance was not positive semi-definite; repairing by "
        "eigenvalue clipping. Variances will shift by the clipped mass.")

    return nearest_positive_semi_definite(covariance), True
