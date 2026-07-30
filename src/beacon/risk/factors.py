# src/beacon/risk/factors.py
"""
Factor risk models, and the decomposition of active risk they make possible.

## Why a factor model, and not just a covariance

A sample covariance says how much risk a portfolio carries. It cannot say
*why*, because it has no vocabulary for why — it is n² numbers with no
structure. A factor model imposes one:

    r = B f + ε

Each asset's return is a set of exposures ``B`` to a handful of common factors
whose returns are ``f``, plus a residual ``ε`` that is specific to that asset.
The factors are the vocabulary: a portfolio is overweight momentum, or short
size, and those are statements a person can act on.

That structure produces a covariance too:

    Σ = B F Bᵀ + D

with ``F`` the factor covariance and ``D`` the diagonal of specific variances.
The two terms are the whole point — common risk and idiosyncratic risk, cleanly
separated.

## The identity, and the condition it needs

For an active position ``a = w - b``, with active factor exposures
``x = Bᵀa``:

    TE² = aᵀΣa = aᵀ(B F Bᵀ + D)a = xᵀ F x + aᵀ D a

so squared tracking error splits exactly into a factor part and a specific
part. This is worth being precise about, because it is easy to state loosely
and get wrong: **the identity holds because Σ is defined as BFBᵀ + D, not
because B and Σ happen to be lying around together.** Take an arbitrary sample
covariance and an arbitrary exposure matrix and there is a cross term, and the
two pieces will not add up.

So the decomposition here reconciles to *this model's* tracking error, not to
the one a sample-covariance model would give for the same portfolio. Comparing
the two is informative — the gap is what the factors fail to explain — but they
are two different numbers and the identity belongs to one of them.

## Fitting

Factor returns are recovered by cross-sectional regression: for each period,
the assets' returns are regressed on their exposures, and the coefficients are
that period's factor returns. Solved by least squares rather than by inverting
``BᵀB``, so collinear exposures degrade gracefully instead of raising.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..exceptions import CalculationError
from .covariance import PERIODS_PER_YEAR, Matrix, annualise

logger = logging.getLogger(__name__)

# A factor with less cross-sectional spread than this carries no information:
# every asset has the same exposure, so it cannot distinguish between them.
# Standardising it would divide by noise.
NEGLIGIBLE_SPREAD = 1e-12

# Name of the intercept column, which stands in for whatever moves every asset
# together. Without it the named factors have to explain the market's own
# return, and their fitted returns come out contaminated by it.
MARKET_FACTOR = "market"


@dataclass(frozen=True)
class ActiveRiskDecomposition:
    """Squared tracking error split into common and specific risk.

    Attributes:
        total_variance: aᵀΣa, the active variance under this factor model.
        factor_variance: xᵀFx, the part explained by common factor exposures.
        specific_variance: aᵀDa, the part from asset-specific residuals.
        exposures: Active factor exposures x, one per factor.
        factor_contributions: Each factor's share of *factor_variance*,
            computed as exposure times marginal contribution, which sums to it
            exactly. Individual entries may be negative: a factor position that
            hedges another genuinely reduces risk, and hiding that behind an
            absolute value would misreport what the portfolio is doing.
    """
    total_variance: float
    factor_variance: float
    specific_variance: float
    exposures: pd.Series
    factor_contributions: pd.Series

    @property
    def tracking_error(self) -> float:
        """Annualised tracking error, the square root of the total."""
        return float(np.sqrt(max(self.total_variance, 0.0)))

    @property
    def factor_share(self) -> float:
        """Fraction of active variance coming from common factors.

        Returns 0.0 for a portfolio with no active risk at all, where the
        question has no answer rather than an answer of zero.
        """
        if self.total_variance <= 0.0:
            return 0.0

        return self.factor_variance / self.total_variance

    @property
    def residual(self) -> float:
        """Total minus the two parts. Zero, up to float noise, by construction."""
        return self.total_variance - self.factor_variance - self.specific_variance

    def reconciles(self,
                   tolerance: float = 1e-12) -> bool:
        """Whether the two parts account for the total."""
        return abs(self.residual) <= tolerance

    def to_frame(self) -> pd.DataFrame:
        """Per-factor exposures and contributions, largest contribution first."""
        frame = pd.DataFrame({"exposure": self.exposures,
                              "contribution": self.factor_contributions})

        return frame.reindex(frame["contribution"].abs()
                             .sort_values(ascending=False).index)


@dataclass
class FactorRiskModel:
    """A fitted factor model: exposures, factor covariance, specific variance.

    Attributes:
        exposures: The n×k matrix B, assets on the index and factors on the
            columns.
        factor_covariance: The k×k matrix F, annualised.
        specific_variance: The diagonal of D, annualised, one per asset.
        factor_returns: The fitted factor returns per period, for inspection.
        r_squared: Fraction of return variance the factors explain. Read it
            against a floor of roughly k/n rather than against zero: fitting k
            factors to an n-asset cross-section explains about that much by
            construction, because k free parameters will always fit something.
            Three factors plus a market term over twelve assets floors at about
            0.33, so only a figure well above that is evidence of structure.
        periods_per_year: The annualisation factor applied.
    """
    exposures: pd.DataFrame
    factor_covariance: pd.DataFrame
    specific_variance: pd.Series
    factor_returns: pd.DataFrame = field(repr=False)
    r_squared: float = 0.0
    periods_per_year: int = PERIODS_PER_YEAR

    @property
    def asset_ids(self) -> list[str]:
        """Assets covered, in exposure-matrix order."""
        return list(self.exposures.index)

    @property
    def factor_names(self) -> list[str]:
        """Factors, in matrix order."""
        return list(self.exposures.columns)

    def covariance(self) -> pd.DataFrame:
        """The implied asset covariance, ``B F Bᵀ + D``.

        Usable anywhere a RiskModel's covariance is, and by construction it is
        the matrix the active-risk decomposition reconciles against.
        """
        loadings = self.exposures.to_numpy(dtype=float)
        factor = self.factor_covariance.to_numpy(dtype=float)
        common = loadings @ factor @ loadings.T
        implied = common + np.diag(self.specific_variance.to_numpy(dtype=float))

        return pd.DataFrame(implied, index=self.asset_ids, columns=self.asset_ids)

    def portfolio_exposures(self,
                            weights: dict[str, float]) -> pd.Series:
        """Factor exposures of a portfolio: ``Bᵀw``.

        Args:
            weights: Mapping of asset id to weight. Assets absent from it count
                as zero.

        Returns:
            pd.Series: One exposure per factor.
        """
        vector = self._weight_vector(weights)
        loadings = self.exposures.to_numpy(dtype=float)

        return pd.Series(loadings.T @ vector,
                         index=self.factor_names,
                         name="exposure")

    def active_exposures(self,
                         weights: dict[str, float],
                         benchmark: dict[str, float]) -> pd.Series:
        """Factor exposures of the active position.

        Args:
            weights: Held weights.
            benchmark: Target weights.

        Returns:
            pd.Series: One active exposure per factor. Zero across the board
            means the portfolio takes no factor bets, whatever its holdings
            look like.
        """
        active = _active_position(weights, benchmark)

        return self.portfolio_exposures(active).rename("active_exposure")

    def decompose_active_risk(self,
                              weights: dict[str, float],
                              benchmark: dict[str, float]) -> ActiveRiskDecomposition:
        """Split squared tracking error into factor and specific risk.

        Args:
            weights: Held weights.
            benchmark: Target weights.

        Returns:
            ActiveRiskDecomposition: The two parts, which sum to the total by
            construction, plus per-factor contributions.
        """
        active = self._weight_vector(_active_position(weights, benchmark))
        loadings = self.exposures.to_numpy(dtype=float)
        factor = self.factor_covariance.to_numpy(dtype=float)
        specific = self.specific_variance.to_numpy(dtype=float)

        exposures = loadings.T @ active
        marginal = factor @ exposures

        factor_variance = float(exposures @ marginal)
        specific_variance = float(active @ (specific * active))

        return ActiveRiskDecomposition(
            total_variance=factor_variance + specific_variance,
            factor_variance=factor_variance,
            specific_variance=specific_variance,
            exposures=pd.Series(exposures, index=self.factor_names,
                                name="active_exposure"),
            factor_contributions=pd.Series(exposures * marginal,
                                           index=self.factor_names,
                                           name="contribution"))

    def _weight_vector(self,
                       weights: dict[str, float]) -> Matrix:
        """Align a weight mapping to the exposure matrix's asset order.

        A weight naming an asset the model does not cover is an error rather
        than a silent drop: the resulting risk number would understate the
        position, which is the wrong direction to be wrong in.
        """
        unknown = sorted(set(weights) - set(self.exposures.index))
        if unknown:
            raise CalculationError(
                "FactorRiskModel",
                f"weights reference assets absent from the factor model: {unknown}.")

        return np.array([weights.get(asset_id, 0.0)
                         for asset_id in self.exposures.index])


def _active_position(weights: dict[str, float],
                     benchmark: dict[str, float]) -> dict[str, float]:
    """Weight differences over the union of both sides."""
    return {asset_id: weights.get(asset_id, 0.0) - benchmark.get(asset_id, 0.0)
            for asset_id in set(weights) | set(benchmark)}


def z_scores(exposures: pd.DataFrame,
             weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Standardise raw factor values across the universe.

    Raw factor values arrive in whatever units they were measured in — a market
    cap in dollars, a book-to-price ratio, a twelve-month return — and cannot be
    compared or combined until they are on one scale. A z-score puts every
    factor in units of cross-sectional standard deviations, so an exposure of
    1.0 means the same thing whichever factor it belongs to.

    Args:
        exposures: Raw values, assets on the index and factors on the columns.
        weights: Benchmark weights to centre on, so a portfolio holding the
            benchmark scores zero on every factor. None centres on the equally
            weighted mean, which makes exposures relative to the average asset
            rather than to the market.

    Returns:
        pd.DataFrame: Standardised exposures, same shape. A factor with no
        cross-sectional spread comes back as zeros — it cannot distinguish
        between assets, so it carries no information, and dividing by its
        spread would be dividing by noise.

    Raises:
        CalculationError: If *exposures* is empty.
    """
    if exposures.empty:
        raise CalculationError("FactorExposures", "no exposures were supplied.")

    values = exposures.to_numpy(dtype=float)
    centre = _centre(values, exposures.index, weights)

    deviations = values - centre
    spread = deviations.std(axis=0, ddof=0)

    # Only divide where the spread is real; a flat factor stays at zero rather
    # than becoming ±inf.
    scaled = np.divide(deviations, spread,
                       out=np.zeros_like(deviations),
                       where=spread > NEGLIGIBLE_SPREAD)

    flat = [name for name, value in zip(exposures.columns, spread, strict=True)
            if value <= NEGLIGIBLE_SPREAD]
    if flat:
        logger.warning(
            f"Factor(s) {flat} have no cross-sectional spread and were "
            f"standardised to zero; they cannot distinguish between assets.")

    return pd.DataFrame(scaled, index=exposures.index, columns=exposures.columns)


def _centre(values: Matrix,
            assets: pd.Index,
            weights: dict[str, float] | None) -> Matrix:
    """The point each factor is measured relative to."""
    if weights is None:
        return np.asarray(values.mean(axis=0), dtype=np.float64)

    vector = np.array([weights.get(asset_id, 0.0) for asset_id in assets])
    total = float(vector.sum())

    if abs(total) <= NEGLIGIBLE_SPREAD:
        raise CalculationError(
            "FactorExposures",
            "the centring weights sum to zero, so there is no benchmark to "
            "measure exposures against.")

    return np.asarray(vector @ values / total, dtype=np.float64)


def fit_factor_model(returns: pd.DataFrame,
                     exposures: pd.DataFrame,
                     periods_per_year: int = PERIODS_PER_YEAR,
                     include_market: bool = True) -> FactorRiskModel:
    """Fit a cross-sectional factor model to a returns panel.

    Each period's asset returns are regressed on the exposures, and the
    resulting coefficients are that period's factor returns. The factor
    covariance is their covariance over time; the specific variances are the
    residuals'.

    Args:
        returns: Period returns, dates on the index and assets on the columns.
            Rows with any missing value are dropped so every factor return is
            estimated over the same cross-section.
        exposures: Loadings, assets on the index and factors on the columns.
            Standardise them with :func:`z_scores` first unless they are
            already comparable.
        periods_per_year: Annualisation factor; 252 for daily returns.
        include_market: Prepend an intercept column standing in for whatever
            moves every asset together. Without it the named factors must
            explain the market's own return as well as the differences between
            assets, and their fitted returns come out contaminated by it.

    Returns:
        FactorRiskModel: The fitted model.

    Raises:
        CalculationError: If the panel and the exposures do not cover the same
            assets, or if there are too few observations to estimate a factor
            covariance.
    """
    assets = list(exposures.index)
    missing = sorted(set(assets) - set(returns.columns))
    if missing:
        raise CalculationError(
            "FactorRiskModel",
            f"the returns panel does not cover every exposed asset: {missing}.")

    complete = returns[assets].dropna(how="any")
    if len(complete) < 2:
        raise CalculationError(
            "FactorRiskModel",
            f"at least 2 complete observations are needed to estimate a factor "
            f"covariance, found {len(complete)}.")

    loadings = _with_market(exposures) if include_market else exposures
    design = loadings.to_numpy(dtype=float)
    panel = complete.to_numpy(dtype=float)

    # One least-squares solve for the whole panel: the design matrix is the
    # same every period, so the systems differ only in their right-hand sides.
    fitted, *_ = np.linalg.lstsq(design, panel.T, rcond=None)
    residuals = panel.T - design @ fitted

    factor_covariance = annualise(np.cov(fitted, ddof=1), periods_per_year)
    specific = residuals.var(axis=1, ddof=1) * periods_per_year

    explained = _explained_variance(panel, residuals)

    logger.info(
        f"Fitted a {design.shape[1]}-factor model over {len(complete)} "
        f"observations and {len(assets)} assets: R² {explained:.4f}.")

    names = list(loadings.columns)

    return FactorRiskModel(
        exposures=loadings,
        factor_covariance=pd.DataFrame(factor_covariance, index=names, columns=names),
        specific_variance=pd.Series(specific, index=assets, name="specific_variance"),
        factor_returns=pd.DataFrame(fitted.T, index=complete.index, columns=names),
        r_squared=explained,
        periods_per_year=periods_per_year)


def _with_market(exposures: pd.DataFrame) -> pd.DataFrame:
    """Prepend a column of ones, unless one is already there."""
    if MARKET_FACTOR in exposures.columns:
        return exposures

    market = pd.DataFrame({MARKET_FACTOR: 1.0}, index=exposures.index)

    return pd.concat([market, exposures], axis=1)


def _explained_variance(panel: Matrix,
                        residuals: Matrix) -> float:
    """Fraction of return variance the factors account for.

    Measured on the pooled returns rather than averaged period by period: a
    period whose cross-section happened to be nearly flat would otherwise
    contribute an R² dominated by its own noise and swing the average around.
    """
    total = float(np.var(panel))
    if total <= NEGLIGIBLE_SPREAD:
        return 0.0

    return 1.0 - float(np.var(residuals)) / total
