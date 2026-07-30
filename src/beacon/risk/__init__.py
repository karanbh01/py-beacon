# src/beacon/risk/__init__.py
"""
Portfolio risk: covariance estimation, correlation, and the RiskModel object.

Distinct from `beacon.analysis.risk`, which holds scalar single-series metrics
(volatility, Sharpe, drawdown). This subpackage is about how assets move
*together* — the matrix an optimiser inverts and a tracking-error calculation
contracts against.

Needs only numpy, so it stays part of the core rather than sitting behind an
extra.
"""
from .covariance import (
    PERIODS_PER_YEAR,
    PSD_TOLERANCE,
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
from .factors import (
    MARKET_FACTOR,
    ActiveRiskDecomposition,
    FactorRiskModel,
    fit_factor_model,
    z_scores,
)
from .model import (
    CONSTANT_CORRELATION,
    SCALED_IDENTITY,
    TARGETS,
    RiskDiagnostics,
    RiskModel,
    estimate_risk_model,
)

__all__ = [
    "CONSTANT_CORRELATION",
    "MARKET_FACTOR",
    "PERIODS_PER_YEAR",
    "PSD_TOLERANCE",
    "SCALED_IDENTITY",
    "TARGETS",
    "ActiveRiskDecomposition",
    "FactorRiskModel",
    "RiskDiagnostics",
    "RiskModel",
    "annualise",
    "average_pairwise_correlation",
    "condition_number",
    "constant_correlation_target",
    "correlation_from_covariance",
    "eigenvalues",
    "estimate_risk_model",
    "fit_factor_model",
    "heuristic_intensity",
    "is_positive_semi_definite",
    "nearest_positive_semi_definite",
    "sample_covariance",
    "scaled_identity_target",
    "shrink_covariance",
    "z_scores",
]
