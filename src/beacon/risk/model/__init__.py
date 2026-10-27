# src/beacon/risk/model/__init__.py
"""
The risk models: the sample covariance, shrunk toward a target
(`estimate_risk_model`, giving a `RiskModel`), and the factor model fitted by
cross-sectional regression (`fit_factor_model`, giving a `FactorRiskModel`).
"""
# BN-386: the factor model moved here from beacon/risk/factors.py, beside the
# sample model.
from .factor import (
    MARKET_FACTOR,
    ActiveRiskDecomposition,
    FactorRiskModel,
    fit_factor_model,
    z_scores,
)
from .sample import (
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
    "SCALED_IDENTITY",
    "TARGETS",
    "ActiveRiskDecomposition",
    "FactorRiskModel",
    "RiskDiagnostics",
    "RiskModel",
    "estimate_risk_model",
    "fit_factor_model",
    "z_scores",
]
