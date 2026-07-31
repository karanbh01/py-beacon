# src/beacon/server/routers/__init__.py
"""
HTTP routers, one module per resource group.

Each module exposes a ``build_router()`` that returns an APIRouter; the app
factory mounts them. Routers never construct their own data source — they read
it from application state, so a request cannot outlive or contradict the
process configuration.
"""
from .beacon import build_beacon_router
from .coverage import build_coverage_router
from .data import build_data_router
from .derivatives import build_derivatives_router
from .indices import build_indices_router
from .jobs import build_events_router, build_jobs_router
from .optimise import build_optimise_router
from .reports import build_reports_router
from .risk import build_risk_router
from .universes import build_universes_router
from .watchlists import build_watchlists_router

__all__ = [
    "build_beacon_router",
    "build_coverage_router",
    "build_data_router",
    "build_derivatives_router",
    "build_events_router",
    "build_indices_router",
    "build_jobs_router",
    "build_optimise_router",
    "build_reports_router",
    "build_risk_router",
    "build_universes_router",
    "build_watchlists_router",
]
