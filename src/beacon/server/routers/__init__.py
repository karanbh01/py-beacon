# src/beacon/server/routers/__init__.py
"""
HTTP routers, one module per resource group.

Each module exposes a ``build_router()`` that returns an APIRouter; the app
factory mounts them. Routers never construct their own data source — they read
it from application state, so a request cannot outlive or contradict the
process configuration.
"""
from .coverage import build_coverage_router
from .data import build_data_router
from .indices import build_indices_router
from .jobs import build_events_router, build_jobs_router
from .universes import build_universes_router
from .watchlists import build_watchlists_router

__all__ = [
    "build_coverage_router",
    "build_data_router",
    "build_events_router",
    "build_indices_router",
    "build_jobs_router",
    "build_universes_router",
    "build_watchlists_router",
]
