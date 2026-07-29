# src/beacon/server/routers/__init__.py
"""
HTTP routers, one module per resource group.

Each module exposes a ``build_router()`` that returns an APIRouter; the app
factory mounts them. Routers never construct their own data source — they read
it from application state, so a request cannot outlive or contradict the
process configuration.
"""
from .data import build_data_router

__all__ = ["build_data_router"]
