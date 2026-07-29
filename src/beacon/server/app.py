# src/beacon/server/app.py
"""
Application factory for the Beacon API server.

The server is a local process owned by a desktop client: it binds loopback,
authenticates every route with a bearer token the client generated, and holds
no state of its own beyond the data source it was handed.
"""
from typing import Any

from .. import __version__
from .._optional import require
from .config import LOCALHOST_ORIGIN_PATTERN, ServerConfig

require("fastapi", "The Beacon API server")

from fastapi import APIRouter, Depends, FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .security import verify_bearer_token  # noqa: E402


def _describe_data_source(config: ServerConfig) -> dict[str, Any]:
    """Summarise the configured data source for /health.

    Returns a shape that is the same whether or not a source is present, so
    the client can read it without branching on null.
    """
    if config.data_fetcher is None:
        return {"configured": False, "identifiers": 0}

    return {"configured": True, "identifiers": len(config.data_fetcher.identifiers)}


def build_router() -> APIRouter:
    """Build the router carrying the routes that exist at the skeleton stage.

    Returns:
        APIRouter: Router with /health, guarded by the bearer dependency.
    """
    router = APIRouter(dependencies=[Depends(verify_bearer_token)])

    @router.get("/health")
    def health(request: Request) -> dict[str, Any]:
        config: ServerConfig = request.app.state.config

        return {
            "status": "ok",
            "version": __version__,
            "data_source": _describe_data_source(config),
            # Always null: DataFetcher reads straight from in-memory
            # MarketData/ReferenceData and caches nothing, so there is no age
            # to report. The field is present because the client's contract
            # expects it, and it becomes meaningful if caching is added.
            "cache_age": None,
        }

    return router


def create_app(config: ServerConfig) -> FastAPI:
    """Build the ASGI application for a given configuration.

    Args:
        config: Settings for this process, including the bearer token every
            route will require and the data source to serve.

    Returns:
        FastAPI: The configured application. Nothing is bound or started
        here — see beacon.server.__main__ for the launcher.
    """
    # No default_response_class: the issue called for an orjson response class,
    # but FastAPI deprecated ORJSONResponse — it now serialises straight to
    # JSON bytes whenever a handler declares a return type, which is faster
    # than routing through a custom class. Every handler here is annotated, so
    # that fast path applies. orjson stays in the server extra for the manual
    # serialisation the WebSocket events will need.
    app = FastAPI(title="Beacon API",
                  version=__version__)

    # Handlers reach these through request.app.state rather than a closure,
    # so the app remains introspectable and testable without rebuilding it.
    app.state.config = config
    app.state.auth_token = config.auth_token

    app.add_middleware(CORSMiddleware,
                       allow_origins=list(config.cors_origins),
                       allow_origin_regex=LOCALHOST_ORIGIN_PATTERN,
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"])

    app.include_router(build_router())

    return app
