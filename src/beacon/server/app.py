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

from .errors import register_exception_handlers  # noqa: E402
from .routers import (  # noqa: E402
    build_coverage_router,
    build_data_router,
    build_watchlists_router,
)
from .schemas import DataSourceStatus, ErrorEnvelope, HealthResponse  # noqa: E402
from .security import verify_bearer_token  # noqa: E402
from .store import DocumentStore  # noqa: E402

# Applied to every route, so the envelope shows up in the generated OpenAPI
# rather than only at runtime.
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": ErrorEnvelope, "description": "Missing or invalid bearer token."},
    404: {"model": ErrorEnvelope, "description": "Requested data does not exist."},
    422: {"model": ErrorEnvelope, "description": "Request or rule failed validation."},
    500: {"model": ErrorEnvelope, "description": "Library error during processing."},
    501: {"model": ErrorEnvelope, "description": "Endpoint exists but is not implemented."},
    503: {"model": ErrorEnvelope, "description": "A required optional dependency is absent."},
}


def _describe_data_source(config: ServerConfig) -> DataSourceStatus:
    """Summarise the configured data source for /health.

    Returns the same shape whether or not a source is present, so the client
    can read it without branching on null.
    """
    if config.data_fetcher is None:
        return DataSourceStatus(configured=False, identifiers=0)

    return DataSourceStatus(configured=True,
                            identifiers=len(config.data_fetcher.identifiers))


def build_router() -> APIRouter:
    """Build the router carrying the routes that exist at the skeleton stage.

    Returns:
        APIRouter: Router with /health, guarded by the bearer dependency.
    """
    router = APIRouter(dependencies=[Depends(verify_bearer_token)],
                       responses=ERROR_RESPONSES)

    @router.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        config: ServerConfig = request.app.state.config

        return HealthResponse(
            status="ok",
            version=__version__,
            data_source=_describe_data_source(config),
            # Always null: DataFetcher reads straight from in-memory
            # MarketData/ReferenceData and caches nothing, so there is no age
            # to report. The field is present because the client's contract
            # expects it, and it becomes meaningful if caching is added.
            cache_age=None)

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
    app.state.watchlist_store = DocumentStore("watchlists", root=config.storage_root)

    app.add_middleware(CORSMiddleware,
                       allow_origins=list(config.cors_origins),
                       allow_origin_regex=LOCALHOST_ORIGIN_PATTERN,
                       allow_credentials=True,
                       allow_methods=["*"],
                       allow_headers=["*"])

    register_exception_handlers(app)

    app.include_router(build_router())

    # Auth and the documented error responses are applied at mount time, so a
    # router cannot end up unauthenticated by forgetting to declare them.
    guard = [Depends(verify_bearer_token)]
    for router in (build_data_router(),
                   build_watchlists_router(),
                   build_coverage_router()):
        app.include_router(router, dependencies=guard, responses=ERROR_RESPONSES)

    return app
