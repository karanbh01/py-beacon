# src/beacon/server/errors.py
"""
Exception to HTTP mapping.

Every library exception is registered here, in one place, so a new one cannot
reach a client as an unlabelled 500. Codes are part of the API contract:
clients branch on them, so they must stay stable even if the message changes.
"""
import logging
from collections.abc import Sequence
from typing import Any

from .._optional import require
from ..exceptions import (
    BeaconError,
    CalculationError,
    ConfigurationError,
    DataNotFoundError,
    InvalidIdentifierError,
    InvalidRuleError,
    MissingDependencyError,
    ReportingError,
)
from .schemas import ErrorDetail, ErrorEnvelope

logger = logging.getLogger(__name__)

require("fastapi", "The Beacon API server")

from fastapi import FastAPI, Request, status  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

# Starlette's base class, not fastapi.HTTPException. The router raises the base
# class directly for an unrouted path or a bad method, and fastapi's subclass
# derives from it — registering on the base therefore catches both. Handling
# only the subclass leaves 404s escaping as a bare {"detail": ...}.
from starlette.exceptions import HTTPException  # noqa: E402

# Library exception -> (HTTP status, stable code). Order matters: the most
# specific subclass must come first, since lookup walks this in sequence.
EXCEPTION_MAPPING: tuple[tuple[type[BeaconError], int, str], ...] = (
    (DataNotFoundError, status.HTTP_404_NOT_FOUND, "DATA_NOT_FOUND"),
    (InvalidIdentifierError, status.HTTP_422_UNPROCESSABLE_CONTENT,
     "INVALID_IDENTIFIER"),
    (InvalidRuleError, status.HTTP_422_UNPROCESSABLE_CONTENT, "INVALID_RULE"),
    (MissingDependencyError, status.HTTP_503_SERVICE_UNAVAILABLE, "MISSING_DEPENDENCY"),
    (ConfigurationError, status.HTTP_500_INTERNAL_SERVER_ERROR, "CONFIGURATION_ERROR"),
    (CalculationError, status.HTTP_500_INTERNAL_SERVER_ERROR, "CALCULATION_ERROR"),
    (ReportingError, status.HTTP_500_INTERNAL_SERVER_ERROR, "REPORTING_ERROR"),
    # Catch-all for any BeaconError subclass added later without a mapping.
    (BeaconError, status.HTTP_500_INTERNAL_SERVER_ERROR, "BEACON_ERROR"),
)

# HTTP status -> code, for errors raised by the framework rather than the
# library (auth failures, unroutable paths, bad methods).
HTTP_STATUS_CODES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    status.HTTP_501_NOT_IMPLEMENTED: "NOT_IMPLEMENTED",
}
FALLBACK_HTTP_CODE = "HTTP_ERROR"
VALIDATION_CODE = "VALIDATION_ERROR"

# A `ValueError` that escapes a request handler.
#
# The library validates arguments with bare `ValueError` in many places --
# "end_date must be after start_date", "document id cannot be empty" -- and
# those had no handler at all, so they reached the client as a bare 500 with
# no envelope, no code and no message. Every one the first working fuzz run
# found was a rejected *input*, which is a 422.
#
# The risk is stating it plainly: a `ValueError` from a genuine internal fault
# would now be reported as the caller's fault. That is mitigated rather than
# eliminated -- the handler logs at ERROR with the traceback, so a real bug is
# still visible to whoever runs the server, and the client gets a message
# instead of the silence a bare 500 gave them. An internal fault is far more
# likely to surface as a TypeError, KeyError or AttributeError, none of which
# this touches.
ARGUMENT_CODE = "INVALID_ARGUMENT"


def classify(exc: BeaconError) -> tuple[int, str]:
    """Map a library exception to its HTTP status and stable code.

    Args:
        exc: The raised library exception.

    Returns:
        tuple: ``(http_status, code)``. Unregistered `BeaconError` subclasses
        fall through to the catch-all rather than escaping as a bare 500.
    """
    for exception_type, http_status, code in EXCEPTION_MAPPING:
        if isinstance(exc, exception_type):
            return http_status, code

    return status.HTTP_500_INTERNAL_SERVER_ERROR, "BEACON_ERROR"


def _envelope(code: str,
              message: str,
              detail: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the response body for an error."""
    return ErrorEnvelope(
        error=ErrorDetail(code=code, message=message, detail=detail)).model_dump()


def _beacon_detail(exc: BeaconError) -> dict[str, Any] | None:
    """Pull the structured attributes a library exception carries.

    Each exception subclass stores the pieces it formatted its message from;
    returning them lets a client react to the specifics without parsing prose.
    """
    fields = {
        key: value
        for key, value in vars(exc).items()
        if key != "message" and not key.startswith("_")
    }

    return fields or None


def _serialisable(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip anything from pydantic's error list that will not serialise.

    A validator that raises `ValueError` -- the documented way to reject a
    value in pydantic v2 -- has the exception object itself placed in the
    error's ``ctx``. Passing that to `JSONResponse` raises `TypeError: Object
    of type ValueError is not JSON serializable`, and because it happens
    *inside the handler for validation errors*, the client gets a 500 for
    what was a correctly detected 422.

    That made every custom field validator a latent 500. It went unnoticed
    because none of the schemas had one until BN-131 added the block-kind
    check, and it is the reason this converts rather than trusting the
    structure: the next validator should not have to know.
    """
    cleaned: list[dict[str, Any]] = []

    for error in errors:
        entry = dict(error)
        context = entry.get("ctx")

        if isinstance(context, dict):
            entry["ctx"] = {key: str(value) for key, value in context.items()}

        # `url` is a documentation link pydantic adds; the rest is already
        # strings, ints and tuples, all of which JSONResponse handles.
        cleaned.append(entry)

    return cleaned


def register_exception_handlers(app: "FastAPI") -> None:
    """Attach the handlers that put every error into the envelope.

    Args:
        app: The application to register on.
    """

    @app.exception_handler(BeaconError)
    async def handle_beacon_error(request: Request,
                                  exc: BeaconError) -> JSONResponse:
        http_status, code = classify(exc)

        return JSONResponse(status_code=http_status,
                            content=_envelope(code, str(exc), _beacon_detail(exc)))

    @app.exception_handler(HTTPException)
    async def handle_http_exception(request: Request,
                                    exc: HTTPException) -> JSONResponse:
        code = HTTP_STATUS_CODES.get(exc.status_code, FALLBACK_HTTP_CODE)

        return JSONResponse(status_code=exc.status_code,
                            content=_envelope(code, str(exc.detail)),
                            headers=exc.headers)

    @app.exception_handler(ValueError)
    async def handle_value_error(request: Request,
                                 exc: ValueError) -> JSONResponse:
        logger.error("ValueError from %s %s: %s",
                     request.method, request.url.path, exc, exc_info=exc)

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_envelope(ARGUMENT_CODE, str(exc)))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request,
                                      exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content=_envelope(VALIDATION_CODE,
                              "Request validation failed.",
                              {"errors": _serialisable(exc.errors())}))
