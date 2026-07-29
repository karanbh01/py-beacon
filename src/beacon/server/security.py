# src/beacon/server/security.py
"""
Bearer-token authentication.

The server binds to loopback but that is not a security boundary: any process
on the machine can reach it. Every route therefore requires the token the
launcher generated, including /health.
"""
import hmac

from .._optional import require

require("fastapi", "The Beacon API server")

from fastapi import HTTPException, Request, status  # noqa: E402

_BEARER_PREFIX = "bearer "


def verify_bearer_token(request: Request) -> None:
    """Reject the request unless it carries the configured bearer token.

    Wired in as a router-level dependency, so it runs before any handler.

    Args:
        request: The incoming request; the expected token is read from
            application state, where create_app() put it.

    Raises:
        HTTPException: 401 when the Authorization header is missing,
            malformed, or carries the wrong token.
    """
    expected: str = request.app.state.auth_token
    header = request.headers.get("Authorization", "")

    if not header.lower().startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"})

    presented = header[len(_BEARER_PREFIX):].strip()

    # Constant-time comparison: a length-or-prefix-sensitive check would leak
    # the token to a local process willing to time its guesses.
    if not hmac.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"})
