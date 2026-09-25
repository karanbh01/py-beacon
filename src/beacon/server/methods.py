# src/beacon/server/methods.py
"""Answer 405 for a method a fixed path does not support.

A fixed path such as `/indices/validate` sits beside a parameterised one such
as `/indices/{index_id}`. The router alone would match `PUT /indices/validate`
against the parameterised path with `index_id="validate"`, and the client would
be told its request was malformed when the truth is that the verb does not
exist there. The published spec lists every method each fixed path supports,
so this reads the spec once and, for a fixed path, refuses with 405 any method
the spec does not list for it. A fixed path with no parameterised sibling gets
the same 405 as always.
"""
# BN-131: before this, `PUT /indices/validate` fell through to
# `PUT /indices/{index_id}` and answered whatever that handler answered (a 422
# for the body, or the reserved-identifier refusal). The fuzz run found five of
# these, one per fixed path sitting beside a path parameter.
from typing import Any

from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import HTTP_STATUS_CODES, _envelope

# OPTIONS is the CORS preflight and belongs to the CORS middleware; HEAD is
# answered wherever GET is. Neither appears in the spec.
_IMPLICIT = frozenset({"OPTIONS"})


class LiteralPathMethods:
    """ASGI middleware refusing undocumented methods on fixed paths."""

    def __init__(self,
                 app: ASGIApp,
                 api: FastAPI):
        self.app = app
        self.api = api
        self._allowed: dict[str, frozenset[str]] | None = None

    def allowed(self) -> dict[str, frozenset[str]]:
        """Methods per fixed path, read from the spec on first use.

        Lazily, because routers are mounted after middleware is added, and the
        spec is complete only once they are.
        """
        if self._allowed is None:
            paths: dict[str, Any] = self.api.openapi().get("paths", {})
            self._allowed = {
                path: _with_head(frozenset(method.upper() for method in operations))
                for path, operations in paths.items()
                if "{" not in path
            }

        return self._allowed

    async def __call__(self,
                       scope: Scope,
                       receive: Receive,
                       send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in _IMPLICIT:
            await self.app(scope, receive, send)
            return

        methods = self.allowed().get(scope["path"])

        if methods is None or scope["method"] in methods:
            await self.app(scope, receive, send)
            return

        allow = ", ".join(sorted(methods))
        code = HTTP_STATUS_CODES[status.HTTP_405_METHOD_NOT_ALLOWED]
        response = JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content=_envelope(code, f"{scope['method']} is not supported on "
                                    f"{scope['path']}. Allowed: {allow}."),
            headers={"Allow": allow})

        await response(scope, receive, send)


def _with_head(methods: frozenset[str]) -> frozenset[str]:
    """Add HEAD wherever GET is, as the router itself does."""
    return methods | {"HEAD"} if "GET" in methods else methods
