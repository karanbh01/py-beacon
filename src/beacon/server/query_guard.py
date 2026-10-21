# src/beacon/server/query_guard.py
"""
A query parameter that takes one value is refused when it is given twice.

Without this, `?limit=10&limit=500` quietly answers with the last value, so a
client that built a URL wrong gets an answer to a question it did not ask. A
list parameter, such as `?fields=NAME&fields=SECTOR`, may repeat. Which is
which is read from the published OpenAPI document, so the rule is the
contract's rather than a second list kept beside it.
"""
# BN-331, found by the nightly fuzz run.
from collections import Counter
from typing import Any

from fastapi import Request
from fastapi.routing import APIRoute

from ..exceptions import InvalidArgumentError


def single_valued_query(request: Request) -> None:
    """Refuse a repeated query parameter that takes a single value.

    Raises:
        InvalidArgumentError: Naming each parameter given more than once.
    """
    counts = Counter(name for name, _ in request.query_params.multi_items())
    repeated = {name for name, count in counts.items() if count > 1}
    route = request.scope.get("route")

    if not repeated or not isinstance(route, APIRoute):
        return

    single = repeated & _single_valued(request, route)

    if single:
        raise InvalidArgumentError(
            f"Give {', '.join(f'`{name}`' for name in sorted(single))} once: "
            f"{'it takes' if len(single) == 1 else 'each takes'} a single value.")


def _single_valued(request: Request,
                   route: APIRoute) -> set[str]:
    """The route's query parameters that take one value, by the spec."""
    operation = (request.app.openapi().get("paths", {}).get(route.path_format, {})
                 .get(request.method.lower(), {}))

    return {parameter["name"] for parameter in operation.get("parameters", [])
            if parameter.get("in") == "query" and not _is_list(parameter.get("schema", {}))}


def _is_list(schema: dict[str, Any]) -> bool:
    """Whether a parameter's schema takes an array, alone or among its arms."""
    return any(arm.get("type") == "array" for arm in [schema, *schema.get("anyOf", [])])
