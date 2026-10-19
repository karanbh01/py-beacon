# src/beacon/server/parameter_schemas.py
"""
The shapes a structured backtest parameter takes, published for its editor.

A parameter whose display type is `json` names one of these in its `ref`, so a
client renders rows or a picker for it rather than a JSON box. No route returns
them, so `publish` adds them to the OpenAPI document's `components.schemas`
beside the schemas the routes reach, where every `ref` is resolved.
"""
# BN-330, asked for by beacon-ui for 0.9.3.
import datetime
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, RootModel
from pydantic.json_schema import models_json_schema

from .schemas import ConstraintRow, TypedSpec


class IdentifierList(RootModel[list[str]]):
    """Instrument identifiers, such as those `GET /data/identifiers` finds."""


class ReferenceValueList(RootModel[list[str]]):
    """Values of one reference column, matched exactly. The parameter's `column`
    names the column, and `GET /data/reference-values` lists what it holds."""


class IdentifierValues(RootModel[dict[str, float]]):
    """A number for each identifier. The parameter's `help` says what the
    numbers are, and whether they are scaled to add up to one."""


class DatedAmounts(RootModel[dict[datetime.date, float]]):
    """An amount for each date (YYYY-MM-DD), in the book's currency, positive
    in and negative out."""


class ConstraintRows(RootModel[list[ConstraintRow]]):
    """Optimiser constraints, one row each. The parameter's `options` names the
    family whose types the rows take."""


class TypedSpecs(RootModel[list[TypedSpec]]):
    """Configurable objects, each a type and its parameters. The parameter's
    `options` names the family their types come from."""


class CountryRates(RootModel[dict[str, float]]):
    """A rate for each two-letter country code, such as {"US": 0.15}, as a
    fraction."""


class FactorSignals(RootModel[dict[str, TypedSpec]]):
    """Each factor by a name of your choosing, as a signal whose scores are the
    names' exposures to it. The parameter's `options` names the family the
    signals' types come from."""


class FactorTargets(RootModel[dict[str, float]]):
    """A target active exposure, in scores, for each factor named in the
    construction's `factors`. A factor with no target is aimed at zero."""


PARAMETER_SCHEMAS: tuple[type[BaseModel], ...] = (
    IdentifierList, ReferenceValueList, IdentifierValues, DatedAmounts, ConstraintRows,
    TypedSpecs, CountryRates, FactorSignals, FactorTargets)


def parameter_schemas() -> dict[str, Any]:
    """The schemas above, and any they refer to, by name."""
    _, document = models_json_schema([(model, "validation") for model in PARAMETER_SCHEMAS],
                                     ref_template="#/components/schemas/{model}")

    return dict(document.get("$defs", {}))


def publish(app: FastAPI) -> None:
    """Add the parameter schemas to *app*'s OpenAPI document.

    A schema a route already publishes, such as `ConstraintRow`, is kept as
    the route publishes it.
    """
    build: Callable[[], dict[str, Any]] = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema

        document = build()
        schemas = document.setdefault("components", {}).setdefault("schemas", {})

        for name, schema in parameter_schemas().items():
            schemas.setdefault(name, schema)

        return document

    app.openapi = openapi  # type: ignore[method-assign]
