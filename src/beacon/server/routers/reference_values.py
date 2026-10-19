# src/beacon/server/routers/reference_values.py
"""
`GET /data/reference-values`: the values a reference column holds.

What a picker for a list of reference values offers, such as the sectors an
exclusion screen may exclude: a parameter's `column` names the column.
"""
# BN-330. Its own module because `routers/data.py` is already far past the
# size this project keeps a module to.
from typing import Annotated

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from ...exceptions import DataNotFoundError
from ..active_data import require_data


class ReferenceValues(BaseModel):
    """Response of `GET /data/reference-values`."""
    column: str = Field(description="The column, as the reference data names it.")
    values: list[str] = Field(
        description="Every value it holds in any record, current or not, sorted. A "
                    "screen matches them exactly.")


def build_reference_values_router() -> APIRouter:
    """The reference-values route, under `/data`."""
    router = APIRouter(prefix="/data", tags=["data"])

    @router.get("/reference-values", response_model=ReferenceValues)
    def reference_values(request: Request,
                         column: Annotated[str, Query(
                             min_length=1,
                             description="The reference column, such as 'SECTOR'. "
                                         "Matched without regard to case.")]
                         ) -> ReferenceValues:
        """The values one reference column holds."""
        reference = require_data(request, "its reference values cannot be listed").reference
        frame = reference.data if reference is not None else None
        named = [name for name in (frame.columns if frame is not None else [])
                 if str(name).upper() == column.upper()]

        if frame is None or not named:
            raise DataNotFoundError(f"reference column '{column}'", source="ReferenceData")

        values = {str(value) for value in frame[named[0]].dropna().unique()}

        return ReferenceValues(column=str(named[0]), values=sorted(values))

    return router
