# src/beacon/server/holding_reference.py
"""
Reference columns on the rows that name a holding, as of each row's date.

The caller chooses the columns with `reference_fields`, a comma-separated list
of reference columns. Without it a row carries a default set: the name, the
asset class, the sector and the dataset's country column. Each row's columns
are read as they stood on that row's date (see
`beacon.data.reference_as_of`), so a past holding shows the classification of
its time, not today's.

A column the dataset lacks is not a mistake when the API documents it: the
columns that say what a series is answer as the engine reads them (a missing
``ASSET_CLASS`` is ``Equity``), and the descriptive columns answer null. Any
other column the dataset lacks is refused, naming the ones it has.
"""
# BN-325. Karan's call on 2026-10-19: the classification is served on the rows,
# as of their dates, rather than joined by the app from an undated lookup.
from collections.abc import Iterable, Sequence
from typing import Annotated, Any

from .._optional import require
from ..data.fetcher import DataFetcher
from ..data.reference_as_of import KIND_COLUMNS, ReferenceAsOf
from ..exceptions import InvalidRuleError, NoDataLoadedError

require("fastapi", "The Beacon API server")

from fastapi import Query  # noqa: E402

# Descriptive columns the API documents, answered null where a dataset lacks
# them rather than refused.
DESCRIPTIVE_COLUMNS = ("SECTOR", "INDUSTRY", "SUB_INDUSTRY", "REGION", "COUNTRY",
                       "COUNTRY_DOMICILE", "COUNTRY_LISTING")
# The country column a default set carries: the first of these the dataset has.
COUNTRY_COLUMNS = ("COUNTRY_DOMICILE", "COUNTRY", "COUNTRY_LISTING")

ReferenceFieldsQuery = Annotated[
    str | None,
    Query(description="Reference columns to put on each row, comma-separated, such as "
                      "'name,sector,country_domicile', each as it stood on the row's "
                      "date. Leave out for the name, asset class, sector and country; "
                      "give an empty value for none. `GET /data/fields` lists the "
                      "reference columns.")]


def chosen_columns(fetcher: DataFetcher,
                   reference_fields: str | None) -> list[str]:
    """The reference columns a request asks for, as stored (upper case).

    Raises:
        InvalidRuleError: If a column is neither in the dataset nor one the
            API documents.
    """
    stored = set(fetcher.reference_columns or [])

    if reference_fields is None:
        country = next((column for column in COUNTRY_COLUMNS if column in stored), None)

        return [column for column in ("NAME", "ASSET_CLASS", "SECTOR", country)
                if column is not None and (column in stored or column in KIND_COLUMNS)]

    asked = list(dict.fromkeys(part.strip().upper() for part in reference_fields.split(",")
                               if part.strip()))
    known = stored | set(KIND_COLUMNS) | set(DESCRIPTIVE_COLUMNS)
    unknown = [column for column in asked if column not in known]

    if unknown:
        raise InvalidRuleError(
            "reference_fields",
            f"unknown reference column(s): {', '.join(column.lower() for column in unknown)}. "
            f"Available: {', '.join(sorted(column.lower() for column in known))}")

    return asked


class HoldingReference:
    """Reference columns for the names a response names, on their dates.

    Args:
        fetcher: The loaded data; None puts no columns on the rows.
        identifiers: Every name the response will ask about.
        columns: The chosen columns; empty puts none on the rows.
    """

    def __init__(self,
                 fetcher: DataFetcher | None,
                 identifiers: Iterable[str],
                 columns: Sequence[str]):
        self.columns = list(columns) if fetcher is not None else []
        self._lookup = (ReferenceAsOf(fetcher, identifiers, self.columns)
                        if fetcher is not None and self.columns else None)

    def of(self,
           identifier: str,
           date: str) -> dict[str, Any] | None:
        """One name's columns on *date*; None when none were chosen or the
        name had no reference row then."""
        return self._lookup.of(identifier, date) if self._lookup is not None else None

    def windows_of(self,
                   identifiers: Iterable[str],
                   start: str,
                   end: str) -> dict[str, list[dict[str, Any]]] | None:
        """Each name's windows over a span; None when none were chosen."""
        if self._lookup is None:
            return None

        return {identifier: self._lookup.windows_of(identifier, start, end)
                for identifier in identifiers}

    def map_of(self,
               identifiers: Iterable[str],
               date: str) -> dict[str, dict[str, Any] | None] | None:
        """Several names' columns on one date; None when none were chosen."""
        if self._lookup is None:
            return None

        return self._lookup.map_of(identifiers, date)


def holding_reference(fetcher: DataFetcher | None,
                      reference_fields: str | None,
                      identifiers: Iterable[str]) -> HoldingReference:
    """The reference columns a request asks for, for the names it will name.

    A response that can be read without data loaded, such as a saved run,
    carries no columns then, unless some were asked for by name.

    Raises:
        NoDataLoadedError: If columns were asked for by name and no data is
            loaded to read them from.
        InvalidRuleError: If a column is unknown (see `chosen_columns`).
    """
    if fetcher is None:
        if reference_fields:
            raise NoDataLoadedError("reference columns cannot be read")

        return HoldingReference(None, (), ())

    return HoldingReference(fetcher, identifiers, chosen_columns(fetcher, reference_fields))
