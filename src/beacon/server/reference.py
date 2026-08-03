# src/beacon/server/reference.py
"""
Assembling a batch reference response.

`/data/reference/{identifier}` is single-name only, so a 512-member universe
table cost 512 requests. The client's answer was to truncate detail at sixty
rows and show dashes for the rest — which reads as a bug rather than as a
limit — and to drop ADV entirely, because that would have needed a *prices*
call per name on top.

Three decisions shape what this returns.

**Order is the request's order.** A table renders rows in the order it asked
for them, and a response sorted by identifier or by whatever the store happened
to hold would force the client to re-sort against its own request. Every
requested identifier gets exactly one entry at its requested position.

**A miss is an entry, not a failure.** One unknown ticker in five hundred must
not fail the batch: the table should render 499 rows and mark one unknown.
Entries carry `found`, so "we have no data for this" and "this name has no
value for that field" stay distinguishable — a distinction a client showing
dashes for both cannot make.

**Derived fields are requested by name alongside stored ones.** `adv_3m` sits
in the same `fields` list as `NAME` and `SECTOR`, so a client asks for what it
wants to display in one place and reads the answer out of one mapping. It is
opt-in because computing it means slicing the price history for every
identifier in the batch, which is work nobody should pay for by default.
"""
import logging
from typing import Any

import pandas as pd

from ..analysis.liquidity import TRAILING_MONTHS, average_daily_volume
from ..data.fetcher import DataFetcher
from ..exceptions import InvalidRuleError
from .schemas import ReferenceEntry

logger = logging.getLogger(__name__)

# Derived field name -> what it means. Not stored on any dataset; computed here
# from market data the server already holds.
ADV_3M = "adv_3m"
DERIVED_FIELDS = {
    ADV_3M: f"Mean daily VOLUME over the trailing {TRAILING_MONTHS} calendar months.",
}

# The most identifiers one request may name. Above the 512-member universe pane
# with room to spare, and low enough that a malformed client cannot ask the
# server to assemble an unbounded response. A caller needing more paginates.
MAX_BATCH = 1000


def parse_list(raw: list[str] | None) -> list[str]:
    """Split a repeatable query parameter into a clean, ordered list.

    Accepts both repetition (`?fields=A&fields=B`) and the comma-separated
    form (`?fields=A,B`), because both are natural to write and a client
    should not have to know which one this server prefers. Applied to *every*
    list parameter here rather than only to identifiers: handling one and not
    the other is how `?fields=NAME,SECTOR` ends up rejected as a single column
    literally named "NAME,SECTOR".

    Args:
        raw: Values as FastAPI parsed them.

    Returns:
        list: Entries in request order, duplicates removed.
    """
    parts: list[str] = []
    for value in raw or []:
        parts += [part.strip() for part in value.split(",") if part.strip()]

    # Deduplicated, because a repeat in the request would otherwise produce two
    # rows the client has to reconcile — order of first appearance is kept.
    return list(dict.fromkeys(parts))


def parse_identifiers(raw: list[str] | None) -> list[str]:
    """The `identifiers` parameter, validated against the batch limit.

    Args:
        raw: Values as FastAPI parsed them.

    Returns:
        list: Identifiers, in request order, with duplicates removed.

    Raises:
        InvalidRuleError: If none were supplied, or more than MAX_BATCH.
    """
    unique = parse_list(raw)

    if not unique:
        raise InvalidRuleError(
            "identifiers",
            "at least one identifier is required, e.g. "
            "?identifiers=AAA,BBB")

    if len(unique) > MAX_BATCH:
        raise InvalidRuleError(
            "identifiers",
            f"{len(unique)} identifiers requested but at most {MAX_BATCH} may "
            f"be named in one call; paginate the request")

    return unique


def _stored_fields(fetcher: DataFetcher,
                   identifiers: list[str],
                   date: str | None,
                   columns: list[str] | None) -> dict[str, dict[str, Any]]:
    """Reference rows for the batch, keyed by identifier."""
    frame = fetcher.fetch_reference_data(identifiers, date, columns)
    if frame.empty:
        return {}

    # An identifier with several validity windows can return more than one row
    # for a date-less query. The first is taken, matching the single-name
    # endpoint, rather than inventing a merge the data model does not define.
    rows: dict[str, dict[str, Any]] = {}
    for identifier, row in frame.iterrows():
        rows.setdefault(str(identifier), _clean(row))

    return rows


def _clean(row: pd.Series) -> dict[str, Any]:
    """One reference row as a JSON-safe mapping.

    Timestamps become ISO strings and NaN becomes None, so a client never has
    to recognise a float that means "absent".
    """
    fields: dict[str, Any] = {}

    for name, value in row.items():
        if isinstance(value, pd.Timestamp):
            fields[str(name)] = value.isoformat()
        elif pd.isna(value):
            fields[str(name)] = None
        else:
            fields[str(name)] = value.item() if hasattr(value, "item") else value

    return fields


def _derived_fields(fetcher: DataFetcher,
                    identifiers: list[str],
                    requested: set[str],
                    as_of: str | None) -> dict[str, dict[str, Any]]:
    """Compute the requested derived fields for the batch."""
    if ADV_3M not in requested:
        return {}

    end = pd.Timestamp(as_of) if as_of else fetcher.date_range[1]

    # One slice for the whole batch rather than one per identifier: the point
    # of this endpoint is that the client stops fanning out, and fanning out
    # inside the server instead would only move the cost.
    start = end - pd.DateOffset(months=TRAILING_MONTHS)
    market = fetcher.fetch_market_data(identifiers,
                                       start.strftime("%Y-%m-%d"),
                                       end.strftime("%Y-%m-%d"))

    volumes = average_daily_volume(market, end)

    return {identifier: {ADV_3M: float(value)}
            for identifier, value in volumes.items()
            if pd.notna(value)}


def build_entries(fetcher: DataFetcher,
                  identifiers: list[str],
                  date: str | None = None,
                  fields: list[str] | None = None) -> list[ReferenceEntry]:
    """Assemble one entry per requested identifier, in request order.

    Args:
        fetcher: The data source.
        identifiers: What to look up, already validated.
        date: Point-in-time date for reference validity.
        fields: Stored columns and derived field names to return. None returns
            every stored column and no derived field — computing ADV for a
            batch nobody asked it for would be the endpoint's whole cost paid
            by every caller.

    Returns:
        list: One `ReferenceEntry` per requested identifier, in order.

    Raises:
        InvalidRuleError: If a requested stored column is not in the dataset.
            A silently absent column would show as an empty table row and be
            read as missing data rather than as a misspelled request.
    """
    requested = set(parse_list(fields))
    derived_requested = requested & set(DERIVED_FIELDS)
    stored_requested = sorted(requested - derived_requested)

    _reject_unknown_columns(fetcher, stored_requested)

    stored = _stored_fields(fetcher, identifiers, date,
                            stored_requested or None)
    derived = _derived_fields(fetcher, identifiers, derived_requested, date)

    entries = []
    for identifier in identifiers:
        payload = dict(stored.get(identifier, {}))
        payload.update(derived.get(identifier, {}))

        # `found` keys off reference data alone. A name the server holds prices
        # but no reference data for is genuinely absent from *this* dataset,
        # and saying otherwise would put a row with no name into the table.
        entries.append(ReferenceEntry(identifier=identifier,
                                      found=identifier in stored,
                                      fields=payload))

    missing = sum(1 for entry in entries if not entry.found)
    if missing:
        logger.warning("%d of %d requested identifier(s) had no reference data.",
                       missing, len(identifiers))

    return entries


def _reject_unknown_columns(fetcher: DataFetcher,
                            columns: list[str]) -> None:
    """Fail a request naming a column the reference data does not carry."""
    available = fetcher.reference_columns
    if available is None or not columns:
        return

    unknown = sorted(set(columns) - set(available))
    if unknown:
        raise InvalidRuleError(
            "fields",
            f"unknown reference column(s): {', '.join(unknown)}. Available: "
            f"{', '.join(sorted(available))}. Derived fields: "
            f"{', '.join(sorted(DERIVED_FIELDS))}")
