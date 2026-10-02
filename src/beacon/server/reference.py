# src/beacon/server/reference.py
"""
Assembling a batch reference response.

Reference data for many identifiers in one request, so a universe table does
not need one call per name.

**Order is the request's order.** A table renders rows in the order it asked
for them, and a response sorted by identifier or by whatever the store happened
to hold would force the client to re-sort against its own request. Every
requested identifier gets exactly one entry at its requested position.

**A miss is an entry, not a failure.** One unknown ticker in five hundred must
not fail the batch: the table should render 499 rows and mark one unknown.
Entries carry `found`, so "we have no data for this" and "this name has no
value for that field" stay distinguishable.

**Derived fields are requested by name alongside stored ones.** `adv_3m` sits
in the same `fields` list as `NAME` and `SECTOR`, so a client asks for what it
wants to display in one place and reads the answer out of one mapping. It is
opt-in because computing it means slicing the price history for every
identifier in the batch, which is work nobody should pay for by default.

**A money field is published twice, in two named currencies.** Each money field
carries its local figure (`market_cap_local`, in `local_currency`) beside the
converted one (`market_cap`, in `market_cap_currency`), and the optional
`currency` parameter names what the converted one is converted into, USD when
the caller says nothing. The local number is a fact about the company, the
converted one is what compares to a weight, and publishing both means no
client has to choose and none can be misled by the choice. A missing rate
nulls the converted figure only: the local one is knowable whatever the FX
situation, and nulling it would hide something the server holds.
"""
# History.
#
# `/data/reference/{identifier}` is single-name only, so a 512-member universe
# table cost 512 requests. The client's answer was to truncate detail at sixty
# rows and show dashes for the rest (which reads as a bug rather than as a
# limit) and to drop ADV entirely, because that would have needed a *prices*
# call per name on top. A client showing dashes for both a miss and an empty
# field could not tell them apart, hence `found`.
#
# Two currencies since BN-189. A market cap used to come back converted into a
# hard-coded USD, which was right for a dollar index and mislabelled for any
# other: after BN-188 the weighting converts into the *index's* currency, so a
# EUR index showed dollar caps beside euro weights and the row that made
# BN-188's bug visible stopped being comparable. USD stays the default so no
# existing caller moved.
import logging
from typing import Any

import pandas as pd

from ..analysis.liquidity import TRAILING_MONTHS, average_daily_volume
from ..data.fetcher import DataFetcher
from ..exceptions import InvalidRuleError
from .market_caps import (
    DEFAULT_CURRENCY,
    DERIVED_CURRENCY_FIELD,
    FREE_FLOAT_MARKET_CAP,
    FREE_FLOAT_MARKET_CAP_LOCAL,
    LOCAL_CURRENCY_FIELD,
    LOOKBACK_DAYS,
    MARKET_CAP,
    MARKET_CAP_LOCAL,
    PRICE_IS_STALE_FIELD,
    PRICED_FROM_FIELD,
    market_caps,
)
from .schemas import ReferenceEntry

logger = logging.getLogger(__name__)

# Derived field name -> what it means. Not stored on any dataset; computed here
# from market data the server already holds.
ADV_3M = "adv_3m"

# What a caller may name in `fields`. Descriptions say which currency a figure
# is in and name its counterpart (BN-181) -- and they say it by naming the
# parameter, not a currency: the unit is per-request since BN-189, so a
# description that interpolated one would be wrong for every caller who names
# another. The default is stated because a caller who names nothing still
# needs to know what they got.
DERIVED_FIELDS = {
    ADV_3M: f"Mean daily VOLUME over the trailing {TRAILING_MONTHS} calendar months.",
    MARKET_CAP: (f"Price x shares outstanding, converted into the `currency` "
                 f"the request names ({DEFAULT_CURRENCY} by default) and "
                 f"stated in `{DERIVED_CURRENCY_FIELD}`; null when that rate "
                 f"is unknown. The unconverted figure is `{MARKET_CAP_LOCAL}`, "
                 f"in `{LOCAL_CURRENCY_FIELD}`."),
    FREE_FLOAT_MARKET_CAP: (
        f"Market cap x free float, converted into the `currency` the request "
        f"names ({DEFAULT_CURRENCY} by default) and stated in "
        f"`{DERIVED_CURRENCY_FIELD}`; null when that rate is unknown. The "
        f"unconverted figure is `{FREE_FLOAT_MARKET_CAP_LOCAL}`, in "
        f"`{LOCAL_CURRENCY_FIELD}`."),
}

# Published alongside a money field rather than requested: asking for
# `market_cap` returns all four keys. They are not in DERIVED_FIELDS because
# that is the set a caller may name, and a companion that could be requested
# on its own would be a second way to ask the same question -- and a fifth
# name for the field picker, the catalogue and the expression namespace to
# learn. Documented here so nothing has to infer them from the payload.
COMPANION_FIELDS = {
    DERIVED_CURRENCY_FIELD: (f"ISO code the converted money fields are in: "
                             f"the request's `currency`, {DEFAULT_CURRENCY} "
                             f"by default."),
    LOCAL_CURRENCY_FIELD: ("ISO code the `_local` money fields are in: the "
                           "instrument's own quote currency."),
    MARKET_CAP_LOCAL: (f"Price x shares outstanding, unconverted, in "
                       f"`{LOCAL_CURRENCY_FIELD}` -- what the exchange "
                       f"reports. Its converted counterpart is "
                       f"`{MARKET_CAP}`."),
    FREE_FLOAT_MARKET_CAP_LOCAL: (
        f"Market cap x free float in `{LOCAL_CURRENCY_FIELD}`, unconverted. "
        f"Its converted counterpart is `{FREE_FLOAT_MARKET_CAP}`."),
    PRICED_FROM_FIELD: (
        f"The date the close and share count behind every money field on this "
        f"entry were printed, YYYY-MM-DD -- **not** the `date` the request "
        f"asked about. They differ for a name that has not traded recently, "
        f"which is valued at its last print rather than reported as unknown. "
        f"Null when the instrument has no price anywhere in the data, which is "
        f"a real absence rather than a stale one. `{PRICE_IS_STALE_FIELD}` is "
        f"the same fact as a flag."),
    PRICE_IS_STALE_FIELD: (
        f"Whether `{PRICED_FROM_FIELD}` is more than {LOOKBACK_DAYS} days "
        f"before the requested date. Published rather than left to be derived: "
        f"a client should not have to subtract two dates to learn that a "
        f"figure is months old, and an index can carry a real weight in a name "
        f"whose last trade is long past. Null when there is no price at all."),
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
                    as_of: str | None,
                    currency: str) -> dict[str, dict[str, Any]]:
    """Compute the requested derived fields for the batch."""
    if not requested & set(DERIVED_FIELDS):
        return {}

    end = pd.Timestamp(as_of) if as_of else fetcher.date_range[1]
    caps = market_caps(fetcher, identifiers, requested, end, currency)

    if ADV_3M not in requested:
        return caps

    # One slice for the whole batch rather than one per identifier: the point
    # of this endpoint is that the client stops fanning out, and fanning out
    # inside the server instead would only move the cost.
    start = end - pd.DateOffset(months=TRAILING_MONTHS)
    market = fetcher.fetch_market_data(identifiers,
                                       start.strftime("%Y-%m-%d"),
                                       end.strftime("%Y-%m-%d"))

    volumes = average_daily_volume(market, end)

    # Present for every identifier asked about, `null` where it cannot be
    # computed. Dropping the key instead produced an entry that reported
    # `found: true` and then silently lacked a field the caller had
    # explicitly requested -- which a client has to defend against with a
    # membership test rather than a null check.
    #
    # It became reachable with BN-130: a name delisted before the window has
    # no volume in it, so there is nothing to average. That is an answer, not
    # an absence of one.
    computed = {identifier: (float(value) if pd.notna(value) else None)
                for identifier, value in volumes.items()}

    return {identifier: {ADV_3M: computed.get(identifier),
                         **caps.get(identifier, {})}
            for identifier in identifiers}


def parse_currency(raw: str | None) -> str:
    """The `currency` parameter, validated, or the default when absent.

    Raises:
        InvalidRuleError: If it is not a three-letter code. An unknown-but-
            well-formed code converts to nothing and reports null, which is a
            data answer; "dollars" is a request that cannot be honoured, and
            silently nulling every converted figure for it would look like
            missing FX data rather than a typo.
    """
    if raw is None or not raw.strip():
        return DEFAULT_CURRENCY

    code = raw.strip().upper()

    if len(code) != 3 or not code.isalpha():
        raise InvalidRuleError(
            "currency",
            f"'{raw}' is not a currency code; name a three-letter ISO code "
            f"such as {DEFAULT_CURRENCY} or EUR. It selects what the "
            f"converted money fields are converted into; the local figures "
            f"are returned either way.")

    return code


def build_entries(fetcher: DataFetcher,
                  identifiers: list[str],
                  date: str | None = None,
                  fields: list[str] | None = None,
                  currency: str = DEFAULT_CURRENCY) -> list[ReferenceEntry]:
    """Assemble one entry per requested identifier, in request order.

    Args:
        fetcher: The data source.
        identifiers: What to look up, already validated.
        date: Point-in-time date for reference validity.
        fields: Stored columns and derived field names to return. None returns
            every stored column and no derived field: computing ADV for a
            batch nobody asked it for would be the endpoint's whole cost paid
            by every caller.
        currency: What the converted money fields are converted into,
            defaulting to USD. The local figures come back in the
            instrument's own currency whatever this says, so a caller that
            names nothing still gets both numbers and both labels.

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
    derived = _derived_fields(fetcher, identifiers, derived_requested, date,
                              currency)

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
