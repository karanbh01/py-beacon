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

**A money field is published twice, in two named currencies (BN-189).** A
market cap used to come back converted into a hard-coded USD, which was right
for a dollar index and mislabelled for any other: after BN-188 the weighting
converts into the *index's* currency, so a EUR index showed dollar caps beside
euro weights and the row that made BN-188's bug visible stopped being
comparable. Each money field now carries its local figure (`market_cap_local`,
in `local_currency`) beside the converted one (`market_cap`, in
`market_cap_currency`), and the optional `currency` parameter names what the
converted one is converted into -- USD when the caller says nothing, so no
existing caller moves. The local number is a fact about the company, the
converted one is what compares to a weight, and publishing both means no
client has to choose and none can be misled by the choice. A missing rate
nulls the converted figure only: the local one is knowable whatever the FX
situation, and nulling it would hide something the server holds.
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
MARKET_CAP = "market_cap"
FREE_FLOAT_MARKET_CAP = "free_float_market_cap"

# The currency a derived money amount is converted into when the caller names
# none.
#
# A market capitalisation is money, and since BN-128 the members of one
# universe are quoted in seven currencies. Returning raw local values alone
# would make the column unsortable and every comparison silently wrong -- a yen
# cap ranks above a dollar one on magnitude alone -- so a converted figure is
# published too, and `market_cap_currency` states what it was converted into
# rather than leaving a client to assume. USD is the default rather than the
# answer: `currency` names another (BN-189).
DEFAULT_CURRENCY = "USD"
DERIVED_CURRENCY_FIELD = "market_cap_currency"

# The local half of each money pair: the figure as the exchange reports it,
# in the instrument's own currency.
LOCAL_CURRENCY_FIELD = "local_currency"
MARKET_CAP_LOCAL = "market_cap_local"
FREE_FLOAT_MARKET_CAP_LOCAL = "free_float_market_cap_local"

# The derived fields that are money, and so come as a pair. The key is what a
# caller asks for; both halves and both currency codes come back together,
# because a client that has to request the unit separately from the number is
# a client that will one day render the number without it.
MONEY_FIELDS = {MARKET_CAP: MARKET_CAP_LOCAL,
                FREE_FLOAT_MARKET_CAP: FREE_FLOAT_MARKET_CAP_LOCAL}

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
}

# The most identifiers one request may name. Above the 512-member universe pane
# with room to spare, and low enough that a malformed client cannot ask the
# server to assemble an unbounded response. A caller needing more paginates.
MAX_BATCH = 1000

# How far back to look for the last price and share count. Long enough to
# cross a delisting or a quiet stretch, short enough that a name absent from
# the whole window is reported as having no cap rather than one from years
# ago.
LOOKBACK_DAYS = 30


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


def _market_caps(fetcher: DataFetcher,
                 identifiers: list[str],
                 requested: set[str],
                 end: pd.Timestamp,
                 currency: str) -> dict[str, dict[str, Any]]:
    """Market capitalisation per name, local and converted into *currency*.

    A point-in-time cap is a price times a share count, both of which move, so
    this is a market-data join rather than a stored attribute -- which is why
    it is derived and has to be asked for by name.
    """
    wanted = requested & set(MONEY_FIELDS)

    if not wanted:
        return {}

    # Narrowed to what the store actually has (BN-209). `fetch_market_data`
    # selects strictly, so naming a column the store lacks raises `KeyError`
    # and escapes as a bare 500 -- and `FREE_FLOAT` is optional by this
    # library's own contract, so a store with prices and share counts and no
    # free float is ordinary rather than broken. `market_cap` does not even
    # use that column; it shares this list with its sibling field, so asking
    # for one used to drag in the requirements of both.
    #
    # `adv_3m` below has always done the equivalent by passing no column list
    # at all. Narrowing is kept rather than copied away because a wide store
    # has no reason to ship every column to answer a question about three.
    available = set(fetcher.market_columns)
    columns: list[str] = [
        name for name in ("CLOSE", "SHARES_OUTSTANDING", "FREE_FLOAT")
        if name in available]
    start = (end - pd.DateOffset(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    frame = fetcher.fetch_market_data(identifiers, start,
                                      end.strftime("%Y-%m-%d"), columns)

    computed: dict[str, dict[str, Any]] = {}

    for identifier in identifiers:
        # Every key the request implies is present on every entry, null where
        # it could not be computed. A name whose row is missing entirely still
        # reports the currency it would have been converted into, because that
        # is a property of the request rather than of the name.
        entry: dict[str, Any] = dict.fromkeys(wanted)
        entry.update(dict.fromkeys(MONEY_FIELDS[field] for field in wanted))
        entry[DERIVED_CURRENCY_FIELD] = currency
        entry[LOCAL_CURRENCY_FIELD] = None

        entry.update(_cap_fields(fetcher, identifier, wanted,
                                 _rows_for(frame, identifier), end, currency))

        computed[identifier] = entry

    return computed


def _cap_fields(fetcher: DataFetcher,
                identifier: str,
                wanted: set[str],
                rows: pd.DataFrame | None,
                end: pd.Timestamp,
                currency: str) -> dict[str, Any]:
    """One name's money fields, or an empty mapping when it has none.

    Free float is applied as a multiplier rather than fetched separately: it
    lives in the same row, and reading it twice would double the work to
    produce a number that is the first one scaled.

    **An unconvertible cap is reported as unknown (BN-188).** This used to fall
    back to a rate of 1.0 and return the local figure under a
    `market_cap_currency` of USD, which is how a yen cap came to sort above a
    dollar one in the universe table. A null is a value the client already
    renders as "we do not have this"; a local number wearing a dollar label is
    one it cannot tell from a real answer. The batch still succeeds -- one name
    without a rate must not fail four hundred and ninety-nine that have one.

    **Only the converted half goes null (BN-189).** The local figure needs no
    rate to be true, so withholding it because an FX pair is absent would hide
    a number the server is holding -- and would take the cross-check against
    an external source down with it, which is the one thing a reader can do
    when the converted column is missing.
    """
    if rows is None or rows.empty:
        return {}

    # The last observation on or before the date, so a name that stopped
    # trading before it is valued at its final print rather than reported as
    # missing -- and one that never traded is.
    latest = rows.iloc[-1]
    price = latest.get("CLOSE")
    shares = latest.get("SHARES_OUTSTANDING")

    if pd.isna(price) or pd.isna(shares):
        return {}

    local_currency = _currency_of(fetcher, identifier, end)
    rate = fetcher.fx_rate_on(local_currency, currency, end)

    if rate is None:
        logger.warning(
            "No %s/%s rate on or before %s; %s's market cap is reported in "
            "%s alone, with the converted figure left unknown rather than "
            "quoted as an unconverted %s number.",
            local_currency, currency, end.date(), identifier, local_currency,
            local_currency)

    fields: dict[str, Any] = {LOCAL_CURRENCY_FIELD: local_currency}

    # Each half is scaled from its own base rather than one from the other:
    # multiplication is not associative in floating point, and converting the
    # floated local figure would move the published `free_float_market_cap` by
    # a last-place bit against what this endpoint returned before BN-189.
    local_cap = float(price) * float(shares)
    converted_cap = local_cap * rate if rate is not None else None

    if MARKET_CAP in wanted:
        fields[MARKET_CAP_LOCAL] = local_cap
        fields[MARKET_CAP] = converted_cap

    if FREE_FLOAT_MARKET_CAP in wanted:
        free_float = latest.get("FREE_FLOAT")

        # Unknown float is an unknown float-adjusted cap, not a float of one
        # (BN-209). Assuming 100% returned the FULL cap under a free-float
        # heading -- identical to `market_cap`, and indistinguishable from a
        # name that genuinely has no restricted stock. `MarketCapWeighted`
        # refuses the same gap in so many words: "using its full market cap
        # instead would weight one name on a different basis from the rest."
        # Two surfaces over one question must not answer it oppositely.
        share = float(free_float) if pd.notna(free_float) else None

        fields[FREE_FLOAT_MARKET_CAP_LOCAL] = (
            None if share is None else local_cap * share)
        fields[FREE_FLOAT_MARKET_CAP] = (
            None if share is None or converted_cap is None
            else converted_cap * share)

    return fields


def _rows_for(frame: pd.DataFrame,
              identifier: str) -> pd.DataFrame | None:
    """One instrument's rows out of a multi-identifier frame."""
    if frame.empty:
        return None

    if isinstance(frame.index, pd.MultiIndex):
        if identifier not in frame.index.get_level_values("IDENTIFIER"):
            return None

        return frame.xs(identifier, level="IDENTIFIER")

    return frame


def _currency_of(fetcher: DataFetcher,
                 identifier: str,
                 as_of: pd.Timestamp) -> str:
    """The currency an instrument is quoted in, from reference data.

    Falls back to the default rather than to whatever the request asked to
    convert into: a name with no stated currency is an unknown, and answering
    with the request's currency would make the local and converted figures
    agree by construction and report a EUR figure for a name nobody has said
    trades in euros.
    """
    reference = fetcher.fetch_reference_data(identifier,
                                             as_of.strftime("%Y-%m-%d"))

    if reference.empty or "CURRENCY" not in reference.columns:
        return DEFAULT_CURRENCY

    value = reference["CURRENCY"].iloc[0]

    return str(value).upper() if pd.notna(value) else DEFAULT_CURRENCY


def _derived_fields(fetcher: DataFetcher,
                    identifiers: list[str],
                    requested: set[str],
                    as_of: str | None,
                    currency: str) -> dict[str, dict[str, Any]]:
    """Compute the requested derived fields for the batch."""
    if not requested & set(DERIVED_FIELDS):
        return {}

    end = pd.Timestamp(as_of) if as_of else fetcher.date_range[1]
    caps = _market_caps(fetcher, identifiers, requested, end, currency)

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
            every stored column and no derived field — computing ADV for a
            batch nobody asked it for would be the endpoint's whole cost paid
            by every caller.
        currency: What the converted money fields are converted into,
            defaulting to USD (BN-189). The local figures come back in the
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
