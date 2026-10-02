# src/beacon/server/market_caps.py
"""
Each name's market cap at a date, local and converted, and how old it is.

A point-in-time cap is a price times a share count, both of which move, so it
is a market-data join rather than a stored attribute. `/data/reference`
publishes it as a derived field and the index preview beside each
constituent; both call `market_caps`, so the two never disagree.
"""
# Moved out of reference.py in BN-278, when the preview started publishing
# caps too.
import logging
from typing import Any

import pandas as pd

from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

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

# When the observation behind every money field on this entry was printed, and
# whether that is old enough to be worth saying so (BN-210). Published
# alongside the caps rather than requested, on the same terms as the currency
# fields: a figure whose age a reader has to infer is a figure they will
# eventually mis-read, and the null this replaced said "unknown" for a name
# whose value was perfectly well known and simply not recent.
PRICED_FROM_FIELD = "priced_from"
PRICE_IS_STALE_FIELD = "price_is_stale"
MARKET_CAP_LOCAL = "market_cap_local"
FREE_FLOAT_MARKET_CAP_LOCAL = "free_float_market_cap_local"

# The derived fields that are money, and so come as a pair. The key is what a
# caller asks for; both halves and both currency codes come back together,
# because a client that has to request the unit separately from the number is
# a client that will one day render the number without it.
MONEY_FIELDS = {MARKET_CAP: MARKET_CAP_LOCAL,
                FREE_FLOAT_MARKET_CAP: FREE_FLOAT_MARKET_CAP_LOCAL}

# What counts as a recent price, in days (BN-210).
#
# Two jobs, and it used to have only the first. It is the window read for the
# last price and share count -- the one nearly every name is answered from --
# and it is the threshold above which a price is reported as stale.
#
# What it is no longer is a *bound on the answer*. A name absent from the
# window used to be reported as having no market cap at all, which put a blank
# beside a weight the index had computed perfectly well from the same missing
# name's last print. The cap is now computed from whatever price exists and
# carries the date it came from; this number decides whether that date is worth
# flagging, not whether the figure is published.
LOOKBACK_DAYS = 30


def market_caps(fetcher: DataFetcher,
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

    # One row per name, read in two stages and reduced once. Lives on the
    # fetcher since BN-211, because the staleness gate needs the same read
    # and two copies of it would be two things to keep in step.
    frame = fetcher.latest_rows(identifiers, end, columns,
                                recent_days=LOOKBACK_DAYS)

    # BN-277: the rows, currencies and FX rates are each read once for the
    # batch. Read per name they were 80% of a 1,000-name request (1.8s): a
    # reference lookup and an `xs` on the frame for every name.
    latest = _latest_by_name(frame, identifiers)
    currencies = _currencies_of(fetcher, identifiers, end)
    rates: dict[str, float | None] = {}

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
        entry[PRICED_FROM_FIELD] = None
        entry[PRICE_IS_STALE_FIELD] = None

        row = latest.get(identifier)

        if row is not None:
            local_currency = currencies.get(identifier, DEFAULT_CURRENCY)

            if local_currency not in rates:
                rates[local_currency] = fetcher.fx_rate_on(local_currency,
                                                           currency, end)

            entry.update(_cap_fields(identifier, wanted, row, end, currency,
                                     local_currency, rates[local_currency]))

        computed[identifier] = entry

    return computed




def _cap_fields(identifier: str,
                wanted: set[str],
                row: tuple[pd.Timestamp, dict[str, Any]],
                end: pd.Timestamp,
                currency: str,
                local_currency: str,
                rate: float | None) -> dict[str, Any]:
    """One name's money fields from its latest *row* (the day it was printed
    and its values), or an empty mapping when it has no price or share count.

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
    # The last observation on or before the date, so a name that stopped
    # trading before it is valued at its final print rather than reported as
    # missing -- and one that never traded is.
    priced_from, latest = row
    price: Any = latest.get("CLOSE")
    shares: Any = latest.get("SHARES_OUTSTANDING")

    if pd.isna(price) or pd.isna(shares):
        return {}

    # The day the numbers above were printed, which is not necessarily the day
    # the caller asked about (BN-210). A cap computed from a three-month-old
    # close is a true statement about a company that has not traded since; it
    # is only misleading if nothing says how old it is.
    stale = (end - priced_from).days > LOOKBACK_DAYS

    if rate is None:
        logger.warning(
            "No %s/%s rate on or before %s; %s's market cap is reported in "
            "%s alone, with the converted figure left unknown rather than "
            "quoted as an unconverted %s number.",
            local_currency, currency, end.date(), identifier, local_currency,
            local_currency)

    fields: dict[str, Any] = {
        LOCAL_CURRENCY_FIELD: local_currency,
        PRICED_FROM_FIELD: priced_from.date().isoformat(),
        PRICE_IS_STALE_FIELD: stale,
    }

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
        free_float: Any = latest.get("FREE_FLOAT")

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


def _latest_by_name(frame: pd.DataFrame,
                    identifiers: list[str]
                    ) -> dict[str, tuple[pd.Timestamp, dict[str, Any]]]:
    """Each name's latest row as (date, values), split out of the frame in
    one pass. A frame without an identifier level holds one name's rows,
    and answers for every name asked about."""
    if frame.empty:
        return {}

    if not isinstance(frame.index, pd.MultiIndex):
        last = (pd.Timestamp(frame.index[-1]), frame.iloc[-1].to_dict())

        return dict.fromkeys(identifiers, last)

    # Sorted by date within each name, so a later row replaces an earlier.
    return {str(name): (pd.Timestamp(date), values)
            for (name, date), values
            in frame.sort_index().to_dict("index").items()}


def _currencies_of(fetcher: DataFetcher,
                   identifiers: list[str],
                   as_of: pd.Timestamp) -> dict[str, str]:
    """The currency each instrument is quoted in, from reference data.

    A name with no stated currency is absent, and its caller falls back to
    the default rather than to whatever the request asked to convert into:
    a name with no stated currency is an unknown, and answering with the
    request's currency would make the local and converted figures agree by
    construction and report a EUR figure for a name nobody has said trades
    in euros.
    """
    reference = fetcher.fetch_reference_data(identifiers,
                                             as_of.strftime("%Y-%m-%d"))

    if reference.empty or "CURRENCY" not in reference.columns:
        return {}

    # The first row per name, which is what a one-name read used.
    stated = reference.loc[~reference.index.duplicated(keep="first"), "CURRENCY"]

    return {str(name): str(value).upper()
            for name, value in stated.items() if pd.notna(value)}
