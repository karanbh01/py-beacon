# src/beacon/data/kinds.py
"""
What each series in the data is, as its reference data says.

Every market series carries its value in ``CLOSE``: an instrument's price, a
currency pair's rate, a short-term interest rate. Reference data tells them
apart:

- ``ASSET_CLASS``: ``Equity``, ``Currency`` or ``Fixed Income``. A row
  without one, or reference data without the column, is ``Equity``. Another
  value is kept as given and read like an equity: a priced series.
- ``SECURITY_TYPE``: required for a currency or fixed income row, because it
  says how ``CLOSE`` is read. ``Spot`` is a currency pair; ``Short Rate`` is
  a short-term interest rate.
- ``BASE_CURRENCY`` and ``QUOTE_CURRENCY``, on a spot pair: ``CLOSE`` is units
  of the quote currency per unit of the base. ``CURRENCY``, when given, must
  be the quote currency.
- ``CURRENCY``, on a short rate: the currency the rate is for. ``CLOSE`` is an
  annual decimal (0.0525 for 5.25%). ``TENOR`` (``ON``, ``1M``, ``3M``, ...)
  and ``DAY_COUNT`` (``ACT/360``, the default, or ``ACT/365``) are optional.
- ``TRADABLE``: whether a fund can hold the series. Spot pairs and short rates
  are references and default to not tradable; everything else defaults to
  tradable. A value in the data overrides the default.

Values are matched without regard to case. An identifier's latest reference
row decides what it is.
"""
# BN-308. Decided with Karan on 2026-10-12: market data is OHLC with every
# value in CLOSE, and reference data says what a series is, so the RATE and
# INTEREST_RATE columns and the "{FROM}{TO}" naming rule are gone.
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ASSET_CLASS = "ASSET_CLASS"
SECURITY_TYPE = "SECURITY_TYPE"
BASE_CURRENCY = "BASE_CURRENCY"
QUOTE_CURRENCY = "QUOTE_CURRENCY"
TENOR = "TENOR"
DAY_COUNT = "DAY_COUNT"
TRADABLE = "TRADABLE"

EQUITY = "Equity"
CURRENCY = "Currency"
FIXED_INCOME = "Fixed Income"
ASSET_CLASSES = (EQUITY, CURRENCY, FIXED_INCOME)

SPOT = "Spot"
SHORT_RATE = "Short Rate"

ACT_360 = "ACT/360"
ACT_365 = "ACT/365"
DAY_COUNTS = {ACT_360: 360.0, ACT_365: 365.0}

# The tenors a hedge prefers when a currency has several short rates: one
# month matches its one-month forward, and overnight is the usual fallback.
PREFERRED_TENORS = ("1M", "ON")

# A short rate above this is taken for a percent, not a decimal.
MAX_SHORT_RATE = 1.0

# The column that once marked a currency pair, now refused.
LEGACY_RATE_COLUMN = "RATE"

_TRUE = {"TRUE", "YES", "Y", "1"}
_FALSE = {"FALSE", "NO", "N", "0"}


class SeriesError(ValueError):
    """Reference data describes a series wrongly, a short rate looks like a
    percent, or market data still has a ``RATE`` column."""


@dataclass(frozen=True)
class ShortRate:
    """A currency's short-term interest rate series.

    Attributes:
        identifier: Its market identifier.
        currency: The currency the rate is for.
        tenor: Its tenor, such as ``"ON"`` or ``"1M"``, or None.
        day_count: Days in a year for its interest: 360.0 or 365.0.
    """
    identifier: str
    currency: str
    tenor: str | None
    day_count: float


@dataclass
class Kinds:
    """What each identifier in the reference data is.

    Attributes:
        pairs: Each spot pair's identifier, by (base, quote) currency.
        short_rates: Each currency's short rate, the preferred one when it
            has several.
        untradable: Identifiers a fund cannot hold.
    """
    pairs: dict[tuple[str, str], str] = field(default_factory=dict)
    short_rates: dict[str, ShortRate] = field(default_factory=dict)
    untradable: frozenset[str] = frozenset()

    @property
    def pair_identifiers(self) -> list[str]:
        """Every spot pair's identifier, sorted."""
        return sorted(self.pairs.values())

    @property
    def short_rate_identifiers(self) -> list[str]:
        """Every chosen short rate's identifier, sorted."""
        return sorted(rate.identifier for rate in self.short_rates.values())


@dataclass(frozen=True)
class Described:
    """What one reference row says its series is.

    Attributes:
        asset_class: Its asset class, ``Equity`` when not given.
        security_type: Its security type, or None.
        pair: (base, quote) for a spot pair, else None.
        short_rate: The rate it is, for a short rate, else None.
        tradable: Whether a fund can hold it.
    """
    asset_class: str
    security_type: str | None
    pair: tuple[str, str] | None
    short_rate: ShortRate | None
    tradable: bool


def describe(identifier: str,
             row: pd.Series) -> Described:
    """What *row*, a reference row of *identifier*, says its series is.

    Raises:
        SeriesError: If a currency or fixed income row has no security type, a
            spot pair lacks its currencies or is quoted in another, a short
            rate has no currency or an unknown day count, or a TRADABLE value
            is not a yes or no.
    """
    asset_class = _asset_class(row)
    security_type = _security_type(identifier, row, asset_class)
    pair = _pair(identifier, row) if (asset_class, security_type) == (CURRENCY, SPOT) else None
    rate = (_short_rate(identifier, row)
            if (asset_class, security_type) == (FIXED_INCOME, SHORT_RATE) else None)
    reference_only = pair is not None or rate is not None

    return Described(asset_class=asset_class, security_type=security_type, pair=pair,
                     short_rate=rate,
                     tradable=_tradable(identifier, row, default=not reference_only))


def classify(reference: pd.DataFrame | None) -> Kinds:
    """Read what each identifier is from *reference*, indexed by identifier.

    Raises:
        SeriesError: If a row describes its series wrongly (see `describe`),
            or two spot pairs quote the same currencies.
    """
    if reference is None or reference.empty:
        return Kinds()

    latest = reference.sort_values("DATE_FROM", kind="stable")
    latest = latest[~latest.index.duplicated(keep="last")]

    pairs: dict[tuple[str, str], str] = {}
    rates: dict[str, list[ShortRate]] = {}
    untradable: set[str] = set()

    for identifier, row in latest.iterrows():
        name = str(identifier)
        described = describe(name, row)

        if described.pair is not None:
            _add_pair(pairs, name, described.pair)

        if described.short_rate is not None:
            rates.setdefault(described.short_rate.currency, []).append(described.short_rate)

        if not described.tradable:
            untradable.add(name)

    return Kinds(pairs=pairs,
                 short_rates={currency: _preferred(currency, found)
                              for currency, found in rates.items()},
                 untradable=frozenset(untradable))


def problems(reference: pd.DataFrame) -> Iterator[tuple[int | None, str | None, str]]:
    """Everything wrong with how *reference*, a frame with an
    ``IDENTIFIER`` column, describes its series: each as (row position or
    None, column or None, message). An equity needs an ``EXCHANGE``."""
    found = False

    for position, row in reference.iterrows():
        try:
            described = describe(str(row["IDENTIFIER"]).strip(), row)
        except ValueError as error:
            found = True
            yield int(position), None, str(error)
            continue

        if described.asset_class == EQUITY and _text(row, "EXCHANGE") is None:
            found = True
            yield int(position), "EXCHANGE", "EXCHANGE is empty. An equity needs one."

    if found:
        return

    # Across rows: two spot pairs for the same currencies.
    try:
        classify(reference.set_index(reference["IDENTIFIER"].astype(str).str.strip()))
    except ValueError as error:
        yield None, None, str(error)


def short_rates_in(reference: pd.DataFrame) -> set[str]:
    """The identifiers *reference*, a frame with an ``IDENTIFIER`` column,
    describes as short rates. Rows that describe their series wrongly are
    left out: `problems` reports them."""
    found = set()

    for _, row in reference.iterrows():
        identifier = str(row.get("IDENTIFIER", "")).strip()

        try:
            if describe(identifier, row).short_rate is not None:
                found.add(identifier)
        except ValueError:
            continue

    return found


def refuse_rate_column(columns: list[str]) -> None:
    """Refuse market data that still carries a ``RATE`` column.

    Raises:
        SeriesError: If *columns* holds ``RATE``.
    """
    if LEGACY_RATE_COLUMN in columns:
        raise SeriesError(
            "Market data has a RATE column. Exchange rates are now read from "
            "CLOSE like every other series: move each pair's rates into CLOSE "
            "and add a reference row for it with ASSET_CLASS 'Currency', "
            "SECURITY_TYPE 'Spot', BASE_CURRENCY and QUOTE_CURRENCY.")


def _text(row: pd.Series,
          column: str) -> str | None:
    """A text cell, stripped, or None when the column is absent or blank."""
    value = row.get(column)

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None

    text = str(value).strip()

    return text or None


def _canonical(value: str,
               allowed: tuple[str, ...]) -> str | None:
    """*value* in its listed spelling, matched without regard to case."""
    return next((option for option in allowed if option.upper() == value.upper()), None)


def _asset_class(row: pd.Series) -> str:
    value = _text(row, ASSET_CLASS)

    if value is None:
        return EQUITY

    return _canonical(value, ASSET_CLASSES) or value


def _security_type(identifier: str,
                   row: pd.Series,
                   asset_class: str) -> str | None:
    value = _text(row, SECURITY_TYPE)

    if value is None and asset_class in (CURRENCY, FIXED_INCOME):
        raise SeriesError(f"{identifier}: a {asset_class} row needs a SECURITY_TYPE, "
                         f"such as '{SPOT if asset_class == CURRENCY else SHORT_RATE}'.")

    if value is None:
        return None

    return _canonical(value, (SPOT, SHORT_RATE)) or value


def _currency(identifier: str,
              row: pd.Series,
              column: str) -> str:
    value = _text(row, column)

    if value is None or len(value) != 3 or not value.isalpha():
        raise SeriesError(f"{identifier}: {column} must be a three-letter currency "
                         f"code, got {value!r}.")

    return value.upper()


def _pair(identifier: str,
          row: pd.Series) -> tuple[str, str]:
    base = _currency(identifier, row, BASE_CURRENCY)
    quote = _currency(identifier, row, QUOTE_CURRENCY)
    quoted_in = _text(row, "CURRENCY")

    if base == quote:
        raise SeriesError(f"{identifier}: BASE_CURRENCY and QUOTE_CURRENCY are both {base}.")

    if quoted_in is not None and quoted_in.upper() != quote:
        raise SeriesError(f"{identifier}: CURRENCY {quoted_in} is not its "
                         f"QUOTE_CURRENCY {quote}. A pair's rate is quoted in "
                         f"its quote currency.")

    return base, quote


def _add_pair(pairs: dict[tuple[str, str], str],
              identifier: str,
              pair: tuple[str, str]) -> None:
    base, quote = pair

    if pair in pairs:
        raise SeriesError(f"Two spot pairs quote {base} in {quote}: "
                         f"{pairs[(base, quote)]} and {identifier}. Keep one.")

    pairs[pair] = identifier


def _short_rate(identifier: str,
                row: pd.Series) -> ShortRate:
    currency = _currency(identifier, row, "CURRENCY")
    tenor = _text(row, TENOR)
    convention = _text(row, DAY_COUNT) or ACT_360
    canonical = _canonical(convention, tuple(DAY_COUNTS))

    if canonical is None:
        raise SeriesError(f"{identifier}: DAY_COUNT '{convention}' is not one of "
                         f"{', '.join(DAY_COUNTS)}.")

    return ShortRate(identifier=identifier, currency=currency,
                     tenor=tenor.upper() if tenor is not None else None,
                     day_count=DAY_COUNTS[canonical])


def _preferred(currency: str,
               found: list[ShortRate]) -> ShortRate:
    """The short rate a currency is read at: one month, else overnight,
    else the first by identifier. Said when there was a choice."""
    ranked = sorted(found, key=lambda rate: (
        PREFERRED_TENORS.index(rate.tenor) if rate.tenor in PREFERRED_TENORS
        else len(PREFERRED_TENORS), rate.identifier))

    if len(ranked) > 1:
        logger.info("%s has %d short rates; reading %s.", currency, len(ranked),
                    ranked[0].identifier)

    return ranked[0]


def _tradable(identifier: str,
              row: pd.Series,
              default: bool) -> bool:
    value = row.get(TRADABLE)

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return default

    if isinstance(value, bool | np.bool_):
        return bool(value)

    if isinstance(value, int | float | np.number) and value in (0, 1):
        return bool(value)

    text = str(value).strip().upper()

    if text in _TRUE:
        return True

    if text in _FALSE:
        return False

    raise SeriesError(f"{identifier}: TRADABLE '{value}' is not a yes or no.")


def pair_reference(identifier: str,
                   base: str | None = None,
                   quote: str | None = None,
                   date_from: str = "1900-01-01") -> dict[str, object]:
    """A reference row that makes *identifier* a spot currency pair.

    *base* and *quote* default to the identifier's two halves, so
    ``pair_reference("GBPUSD")`` describes pounds quoted in dollars.

    Raises:
        ValueError: If the currencies are not given and the identifier is not
            six letters.
    """
    if base is None or quote is None:
        if len(identifier) != 6 or not identifier.isalpha():
            raise ValueError(f"Give the base and quote currencies of {identifier!r}: "
                             f"they can be read only from a six-letter name.")

        base, quote = base or identifier[:3], quote or identifier[3:]

    return {"IDENTIFIER": identifier, "NAME": f"{base.upper()}/{quote.upper()}",
            "CURRENCY": quote.upper(), "DATE_FROM": date_from,
            ASSET_CLASS: CURRENCY, SECURITY_TYPE: SPOT,
            BASE_CURRENCY: base.upper(), QUOTE_CURRENCY: quote.upper()}


def short_rate_reference(identifier: str,
                         currency: str,
                         tenor: str | None = None,
                         day_count: str = ACT_360,
                         date_from: str = "1900-01-01") -> dict[str, object]:
    """A reference row that makes *identifier* *currency*'s short rate."""
    return {"IDENTIFIER": identifier, "NAME": identifier, "CURRENCY": currency.upper(),
            "DATE_FROM": date_from, ASSET_CLASS: FIXED_INCOME,
            SECURITY_TYPE: SHORT_RATE, TENOR: tenor, DAY_COUNT: day_count}
