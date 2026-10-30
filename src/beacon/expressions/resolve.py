# src/beacon/expressions/resolve.py
"""
Turning an expression into an answer for one instrument on one date.

An expression is a description; resolving it needs an instrument and a date,
and in an index that date is the **rebalance**.

## Point in time

A screen on `data.features.fundamentals.revenue` at a rebalance on 1 April
must see what was published by 1 April. Q1 revenue announced in mid-May is
invisible on that date, however completely the quarter had ended.

Every feature read here goes through `DataFetcher.fetch_feature`, which is the
accessor that enforces this. Reading the table directly would put look-ahead
back in, and the resulting backtest would look *better* and be wrong: the
failure nobody catches, because a better number is not a symptom anybody
investigates.

Market and reference data are read as of the same date for the same reason. A
market column takes its last value on or before the date (looking back at most
10 days), so a rebalance on a day the instrument did not trade still sees its
latest value.

## Missing is not zero

A name with no value for a field yields `None`, and `on_missing` decides what
the comparison answers. This is deliberately distinct from a value that is
legitimately zero: zero fails a `> 0` test honestly, where missing has nothing
to compare at all. Collapsing the two would let a screen for "revenue above a
billion" quietly admit every company the dataset has never heard of.
"""
# Point-in-time feature reads through `fetch_feature` are BN-135.
import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.liquidity import TRAILING_MONTHS, average_daily_volume
from ..data.features import MAX_AGE_DAYS
from ..data.fetcher import DataFetcher
from ..exceptions import ExpressionError
from .core import (
    ALL,
    ANY,
    BETWEEN,
    EQ,
    GE,
    GT,
    IN,
    LE,
    LT,
    NE,
    NOT,
    Comparison,
    Expression,
    Field,
    Not,
    _Group,
)
from .namespaces import (
    FEATURES,
    MARKET,
    REFERENCE,
    column_for,
)

logger = logging.getLogger(__name__)

# Derived market fields, computed here rather than read.
ADV_3M = "adv_3m"
MARKET_CAP = "market_cap"
FREE_FLOAT_MARKET_CAP = "free_float_market_cap"

# A stored market field, but read through the fetcher's free float backfill.
FREE_FLOAT_FIELD = "free_float"

# Derived money amounts are reported in one currency, matching the reference
# endpoint, so a cap comparison is not a currency comparison.
BASE_CURRENCY = "USD"

CLOSE = "CLOSE"
SHARES = "SHARES_OUTSTANDING"
FREE_FLOAT = "FREE_FLOAT"

# How a comparison is applied once both sides are in hand. Annotated rather
# than inferred: bare lambdas read as untyped, and calling one from a typed
# function is an error under strict mode.
OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    GT: lambda value, other: value > other,
    GE: lambda value, other: value >= other,
    LT: lambda value, other: value < other,
    LE: lambda value, other: value <= other,
    EQ: lambda value, other: value == other,
    NE: lambda value, other: value != other,
    IN: lambda value, other: value in other,
    BETWEEN: lambda value, other: other[0] <= value <= other[1],
}


def resolve(expression: Expression,
            identifier: str,
            date: pd.Timestamp,
            fetcher: DataFetcher,
            on_missing: bool = False,
            max_age_days: int | None = MAX_AGE_DAYS,
            currency: str = BASE_CURRENCY) -> bool:
    """Whether an instrument passes an expression on a date.

    Args:
        expression: The tree to evaluate.
        identifier: The instrument.
        date: The date to stand on (in an index, a rebalance date).
        fetcher: The data.
        on_missing: What a comparison answers when the field has no value.
        max_age_days: How stale a feature may be and still count.
        currency: The currency money fields (`market_cap`,
            `free_float_market_cap`) are compared in. An index passes its
            own; USD otherwise.

    Returns:
        bool: The answer for this instrument on this date.
    """
    if isinstance(expression, _Group):
        answers = (resolve(operand, identifier, date, fetcher, on_missing,
                           max_age_days, currency)
                   for operand in expression.operands)

        return all(answers) if expression.node == ALL else any(answers)

    if isinstance(expression, Not):
        return not resolve(expression.operand, identifier, date, fetcher,
                           on_missing, max_age_days, currency)

    if isinstance(expression, Comparison):
        return _compare(expression, identifier, date, fetcher, on_missing,
                        max_age_days, currency)

    # A bare `Field` is not a question. Reaching here means an expression was
    # built but never compared, which is a mistake worth surfacing rather than
    # answering arbitrarily. `ExpressionError` rather than `TypeError`: the
    # expression may have come from a request, and a `TypeError` has no
    # mapping, so a universe filter of a bare field answered a plain-text 500
    # where it should have been told its filter was not a condition (BN-131).
    raise ExpressionError(
        f"cannot resolve {expression!r}: it is not a comparison. A screen "
        f"must compare a field to a value, e.g. `market_cap > 1e9`.")


def _compare(comparison: Comparison,
             identifier: str,
             date: pd.Timestamp,
             fetcher: DataFetcher,
             on_missing: bool,
             max_age_days: int | None,
             currency: str = BASE_CURRENCY) -> bool:
    """One comparison."""
    value = value_of(comparison.field, identifier, date, fetcher,
                     max_age_days, currency)

    if value is None:
        logger.debug("%s has no %s knowable on %s; treated as %s.",
                     identifier, comparison.field.path,
                     date.strftime("%Y-%m-%d"), on_missing)

        return on_missing

    return bool(OPERATIONS[comparison.comparison](value, comparison.value))


def value_of(field: Field,
             identifier: str,
             date: pd.Timestamp,
             fetcher: DataFetcher,
             max_age_days: int | None = MAX_AGE_DAYS,
             currency: str = BASE_CURRENCY) -> Any:
    """One field's value for one instrument, as of a date.

    Money fields are in *currency*.

    Returns:
        The value, or None when the instrument has none knowable by `date`.
    """
    if field.namespace == FEATURES:
        return fetcher.fetch_feature(identifier, field.name, date,
                                     field.dataset, max_age_days)

    if field.namespace == REFERENCE:
        return _reference_value(field, identifier, date, fetcher)

    if field.namespace == MARKET:
        return _market_value(field, identifier, date, fetcher, currency)

    return None


def numbers_of(field: Field,
               identifiers: list[str],
               date: pd.Timestamp,
               fetcher: DataFetcher,
               max_age_days: int | None = MAX_AGE_DAYS,
               currency: str = BASE_CURRENCY) -> dict[str, float | None]:
    """`value_of` for many instruments on one date, as numbers: the same
    answers, None where there is no finite number.

    Read in one pass where the field allows (a feature, a market column, a
    market cap), so a rebalance across thousands of names reads the data
    once rather than once a name; name by name otherwise.
    """
    # BN-387: a factor scored over 4,500 names spent 27 s here, name by name.
    if field.namespace == FEATURES:
        table = fetcher.features.table_as_of(identifiers, field.name, [date],
                                             field.dataset, max_age_days)
        # As a dict, so each name is a lookup in it (BN-416).
        row = table.iloc[0].to_dict() if len(table) else {}

        return {name: _finite(row.get(name)) for name in identifiers}

    if field.namespace == MARKET and field.name not in (ADV_3M, FREE_FLOAT_MARKET_CAP,
                                                        FREE_FLOAT_FIELD):
        return _market_numbers(field, identifiers, date, fetcher, currency)

    return {name: _finite(value_of(field, name, date, fetcher, max_age_days, currency))
            for name in identifiers}


def _market_numbers(field: Field,
                    identifiers: list[str],
                    date: pd.Timestamp,
                    fetcher: DataFetcher,
                    currency: str) -> dict[str, float | None]:
    """A market column's last value on or before the date, or a market cap
    in *currency*, for each name, from one read of the window."""
    start = date - pd.Timedelta(days=10)
    names = list(identifiers)
    read: pd.DataFrame | None = None

    def last(column: str) -> pd.Series:
        nonlocal read
        # Each name's last value straight from the store's session index
        # when it can give it (BN-411), else from a read of the window.
        quick = getattr(fetcher, "last_market_values", None)
        values = quick(names, start, date, column) if callable(quick) else None

        if isinstance(values, np.ndarray):
            return pd.Series(values, index=names).dropna()

        if read is None:
            read = fetcher.fetch_market_data(names, start.strftime("%Y-%m-%d"),
                                             date.strftime("%Y-%m-%d"))

        if read.empty or column not in read.columns:
            return pd.Series(dtype=float)

        return read[column].groupby(level=0).last()

    if field.name != MARKET_CAP:
        found = last(column_for(field))

        values = found.to_dict()

        return {name: _finite(values.get(name)) for name in identifiers}

    # As dicts, so each name is a lookup in them (BN-416): looked up in the
    # Series it was 1.6 s over 20 rebalances at 6,000 names.
    prices, shares = last(CLOSE).to_dict(), last(SHARES).to_dict()
    rates = _rates(identifiers, date, fetcher, currency)

    caps: dict[str, float | None] = {}

    for name in identifiers:
        price, held, rate = _finite(prices.get(name)), _finite(shares.get(name)), rates.get(name)
        caps[name] = (None if price is None or held is None or rate is None
                      else _finite(price * held * rate))

    return caps


def _rates(identifiers: list[str],
           date: pd.Timestamp,
           fetcher: DataFetcher,
           currency: str) -> dict[str, float | None]:
    """Each name's FX from its quote currency into *currency*, as `_rate`
    finds it, each currency's rate read once."""
    reference = fetcher.fetch_reference_data(list(identifiers), date.strftime("%Y-%m-%d"))
    quoted: dict[str, str | None] = dict.fromkeys(identifiers)

    if not reference.empty and "CURRENCY" in reference.columns:
        for name, value in reference["CURRENCY"].groupby(level=0).first().items():
            quoted[str(name)] = None if pd.isna(value) else str(value).upper()

    by_currency = {code: fetcher.fx_rate_on(code, currency.upper(), date)
                   for code in {code for code in quoted.values() if code is not None}}

    return {name: None if code is None else by_currency[code] for name, code in quoted.items()}


def _finite(value: Any) -> float | None:
    """A finite number, or None."""
    if value is None or isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _reference_value(field: Field,
                     identifier: str,
                     date: pd.Timestamp,
                     fetcher: DataFetcher) -> Any:
    """A reference dimension, resolved as of the date.

    Point-in-time for the same reason as everything else here: a name that
    changed sector in June was in its old sector at a March rebalance, and a
    screen that used today's sector would rewrite history.
    """
    column = column_for(field)
    reference = fetcher.fetch_reference_data(identifier,
                                             date.strftime("%Y-%m-%d"))

    if reference.empty or column not in reference.columns:
        return None

    values = reference[column].dropna()

    return None if values.empty else values.iloc[0]


def _market_value(field: Field,
                  identifier: str,
                  date: pd.Timestamp,
                  fetcher: DataFetcher,
                  currency: str = BASE_CURRENCY) -> Any:
    """A market column or a derived quantity, as of the date."""
    if field.name == ADV_3M:
        return _adv(identifier, date, fetcher)

    if field.name in (MARKET_CAP, FREE_FLOAT_MARKET_CAP):
        return _cap(identifier, date, fetcher,
                    float_adjusted=field.name == FREE_FLOAT_MARKET_CAP,
                    currency=currency)

    if field.name == FREE_FLOAT_FIELD:
        return _free_float(identifier, date, fetcher)

    return _column(identifier, date, fetcher, column_for(field))


def _free_float(identifier: str,
                date: pd.Timestamp,
                fetcher: DataFetcher) -> float | None:
    """The free float in force on the date, as every other read finds it.

    Through the fetcher's `free_float_backfill_days`, so an expression and
    the market-cap weighting agree about which names have a float.
    """
    # BN-254: read through the 10-day market window until then.
    return fetcher.fetch_free_float_factor(identifier,
                                           date.strftime("%Y-%m-%d"))


def _column(identifier: str,
            date: pd.Timestamp,
            fetcher: DataFetcher,
            column: str) -> Any:
    """The last value of a market column on or before the date.

    On or before rather than exactly on: a rebalance can land on a day this
    instrument did not trade, and treating that as "no value" would drop names
    from an index for a reason that has nothing to do with the screen.
    """
    frame = _window(identifier, date, fetcher, days=10)

    if frame.empty or column not in frame.columns:
        return None

    values = frame[column].dropna()

    return None if values.empty else values.iloc[-1]


def _cap(identifier: str,
         date: pd.Timestamp,
         fetcher: DataFetcher,
         float_adjusted: bool,
         currency: str = BASE_CURRENCY) -> float | None:
    """Price x shares outstanding in *currency*, optionally float-adjusted.

    Computed rather than read: a market cap is not stored anywhere, and both
    of its inputs move.

    **Converted into *currency***: the index's, when the expression is part
    of one (so `market_cap > 1e9` means the same as `MarketCapRule`'s bound),
    and USD otherwise, matching the reference endpoint's default (BN-133,
    BN-253). Since BN-128 the members of one universe are quoted in
    seven currencies, so comparing raw local values would rank a yen cap above
    a dollar one on magnitude alone -- and a screen like `market_cap > 1e9`
    would then select on currency as much as on size. The two surfaces have to
    agree, or the field a user picks from the catalogue means something
    different from the one their screen compares.
    """
    price = _column(identifier, date, fetcher, CLOSE)
    shares = _column(identifier, date, fetcher, SHARES)
    rate = _rate(identifier, date, fetcher, currency)

    # An unconvertible cap is a cap this screen cannot compute, which is the
    # answer `price is None` already gets (BN-206). It is emphatically not the
    # local number: comparing that against dollars ranks on currency as much
    # as on size, which is the bug BN-188 was raised for.
    if price is None or shares is None or rate is None:
        return None

    cap = float(price) * float(shares) * rate

    if not float_adjusted:
        return cap

    free_float = _free_float(identifier, date, fetcher)

    return None if free_float is None else cap * float(free_float)


def _rate(identifier: str,
          date: pd.Timestamp,
          fetcher: DataFetcher,
          currency: str = BASE_CURRENCY) -> float | None:
    """FX from an instrument's quote currency into *currency*, or None."""
    # BN-206: None where the old answer was 1.0, in both of the two ways
    # this can fail. A cap quoted in yen and compared as though it were dollars
    # is BN-188's original bug -- a 156x overstatement for a JPY name, in a
    # screen whose whole purpose is ranking by size -- and it survived here
    # because BN-188 swept `index/`, `data/` and `server/` and this module is in
    # none of them. The currency being unknown is the second way, and it used to
    # answer 1.0 too: an instrument with no reference row was silently treated
    # as already quoted in USD. That is the same claim about two currencies,
    # made with even less to go on.
    reference = fetcher.fetch_reference_data(identifier,
                                             date.strftime("%Y-%m-%d"))

    if reference.empty or "CURRENCY" not in reference.columns:
        return None

    value = reference["CURRENCY"].iloc[0]

    if pd.isna(value):
        return None

    return fetcher.fx_rate_on(str(value).upper(), currency.upper(), date)


def _adv(identifier: str,
         date: pd.Timestamp,
         fetcher: DataFetcher) -> float | None:
    """Mean daily volume over the trailing window."""
    start = date - pd.DateOffset(months=TRAILING_MONTHS)
    frame = fetcher.fetch_market_data([identifier],
                                      start.strftime("%Y-%m-%d"),
                                      date.strftime("%Y-%m-%d"))

    volumes = average_daily_volume(frame, date)

    if identifier not in volumes.index:
        return None

    value = volumes.loc[identifier]

    return None if pd.isna(value) else float(value)


def _window(identifier: str,
            date: pd.Timestamp,
            fetcher: DataFetcher,
            days: int) -> pd.DataFrame:
    """A short lookback ending at the date.

    Bounded rather than open-ended: reading an instrument's whole history to
    answer one comparison would make a rebalance across thousands of names
    quadratic in the length of the panel.
    """
    start = date - pd.Timedelta(days=days)

    return fetcher.fetch_market_data(identifier, start.strftime("%Y-%m-%d"),
                                     date.strftime("%Y-%m-%d"))


# Re-exported so the rule module does not reach into the private group class.
GROUP_NODES = (ALL, ANY, NOT)
