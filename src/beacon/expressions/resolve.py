# src/beacon/expressions/resolve.py
"""
Turning an expression into an answer for one instrument on one date.

An expression is a description; resolving it needs an instrument and a date,
and in an index that date is the **rebalance**.

## Point-in-time, or the whole features layer was pointless

A screen on `data.features.fundamentals.revenue` at a rebalance on 1 April
must see what was published by 1 April. Q1 revenue announced in mid-May is
invisible on that date, however completely the quarter had ended.

Every feature read here goes through `DataFetcher.fetch_feature`, which is the
accessor that enforces this (BN-135). Reading the table directly would put
look-ahead straight back in, and the resulting backtest would look *better*
and be wrong — which is the failure nobody catches, because a better number is
not a symptom anybody investigates.

Market and reference data are read as-of the same date for the same reason.

## Missing is not zero

A name with no value for a field yields `None`, and the rule above decides
what that means. This is deliberately distinct from a value that is
legitimately zero: zero fails a `> 0` test honestly, where missing has nothing
to compare at all. Collapsing the two would let a screen for "revenue above a
billion" quietly admit every company the dataset has never heard of.
"""
import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pandas as pd

from ..analysis.liquidity import TRAILING_MONTHS, average_daily_volume
from ..data.features import MAX_AGE_DAYS
from ..data.fetcher import DataFetcher
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
            max_age_days: int | None = MAX_AGE_DAYS) -> bool:
    """Whether an instrument passes an expression on a date.

    Args:
        expression: The tree to evaluate.
        identifier: The instrument.
        date: The date to stand on — a rebalance date, in an index.
        fetcher: The data.
        on_missing: What a comparison answers when the field has no value.
        max_age_days: How stale a feature may be and still count.

    Returns:
        bool: The answer for this instrument on this date.
    """
    if isinstance(expression, _Group):
        answers = (resolve(operand, identifier, date, fetcher, on_missing,
                           max_age_days)
                   for operand in expression.operands)

        return all(answers) if expression.node == ALL else any(answers)

    if isinstance(expression, Not):
        return not resolve(expression.operand, identifier, date, fetcher,
                           on_missing, max_age_days)

    if isinstance(expression, Comparison):
        return _compare(expression, identifier, date, fetcher, on_missing,
                        max_age_days)

    # A bare `Field` is not a question. Reaching here means an expression was
    # built but never compared, which is a mistake worth surfacing rather than
    # answering arbitrarily.
    raise TypeError(f"cannot resolve {expression!r}: it is not a comparison.")


def _compare(comparison: Comparison,
             identifier: str,
             date: pd.Timestamp,
             fetcher: DataFetcher,
             on_missing: bool,
             max_age_days: int | None) -> bool:
    """One comparison."""
    value = value_of(comparison.field, identifier, date, fetcher,
                     max_age_days)

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
             max_age_days: int | None = MAX_AGE_DAYS) -> Any:
    """One field's value for one instrument, as of a date.

    Returns:
        The value, or None when the instrument has none knowable by `date`.
    """
    if field.namespace == FEATURES:
        return fetcher.fetch_feature(identifier, field.name, date,
                                     field.dataset, max_age_days)

    if field.namespace == REFERENCE:
        return _reference_value(field, identifier, date, fetcher)

    if field.namespace == MARKET:
        return _market_value(field, identifier, date, fetcher)

    return None


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
                  fetcher: DataFetcher) -> Any:
    """A market column or a derived quantity, as of the date."""
    if field.name == ADV_3M:
        return _adv(identifier, date, fetcher)

    if field.name in (MARKET_CAP, FREE_FLOAT_MARKET_CAP):
        return _cap(identifier, date, fetcher,
                    float_adjusted=field.name == FREE_FLOAT_MARKET_CAP)

    return _column(identifier, date, fetcher, column_for(field))


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
         float_adjusted: bool) -> float | None:
    """Price x shares outstanding, in USD, optionally float-adjusted.

    Computed rather than read: a market cap is not stored anywhere, and both
    of its inputs move.

    **Converted into USD**, matching the `market_cap` the reference endpoint
    serves (BN-133). Since BN-128 the members of one universe are quoted in
    seven currencies, so comparing raw local values would rank a yen cap above
    a dollar one on magnitude alone -- and a screen like `market_cap > 1e9`
    would then select on currency as much as on size. The two surfaces have to
    agree, or the field a user picks from the catalogue means something
    different from the one their screen compares.
    """
    price = _column(identifier, date, fetcher, CLOSE)
    shares = _column(identifier, date, fetcher, SHARES)

    if price is None or shares is None:
        return None

    cap = float(price) * float(shares) * _rate(identifier, date, fetcher)

    if not float_adjusted:
        return cap

    free_float = _column(identifier, date, fetcher, FREE_FLOAT)

    return None if free_float is None else cap * float(free_float)


def _rate(identifier: str,
          date: pd.Timestamp,
          fetcher: DataFetcher) -> float:
    """FX from an instrument's quote currency into USD, as of the date."""
    reference = fetcher.fetch_reference_data(identifier,
                                             date.strftime("%Y-%m-%d"))

    if reference.empty or "CURRENCY" not in reference.columns:
        return 1.0

    value = reference["CURRENCY"].iloc[0]
    currency = str(value).upper() if pd.notna(value) else BASE_CURRENCY

    return _rate_into_base(fetcher, currency, date.strftime("%Y-%m-%d"))


@lru_cache(maxsize=4096)
def _rate_into_base(fetcher: DataFetcher,
                    currency: str,
                    as_of: str) -> float:
    """One currency's rate into USD on one date.

    Cached because resolution is per instrument: a rebalance over thousands of
    names spans a handful of currencies, and looking one up per name would be
    thousands of slices to answer seven questions. Keyed on the fetcher too,
    so two stores in one process cannot borrow each other's rates; bounded, so
    a long-running server does not accumulate them without limit.
    """
    if currency == BASE_CURRENCY:
        return 1.0

    series = fetcher.fetch_fx_rates(currency, BASE_CURRENCY, end_date=as_of)

    if series.empty:
        logger.warning("No %s/%s rate on or before %s; caps quoted in %s are "
                       "compared unconverted.",
                       currency, BASE_CURRENCY, as_of, currency)

        return 1.0

    return float(series.iloc[-1])


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
