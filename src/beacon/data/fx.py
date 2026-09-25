# src/beacon/data/fx.py
"""Which rate converts one currency into another, when only some pairs are stored.

A pair is stored as its own market-data identifier, such as `GBPUSD`. With
only `GBPUSD` and `EURUSD` stored, USD to GBP is one over GBPUSD and GBP to
EUR is GBPUSD divided by EURUSD, so neither needs a stored pair of its own.

The rate for a pair is found in this order:

1. the stored pair itself
2. the inverse of the stored reverse pair
3. a cross through USD, each leg found by rules 1 and 2

A cross is built so the settings keep their meaning. Under carry-forward a
leg's rate is the one in force on the day, so the cross on a day uses both
legs as in force that day. Under exact-day a leg must have printed on the day,
so the cross exists only on days both legs printed. Neither ever uses a rate
dated after the day, and the no-look-ahead search in `DataFetcher` applies to
the derived series exactly as to a stored one.

A stored rate of zero or below has no inverse; it is dropped, and that day is
treated as one with no rate.
"""
# BN-235. The lookup used to find only the exact pair asked for, so a GBP
# index holding US shares refused for want of a rate it effectively had. The
# synthetic data stores only `XXXUSD` pairs, which made every non-USD view of
# it refuse.
from collections.abc import Callable

import pandas as pd

# The currency a cross rate passes through. Market data quotes most pairs
# against the dollar, so it is the one pivot that makes a cross available for
# nearly any two currencies.
PIVOT = "USD"

DIRECT = "direct"
INVERSE = "inverse"
CROSS = f"cross via {PIVOT}"

# How a caller reads the pairs it holds: the stored series for FROM/TO, empty
# when that exact pair is not stored.
StoredRates = Callable[[str, str], pd.Series]


def rate_series(stored: StoredRates,
                source: str,
                target: str,
                exact_day: bool) -> tuple[pd.Series, str | None]:
    """The series converting *source* into *target*, and how it was found.

    Args:
        stored: Reads a stored pair, returning an empty series when absent.
        source: The currency converted out of, upper case.
        target: The currency converted into, upper case.
        exact_day: True under the EXACT_DAY policy, where a cross exists only
            on days both legs printed.

    Returns:
        tuple: The rate series indexed by date and sorted, and the route:
        DIRECT, INVERSE or CROSS. An empty series and None when no route
        exists.
    """
    leg, route = _leg(stored, source, target)

    if route is not None:
        return leg, route

    if PIVOT in (source, target):
        return pd.Series(dtype=float), None

    first, first_route = _leg(stored, source, PIVOT)
    second, second_route = _leg(stored, PIVOT, target)

    if first_route is None or second_route is None:
        return pd.Series(dtype=float), None

    return _cross(first, second, exact_day), CROSS


def _leg(stored: StoredRates,
         source: str,
         target: str) -> tuple[pd.Series, str | None]:
    """One pair, stored as asked or as its reverse."""
    direct = stored(source, target)

    if not direct.empty:
        return direct, DIRECT

    reverse = stored(target, source)

    if reverse.empty:
        return pd.Series(dtype=float), None

    # A rate of zero or below has no inverse. It is dropped rather than
    # turned into infinity, and the day is then treated as one with no rate.
    usable = reverse.astype(float).where(reverse > 0)

    return (1.0 / usable).dropna(), INVERSE


def _cross(first: pd.Series,
           second: pd.Series,
           exact_day: bool) -> pd.Series:
    """Two legs multiplied, with both legs taken from the same day."""
    legs = pd.concat([first.astype(float), second.astype(float)],
                     axis=1,
                     join="inner" if exact_day else "outer").sort_index()

    if not exact_day:
        # Each leg carried forward to every day either leg printed, never
        # backward: a day before a leg's first rate stays empty and is
        # dropped, as the carry rule requires.
        legs = legs.ffill()

    return (legs.iloc[:, 0] * legs.iloc[:, 1]).dropna()
