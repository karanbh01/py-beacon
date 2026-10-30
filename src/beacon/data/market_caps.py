# src/beacon/data/market_caps.py
"""
A name's market capitalisation from its price and share count, converted
into another currency and adjusted for free float, and a set of names'
shares of their total.

Each figure is scaled from its own base rather than one from another:
multiplication is not associative in floating point, so the converted
free-float cap is the converted cap times the float, not the local
free-float cap converted. An unknown rate leaves the converted figures
unknown, and an unknown float the free-float ones, rather than assuming 1.
"""
# BN-417: moved from the server's market cap views, which now call these.
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MarketCaps:
    """A name's caps: in its own currency and converted, in full and
    free-float adjusted; None where the rate or the float is unknown."""
    local: float
    converted: float | None
    free_float_local: float | None
    free_float_converted: float | None


def caps_of(price: float,
            shares: float,
            rate: float | None,
            free_float: float | None) -> MarketCaps:
    """The caps of *shares* at *price*, converted by *rate* and adjusted by
    *free_float* (the share of the shares that trade freely)."""
    local = float(price) * float(shares)
    converted = local * rate if rate is not None else None

    return MarketCaps(local=local,
                      converted=converted,
                      free_float_local=None if free_float is None else local * free_float,
                      free_float_converted=(None if free_float is None or converted is None
                                            else converted * free_float))


def is_stale(priced_from: pd.Timestamp,
             on: pd.Timestamp,
             lookback_days: int) -> bool:
    """Whether a figure printed on *priced_from* is more than *lookback_days*
    old on *on*."""
    return bool((on - priced_from).days > lookback_days)


def shares_of_total(values: dict[str, float | None]) -> dict[str, float | None]:
    """Each value's share of the positive values' total; None for a name
    with no value, or when nothing is positive."""
    total = sum(value for value in values.values() if value is not None and value > 0)

    return {name: (float(value) / total if value is not None and total > 0 else None)
            for name, value in values.items()}
