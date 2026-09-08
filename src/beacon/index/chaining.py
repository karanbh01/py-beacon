# src/beacon/index/chaining.py
"""
Weight-rebalanced level chaining, over identifiers.

The arithmetic that turns a schedule of target weights into a daily level
path: units are fixed between rebalances so weights drift with relative
performance, each rebalance rebuilds the units at the value the old holdings
reached — which is what keeps the level continuous across it — and prices are
converted into the index currency with as-of FX rates, exactly as the
calculator and the engine convert theirs. The divisor is 1.0 throughout,
because the aggregate this represents *is* its own portfolio value and there
is no market-value scale for a divisor to absorb.

Split out of `derived.py` (BN-171), which had grown past the file-size
convention with this as its obvious seam. Nothing here knows what a derivation
is: it takes an index's identity, a parent calculation and a solved schedule,
and it would serve any other caller with weights to chain. That is why the
signature carries `index_id`, `base_value` and `currency` as plain arguments
rather than the definition they used to be read from — the three attributes
were the whole of the dependency.

This is a deliberate reimplementation of `IndexCalculator`'s arithmetic rather
than a reuse of its mixins, which are built around Asset objects and per-day
price lookups. The economics are the same, and a test proves it by
recomputing a whole path independently from raw prices.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from .result import IndexResult, daily_weights_frame

logger = logging.getLogger(__name__)


def chain_levels(index_id: str,
                 base_value: float,
                 currency: str,
                 parent: IndexResult,
                 solved: dict[pd.Timestamp, dict[str, float]],
                 data_provider: DataFetcher,
                 price_column: str) -> IndexResult:
    """Chain the solved weights into daily levels on the parent's calendar.

    The weight-rebalanced arithmetic the calculator applies, restated over
    identifiers: units are fixed between rebalances, each rebalance rebuilds
    them at the value the old holdings reached (which is what keeps the level
    continuous), and the path starts at the definition's base value. The
    divisor is 1.0 throughout — the aggregate this index represents is its own
    portfolio value, so there is no market-value scale for a divisor to absorb.

    A day on which the holdings cannot be valued at all carries the level
    forward and records no weights, matching the calculator's behaviour.
    """
    days = parent.index_levels.index
    unit_values = _unit_value_panel(currency, solved, days,
                                    data_provider, price_column)

    levels: dict[pd.Timestamp, float] = {}
    divisors: dict[pd.Timestamp, float] = {}
    constituent_snapshots: dict[pd.Timestamp, list[str]] = {}
    weight_snapshots: dict[pd.Timestamp, dict[str, float]] = {}
    daily_records: list[dict[str, object]] = []

    units: dict[str, float] = {}
    level = base_value
    divisor = 0.0

    for day in days:
        values: dict[str, float] = {}

        if day in solved:
            if divisor <= 0.0:
                # Inception: the chained path starts at the base value.
                aggregate = base_value
                divisor = 1.0
            else:
                aggregate = _valued(units, unit_values, day)

            if aggregate > 0.0:
                units = _units_for(solved[day], aggregate, unit_values, day)
                values = _holding_values(units, unit_values, day)
                level = float(sum(values.values())) / divisor

                constituent_snapshots[day] = sorted(solved[day])
                weight_snapshots[day] = dict(solved[day])
            else:
                logger.warning(
                    "[%s] Holdings of '%s' could not be valued at the "
                    "rebalance; carrying the level and composition forward.",
                    day.date(), index_id)

        elif units and divisor > 0.0:
            values = _holding_values(units, unit_values, day)
            aggregate = float(sum(values.values()))

            if aggregate > 0.0:
                level = aggregate / divisor
            else:
                values = {}

        levels[day] = level
        divisors[day] = divisor
        daily_records.extend(_weight_rows(day, units, values))

    logger.info("Chained '%s': %d day(s), %d rebalance(s), final level %.4f.",
                index_id, len(days), len(weight_snapshots), level)

    return IndexResult(index_id=index_id,
                       index_levels=pd.Series(levels),
                       divisor_history=pd.Series(divisors),
                       constituent_snapshots=constituent_snapshots,
                       weight_snapshots=weight_snapshots,
                       daily_weights=daily_weights_frame(daily_records))


def _unit_value_panel(currency: str,
                      solved: dict[pd.Timestamp, dict[str, float]],
                      days: pd.Index,
                      data_provider: DataFetcher,
                      price_column: str) -> pd.DataFrame:
    """What one unit of each name is worth each day, in the index currency.

    One market-data fetch per name and one FX fetch per currency pair — the
    same conversion the calculator and the engine apply, vectorised: prices
    are quoted where the company lists, the index has one currency, and it is
    the drift in the rate a foreign holding actually experiences. Prices and
    rates are carried forward over gaps; a name with no price yet is NaN and
    is treated as unvaluable (zero units, zero value) until one appears.
    """
    assets = sorted({asset for weights in solved.values() for asset in weights})
    start = days[0].strftime("%Y-%m-%d")
    end = days[-1].strftime("%Y-%m-%d")

    columns: dict[str, pd.Series] = {}
    rates: dict[str, pd.Series] = {}

    for asset in assets:
        prices = _price_series(data_provider, asset, start, end,
                               price_column).reindex(days).ffill()

        currency = _currency_of(data_provider, asset, currency)
        if currency != currency:
            if currency not in rates:
                rates[currency] = _rate_series(data_provider, currency,
                                               currency, days)

            prices = prices * rates[currency]

        columns[asset] = prices

    return pd.DataFrame(columns, index=days)


def _price_series(data_provider: DataFetcher,
                  asset: str,
                  start: str,
                  end: str,
                  price_column: str) -> pd.Series:
    """One name's price series over the window, or an empty series."""
    frame = data_provider.fetch_market_data(asset, start, end)

    if frame.empty or price_column not in frame.columns:
        logger.warning("No '%s' prices for %s over %s..%s; it will hold zero "
                       "units.", price_column, asset, start, end)

        return pd.Series(dtype=float)

    return frame[price_column].astype(float)


def _currency_of(data_provider: DataFetcher,
                 asset: str,
                 default: str) -> str:
    """The currency a name is quoted in, defaulting to the index's own."""
    try:
        frame = data_provider.fetch_reference_data(asset)

        if not frame.empty and "CURRENCY" in frame.columns:
            value = frame["CURRENCY"].iloc[0]

            if pd.notna(value):
                return str(value).upper()
    except Exception as error:
        logger.warning("Could not resolve the currency of %s: %s.",
                       asset, error)

    return default


def _rate_series(data_provider: DataFetcher,
                 from_currency: str,
                 to_currency: str,
                 days: pd.Index) -> pd.Series:
    """An FX pair as-of each calculation day, carried forward over gaps."""
    series = data_provider.fetch_fx_rates(from_currency, to_currency)

    if series.empty:
        logger.warning("No %s/%s rate; those holdings cannot be valued.",
                       from_currency, to_currency)

        return pd.Series(float("nan"), index=days)

    return series.sort_index().astype(float).reindex(days, method="ffill")


def _units_for(weights: dict[str, float],
               aggregate: float,
               unit_values: pd.DataFrame,
               day: pd.Timestamp) -> dict[str, float]:
    """Units realising *weights* of *aggregate* at today's unit values.

    The calculator's `index_units`, restated: a name with no usable unit value
    holds zero units and contributes nothing, rather than an infinite
    position.
    """
    units: dict[str, float] = {}

    for asset, weight in weights.items():
        value = unit_values.at[day, asset]

        if pd.isna(value) or float(value) <= 0.0:
            logger.warning("No unit value for %s on %s; it holds zero units.",
                           asset, day.date())
            units[asset] = 0.0
            continue

        units[asset] = weight * aggregate / float(value)

    return units


def _holding_values(units: dict[str, float],
                    unit_values: pd.DataFrame,
                    day: pd.Timestamp) -> dict[str, float]:
    """What each holding is worth today: units times unit value."""
    values: dict[str, float] = {}

    for asset, count in units.items():
        value = unit_values.at[day, asset]
        values[asset] = 0.0 if pd.isna(value) else count * float(value)

    return values


def _valued(units: dict[str, float],
            unit_values: pd.DataFrame,
            day: pd.Timestamp) -> float:
    """Total value of the holdings on *day*."""
    return float(sum(_holding_values(units, unit_values, day).values()))


def _weight_rows(day: pd.Timestamp,
                 units: dict[str, float],
                 values: dict[str, float]) -> list[dict[str, object]]:
    """One daily-panel record per holding — realised shares of the aggregate.

    The calculator's `weight_rows`, restated over identifiers: a day whose
    holdings are worth nothing records nothing, because it has no weights to
    record.
    """
    total = sum(values.values())

    if total <= 0.0:
        return []

    return [{"DATE": day,
             "IDENTIFIER": asset,
             "AMOUNT": units[asset],
             "WEIGHT": values[asset] / total}
            for asset in units]
