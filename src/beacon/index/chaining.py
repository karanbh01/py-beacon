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
from ..exceptions import CalculationError
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

    Raises:
        CalculationError: If a name the schedule actually allocates to has no
            prices anywhere in the window (BN-184).
    """
    assets = sorted({asset for weights in solved.values() for asset in weights})

    # Names the schedule actually allocates to. A name carried at a weight of
    # zero throughout is never held, so its prices are not needed and their
    # absence is not a failure.
    held = {asset
            for weights in solved.values()
            for asset, weight in weights.items()
            if weight != 0.0}

    start = days[0].strftime("%Y-%m-%d")
    end = days[-1].strftime("%Y-%m-%d")

    columns: dict[str, pd.Series] = {}
    rates: dict[str, pd.Series] = {}

    for asset in assets:
        prices = _price_series(data_provider, asset, start, end,
                               price_column, asset in held)

        columns[asset] = _converted(prices.reindex(days).ffill(),
                                    _currency_of(data_provider, asset, currency),
                                    currency, days, data_provider, rates)

    return pd.DataFrame(columns, index=days)


def _converted(prices: pd.Series,
               quoted_in: str,
               index_currency: str,
               days: pd.Index,
               data_provider: DataFetcher,
               rates: dict[str, pd.Series]) -> pd.Series:
    """*prices*, converted into the index currency, caching the rate series.

    Split out of :func:`_unit_value_panel`, where the conversion used to rebind
    the index-currency argument to the name's own currency and then compare it
    against itself — so the test was always false, no rate was ever applied,
    and every later name was compared against the previous one's currency
    (BN-184). A chained index over foreign names added its prices as though
    every currency's unit were the same size, which is precisely the defect
    BN-188 removed from the market-cap weighting.
    """
    if quoted_in == index_currency:
        return prices

    if quoted_in not in rates:
        rates[quoted_in] = _rate_series(data_provider, quoted_in,
                                        index_currency, days)

    return prices * rates[quoted_in]


def _price_series(data_provider: DataFetcher,
                  asset: str,
                  start: str,
                  end: str,
                  price_column: str,
                  held: bool) -> pd.Series:
    """One name's price series over the window, or an empty series.

    Raises:
        CalculationError: If *held* and the window holds no prices at all. A
            name the schedule allocates to and the data has never priced
            cannot be held: it used to take zero units, which leaves the
            chained index short by that name's whole weight and the level
            correspondingly low, with only a log saying so (BN-184). A name
            carried at zero weight throughout is not held, so its absence is
            allowed through as an all-NaN column.
    """
    frame = data_provider.fetch_market_data(asset, start, end)

    if frame.empty or price_column not in frame.columns:
        if held:
            raise CalculationError(
                calculation_name="ChainedLevels",
                details=(f"no '{price_column}' prices for {asset} over "
                         f"{start}..{end}, but the schedule allocates to it. "
                         f"Holding zero units of it would leave the index "
                         f"short by its whole weight and publish the "
                         f"resulting level as the index's own."))

        logger.warning("No '%s' prices for %s over %s..%s; it is never held, "
                       "so it stays unvalued.", price_column, asset, start, end)

        return pd.Series(dtype=float)

    return frame[price_column].astype(float)


def _currency_of(data_provider: DataFetcher,
                 asset: str,
                 default: str) -> str:
    """The currency a name is quoted in, defaulting to the index's own.

    BN-184, triaged as leave. The default is not a substitute for a value the
    data has: it is what reference data that does not model currency at all
    means. A dataset with no CURRENCY column is single-currency by
    construction, and reading its names as quoted in the index's own currency
    is the only coherent interpretation available — refusing would reject
    every such dataset over a distinction it does not draw. The case that
    *does* matter, a name known to be quoted elsewhere with no rate to convert
    it, is caught downstream in :func:`_rate_series`, which yields NaN rather
    than an unconverted price.

    The former bare ``except Exception`` around this is gone (BN-184): a
    reference lookup that fails is not a name quoted in the index currency,
    and swallowing it here would absorb any refusal the data layer raises.
    """
    frame = data_provider.fetch_reference_data(asset)

    if not frame.empty and "CURRENCY" in frame.columns:
        value = frame["CURRENCY"].iloc[0]

        if pd.notna(value):
            return str(value).upper()

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

    BN-184, triaged as report-not-refuse, and deliberately left alone here.
    This is the twin of `MarketValuesMixin.index_units`, whose identical
    substitution is filed with a demonstrated wrong output as #204; the two
    must move together or the chained and calculated paths will disagree about
    what an unvaluable name means. What is missing either way is that the
    result does not say the index ran under-invested on those days.
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
