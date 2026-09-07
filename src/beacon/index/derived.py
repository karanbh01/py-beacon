# src/beacon/index/derived.py
"""
Optimised indices — an index derived from an index you already built.

The owner's framing (planning/optimised_index_design.md rev 2): create an
index, then optimise it — objective function, constraints — and that creates a
NEW index. The derivation stores exactly three things — the **source** index,
the **objective** and the **constraints** — plus the usual identity attributes.
No weights are stored anywhere: definitions are rules, weights are calculated,
and calculations are cached (BN-160).

Two consequences shape this module:

* **Rebalancing follows the parent.** The child solves exactly at the parent's
  published snapshots; a frequency of its own would have no parent weights at
  the extra dates. The derived definition therefore declares no schedule.
* **The calculation is a normal IndexResult.** Solve the parent's weights at
  each rebalance, then chain the solved weights into daily levels — same
  calendar as the parent, level path equal to compounding the solved-weight
  portfolio's returns from the parent's price data.

:class:`OptimisedIndexDefinition` is deliberately a sibling of
:class:`~beacon.index.constructor.IndexDefinition` rather than a subclass: a
subclass would have to invent eligibility rules, a weighting scheme and a
rebalancing frequency it does not have, and the calculator — which stays
untouched — must never receive one by accident.

The level chaining here is a clean reimplementation of the calculator's
weight-rebalanced arithmetic rather than a reuse of its mixins, which are
built around Asset objects and per-day price lookups. The economics are the
same: units are fixed between rebalances so weights drift with relative
performance, the level is continuous across a rebalance because the new units
are built at the value the old ones reached, and prices are converted into the
index currency with as-of FX rates exactly as the calculator and the engine
convert theirs. The divisor is the identity the level path makes it: the
aggregate this index represents *is* its own portfolio value, so the divisor
initialises to 1.0 and never moves.

scipy enters only when a solve actually runs (`beacon.optimise` imports
scipy-free since BN-166); this module is importable on the core install, and
the core-import test holds it to that.
"""
import logging
from collections.abc import Sequence
from typing import Union

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError
from ..optimise.config import MIN_TRACKING_ERROR, OptimisationConfig
from ..optimise.constraints import Constraint
from ..optimise.result import OptimisationResult
from ..optimise.solver import minimise_tracking_error
from ..risk.model import RiskModel
from .calculation import IndexCalculator
from .constructor import IndexDefinition
from .result import IndexResult, daily_weights_frame

logger = logging.getLogger(__name__)

#: The objectives a derived calculation knows how to solve. Only tracking-error
#: distance ships first; the field is the extension point for covariance-aware
#: objectives (design record, "Not in scope yet").
OBJECTIVES = (MIN_TRACKING_ERROR,)

#: Any definition a calculation or a fingerprint can be asked about: a plain
#: index, or a derivation on one (recursively).
AnyIndexDefinition = Union[IndexDefinition, "OptimisedIndexDefinition"]


class OptimisedIndexDefinition:
    """An optimised index: a derivation on a source index, plus identity.

    The source stays first-class — referenced, never copied — so editing the
    parent changes its optimised children at their next calculation, which is
    what "optimise the index I built" means. Chained optimisation (a source
    that is itself optimised) falls out of the recursion for free.

    Args:
        index_id: A unique identifier for the derived index.
        index_name: The common name of the derived index.
        source: The parent — a plain :class:`IndexDefinition`, or another
            :class:`OptimisedIndexDefinition` for a chain.
        objective: What to minimise. Only ``"min_tracking_error"`` exists
            today; an unknown value fails the calculation loudly, naming the
            accepted ones.
        constraints: What the solved weights must satisfy, as
            :class:`~beacon.optimise.constraints.Constraint` instances. Empty
            means the solver's default of full investment alone.
        base_date: First calculation date (YYYY-MM-DD). None inherits the
            source's, which is the usual case: the child lives on the parent's
            calendar.
        base_value: The level the chained path starts at. None inherits the
            source's.
        currency: The derived index's currency. None inherits the source's.
        description: Optional textual description.
        risk_model: RESERVED — carried but unused, mirroring
            :class:`~beacon.optimise.config.OptimisationConfig`. Setting one
            makes the calculation uncacheable (it cannot be keyed yet) and
            changes no result.
    """

    def __init__(self,
                 index_id: str,
                 index_name: str,
                 source: AnyIndexDefinition,
                 objective: str = MIN_TRACKING_ERROR,
                 constraints: Sequence[Constraint] = (),
                 base_date: str | None = None,
                 base_value: float | None = None,
                 currency: str | None = None,
                 description: str | None = None,
                 risk_model: RiskModel | None = None):
        if not index_id:
            raise ValueError("index_id cannot be empty.")
        if not index_name:
            raise ValueError("index_name cannot be empty.")
        if source is None:
            raise ValueError("source must be provided.")
        if base_value is not None and base_value <= 0:
            raise ValueError("base_value, when provided, must be positive.")

        self.index_id: str = index_id
        self.index_name: str = index_name
        self.source: AnyIndexDefinition = source
        self.objective: str = objective
        self.constraints: tuple[Constraint, ...] = tuple(constraints)
        self.description: str | None = description
        self.risk_model: RiskModel | None = risk_model

        self._base_date: pd.Timestamp | None = (pd.Timestamp(base_date)
                                                if base_date is not None else None)
        self._base_value: float | None = base_value
        self._currency: str | None = currency.upper() if currency is not None else None

        logger.info("OptimisedIndexDefinition '%s' created over source '%s' "
                    "with %d constraint(s).",
                    index_id, source.index_id, len(self.constraints))

    @classmethod
    def from_config(cls,
                    index_id: str,
                    index_name: str,
                    source: AnyIndexDefinition,
                    config: OptimisationConfig) -> "OptimisedIndexDefinition":
        """The derivation an :class:`OptimisationConfig` describes.

        One vocabulary for ad-hoc and stored runs (owner decision): the config
        is the stored derivation minus the source, so an ad-hoc `Backtest.run`
        builds an ephemeral definition through here and calculates it exactly
        as a stored one would be.
        """
        return cls(index_id=index_id,
                   index_name=index_name,
                   source=source,
                   objective=config.objective,
                   constraints=config.constraints,
                   risk_model=config.risk_model)

    # -- inherited identity --------------------------------------------------

    @property
    def base_date(self) -> pd.Timestamp:
        """The first calculation date: own when given, else the source's."""
        return self._base_date if self._base_date is not None else self.source.base_date

    @property
    def base_value(self) -> float:
        """The starting level: own when given, else the source's."""
        return self._base_value if self._base_value is not None else self.source.base_value

    @property
    def currency(self) -> str:
        """The index currency: own when given, else the source's."""
        return self._currency if self._currency is not None else self.source.currency

    @property
    def universe_identifiers(self) -> list[str] | None:
        """The investable universe, which is always the source's.

        The derivation holds no universe of its own — it reallocates over
        exactly the names the parent published — so the answer resolves
        through the chain to the root definition's.
        """
        return self.source.universe_identifiers

    def __repr__(self) -> str:
        return (f"OptimisedIndexDefinition(index_id='{self.index_id}', "
                f"index_name='{self.index_name}', "
                f"source='{self.source.index_id}', "
                f"objective='{self.objective}', "
                f"constraints={len(self.constraints)})")


def calculate_derived_index(definition: OptimisedIndexDefinition,
                            data_provider: DataFetcher,
                            start_date: str | None = None,
                            end_date: str | None = None,
                            price_column: str = "CLOSE",
                            parent_result: IndexResult | None = None) -> IndexResult:
    """Calculate an optimised index into a standard :class:`IndexResult`.

    The three-step workflow of the design record: calculate the parent (or
    accept a pre-supplied calculation — the Backtest integration passes its
    cached one), solve the parent's published weights at every rebalance under
    the definition's constraints, then chain the solved weights into the
    derived index's own daily levels.

    Args:
        definition: The derivation to calculate.
        data_provider: Data source for the parent calculation, prices and FX.
        start_date: First date (YYYY-MM-DD). Defaults to the definition's
            base date. Ignored when *parent_result* is supplied, whose own
            window governs.
        end_date: Last date (YYYY-MM-DD). Required unless *parent_result* is
            supplied.
        price_column: Market-data column read as the price.
        parent_result: The source's calculation, when the caller already has
            it. None calculates the source here — recursively, when the
            source is itself optimised.

    Returns:
        IndexResult: Daily levels, divisor history, constituent and weight
        snapshots at exactly the parent's rebalance dates, and the daily
        weights panel — a normal index result, data-bound to *data_provider*.

    Raises:
        CalculationError: If the objective is unknown, the parent produced no
            rebalance snapshots to solve, or a solve is infeasible — the
            solver's own message names the binding conflict.
        ValueError: If no window end is available to calculate the parent.
    """
    check_objective(definition)

    parent = (parent_result if parent_result is not None
              else calculate_source(definition.source, data_provider,
                                    start_date, end_date, price_column))

    if not parent.weight_snapshots:
        raise CalculationError(
            "DerivedIndex",
            f"the source index '{definition.source.index_id}' produced no "
            f"rebalance snapshots, so there are no weights to optimise for "
            f"'{definition.index_id}'.")

    solved = _solved_schedule(definition, parent)

    return _chain_levels(definition, parent, solved,
                         data_provider, price_column).with_data(data_provider)


def check_objective(definition: OptimisedIndexDefinition) -> None:
    """Refuse an objective this module cannot solve, naming the accepted set.

    Checked before anything expensive happens — a full parent calculation, in
    the usual case — so a typo in a stored derivation fails in the time it
    takes to read the document rather than after a minute of arithmetic.

    Raises:
        CalculationError: If the objective is not one of :data:`OBJECTIVES`.
    """
    if definition.objective not in OBJECTIVES:
        raise CalculationError(
            "DerivedIndex",
            f"unknown objective '{definition.objective}' on index "
            f"'{definition.index_id}'. Accepted objectives: "
            f"{', '.join(OBJECTIVES)}.")


def calculate_source(source: AnyIndexDefinition,
                     data_provider: DataFetcher,
                     start_date: str | None = None,
                     end_date: str | None = None,
                     price_column: str = "CLOSE") -> IndexResult:
    """The parent's calculation — recursive when the parent is itself derived.

    Public because the parent's published weights are what a derivation *is*
    defined against, so anything reasoning about a derivation — the
    calculation below, the server's preview — needs them, and needs them from
    one place. A chain resolves here rather than at each caller.

    Args:
        source: The parent definition: rule-driven, or another derivation.
        data_provider: Data source for prices, reference data and FX.
        start_date: First date (YYYY-MM-DD). None uses the source's base date.
        end_date: Last date (YYYY-MM-DD). Required by the calculator.
        price_column: Market-data column read as the price.

    Returns:
        IndexResult: The source's own calculation over that window.
    """
    if isinstance(source, OptimisedIndexDefinition):
        return calculate_derived_index(source, data_provider,
                                       start_date=start_date,
                                       end_date=end_date,
                                       price_column=price_column)

    return IndexCalculator(source, data_provider,
                           price_column=price_column).run(start_date=start_date,
                                                          end_date=end_date)


def solve_snapshot(definition: OptimisedIndexDefinition,
                   source_weights: dict[str, float]) -> OptimisationResult:
    """Solve one of the parent's snapshots under the derivation's constraints.

    The single solve of this module, so the schedule below and any caller
    asking "what would this derivation do at that date" — the server's preview
    — cannot disagree about what the answer is. The whole
    :class:`~beacon.optimise.result.OptimisationResult` is returned rather than
    only its weights, because which constraints bound and how much room the
    rest had left is the interesting half of the answer, and re-deriving it
    from the weights afterwards would be a second implementation of the rules.

    scipy is required inside the solve (BN-166); an infeasible constraint set
    raises there with a message naming the binding conflict, which is exactly
    the loud failure the design demands — nothing is caught here.

    Args:
        definition: The derivation supplying the objective and constraints.
        source_weights: The parent's published weights at one rebalance.

    Returns:
        OptimisationResult: Solved weights, binding constraints, every
        constraint's slack, and the solver's diagnostics.

    Raises:
        CalculationError: If the objective is unknown, or the solve is
            infeasible or fails to converge.
    """
    check_objective(definition)

    return minimise_tracking_error(source_weights,
                                   constraints=list(definition.constraints))


def _solved_schedule(definition: OptimisedIndexDefinition,
                     parent: IndexResult) -> dict[pd.Timestamp, dict[str, float]]:
    """The parent's snapshots, each solved under the derivation's constraints."""
    schedule: dict[pd.Timestamp, dict[str, float]] = {}

    for date in sorted(parent.weight_snapshots):
        result = solve_snapshot(definition, parent.weight_snapshots[date])
        schedule[date] = {str(asset): float(value)
                          for asset, value in result.weights.items()}

    logger.info("Solved %d rebalance(s) of '%s' under %d constraint(s) for "
                "'%s'.",
                len(schedule), definition.source.index_id,
                len(definition.constraints), definition.index_id)

    return schedule


def _chain_levels(definition: OptimisedIndexDefinition,
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
    unit_values = _unit_value_panel(definition, solved, days,
                                    data_provider, price_column)

    levels: dict[pd.Timestamp, float] = {}
    divisors: dict[pd.Timestamp, float] = {}
    constituent_snapshots: dict[pd.Timestamp, list[str]] = {}
    weight_snapshots: dict[pd.Timestamp, dict[str, float]] = {}
    daily_records: list[dict[str, object]] = []

    units: dict[str, float] = {}
    level = definition.base_value
    divisor = 0.0

    for day in days:
        values: dict[str, float] = {}

        if day in solved:
            if divisor <= 0.0:
                # Inception: the chained path starts at the base value.
                aggregate = definition.base_value
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
                    day.date(), definition.index_id)

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
                definition.index_id, len(days), len(weight_snapshots), level)

    return IndexResult(index_id=definition.index_id,
                       index_levels=pd.Series(levels),
                       divisor_history=pd.Series(divisors),
                       constituent_snapshots=constituent_snapshots,
                       weight_snapshots=weight_snapshots,
                       daily_weights=daily_weights_frame(daily_records))


def _unit_value_panel(definition: OptimisedIndexDefinition,
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

        currency = _currency_of(data_provider, asset, definition.currency)
        if currency != definition.currency:
            if currency not in rates:
                rates[currency] = _rate_series(data_provider, currency,
                                               definition.currency, days)

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
