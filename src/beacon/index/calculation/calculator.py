# src/beacon/index/calculation/calculator.py
"""
Module for the IndexCalculator, responsible for the logic of
constituent selection, weighting, index level calculation, and corporate action adjustments.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity
from ...data.fetcher import DataFetcher
from ...exceptions import CalculationError
from ..capping import CapReport, apply_cap
from ..constructor import IndexDefinition
from ..result import IndexResult, daily_weights_frame
from ..schedule import effective_date, sessions
from .corporate_actions import CorporateActionsMixin
from .deletions import DeletionMixin
from .market_values import MarketValuesMixin
from .selection import (
    UNIVERSE_POSITION,
    SelectionResult,
    SelectionStep,
    select_with_provenance,
)
from .total_return import REINVESTING, TotalReturnMixin, withholding_for

logger = logging.getLogger(__name__)


def weight_rows(date: pd.Timestamp,
                units: dict[Asset, float],
                values: dict[Asset, float]) -> list[dict[str, object]]:
    """One record per constituent for one day: what was held, and its share.

    The weights are *realised* shares of the day's aggregate — value over
    total — so they drift with prices between rebalances, and they renormalise
    the moment a name is deleted. That is why the panel is recorded here
    rather than derived later from the rebalance snapshot: the two only agree
    on the rebalance date itself.

    Plain dicts appended during the run and converted once at the end, the
    pattern the backtest engine already uses for its own records — there is no
    per-day object to build and throw away.

    Args:
        date: The calculation day.
        units: What the index holds after the day's events.
        values: Those holdings valued for *date*, from
            :meth:`~.market_values.MarketValuesMixin.holding_values`.

    Returns:
        list: Records keyed by :data:`~beacon.index.result.DAILY_WEIGHT_COLUMNS`.
        A day whose holdings are worth nothing at all records nothing, because
        it has no weights to record — the level is carried forward on such a
        day, and a row of zeros would read as "held nothing" rather than
        "could not be valued".
    """
    total = sum(values.values())

    if total <= 0.0:
        return []

    return [{"DATE": date,
             "IDENTIFIER": asset.asset_id,
             "AMOUNT": units[asset],
             "WEIGHT": values[asset] / total}
            for asset in units]


class IndexCalculator(MarketValuesMixin, DeletionMixin,
                      TotalReturnMixin, CorporateActionsMixin):
    """
    Stateless index calculator. Accepts an IndexDefinition and DataFetcher,
    and provides methods for constituent selection, weighting, index level
    calculation, and corporate action adjustments. All state is passed
    through method parameters and return values.
    """
    def __init__(self,
                 index_definition: IndexDefinition,
                 data_provider: DataFetcher,
                 price_column: str = "CLOSE"):
        """
        Initializes the IndexCalculator.

        Args:
            index_definition: The IndexDefinition object that specifies the index rules.
            data_provider: A DataFetcher instance to access market and asset data.
            price_column: Market-data column read as the constituent price when
                computing market values. Defaults to ``"CLOSE"``.
        """
        if not index_definition:
            raise ValueError("index_definition must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")

        self.definition: IndexDefinition = index_definition
        self.data: DataFetcher = data_provider
        self.price_column: str = price_column

        logger.info(f"IndexCalculator initialized for index '{self.definition.index_name}'.")


    def _get_universe(self,
                      date: pd.Timestamp) -> list[Asset]:
        """Resolve universe_identifiers from the IndexDefinition into Asset objects.

        Uses ``self.data.fetch_reference_data`` to look up metadata for each
        identifier and constructs :class:`Equity` objects.  Identifiers that
        cannot be resolved are logged as warnings and skipped.

        Args:
            date: Point-in-time date for reference data lookup.

        Returns:
            A list of Asset objects corresponding to resolvable identifiers.
        """
        identifiers = self.definition.universe_identifiers
        if identifiers is None:
            logger.warning(
                f"universe_identifiers is None for index '{self.definition.index_name}'. "
                "Returning empty universe."
            )
            return []

        assets: list[Asset] = []
        date_str = date.strftime('%Y-%m-%d')

        for identifier in identifiers:
            try:
                ref_df = self.data.fetch_reference_data(identifier, date_str)
                if ref_df.empty:
                    logger.warning(
                        f"_get_universe: No reference data for '{identifier}' on "
                        f"{date_str}. Skipping.")
                    continue

                row = ref_df.iloc[0]
                asset = Equity(
                    name=row.get("NAME", identifier),
                    currency=row.get("CURRENCY", self.definition.currency),
                    ticker=identifier,
                    exchange=row.get("EXCHANGE", "UNKNOWN"),
                )
                assets.append(asset)
            except Exception as e:
                logger.warning(f"_get_universe: Failed to resolve '{identifier}': {e}. Skipping.")

        logger.info(
            f"_get_universe: Resolved {len(assets)}/{len(identifiers)} identifiers "
            f"for '{self.definition.index_name}' on {date_str}."
        )
        return assets

    def resolve_universe(self,
                         date: pd.Timestamp) -> list[Asset]:
        """Resolve the definition's universe identifiers into Asset objects.

        The public entry point for universe resolution, for callers outside
        the calculation loop — the constituent preview, for one. Delegates to
        the internal implementation, so anything that stubs that also governs
        this.

        Args:
            date: Point-in-time date for the reference-data lookup.

        Returns:
            list[Asset]: Assets for every identifier that resolved.
        """
        return self._get_universe(date)

    def select_constituents(self,
                            universe: list[Asset],
                            current_date: pd.Timestamp) -> list[Asset]:
        """
        Selects index constituents from a given universe based on eligibility rules.

        A thin projection of :meth:`select_with_provenance`: the survivors, with
        the record of which rule removed each excluded name discarded. Callers
        wanting that record — the preview waterfall, anything answering "why is
        this name missing" — should use the fuller method rather than repeating
        the walk, which is what BN-102 existed to stop.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
            current_date: The date for which selection is being made.

        Returns:
            A list of Asset objects that are eligible for the index.
        """
        return self.select_with_provenance(universe, current_date).survivors

    def select_with_provenance(self,
                               universe: list[Asset],
                               current_date: pd.Timestamp) -> SelectionResult:
        """Select constituents, keeping the record of how the universe narrowed.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
            current_date: The date for which selection is being made.

        Returns:
            SelectionResult: Survivors, one step per rule, and the position of
            the rule that excluded each removed asset.
        """
        logger.info(
            f"[{current_date.strftime('%Y-%m-%d')}] Selecting constituents for "
            f"'{self.definition.index_name}'. Universe size: {len(universe)}")

        if not universe:
            logger.warning("Constituent selection called with an empty universe.")

            return SelectionResult(survivors=[],
                                   steps=[SelectionStep(position=UNIVERSE_POSITION,
                                                        remaining=0)])

        result = select_with_provenance(universe,
                                        self.definition.eligibility_rules,
                                        current_date,
                                        self.data)

        logger.info(
            f"Selected {len(result.survivors)} constituents for "
            f"'{self.definition.index_name}'.")

        return result

    def calculate_constituent_weights(self,
                                      constituents: list[Asset],
                                      current_date: pd.Timestamp) -> dict[Asset, float]:
        """
        Calculates the weights for the given constituents based on the index's weighting scheme.

        Args:
            constituents: A list of Asset objects that are part of the index.
            current_date: The date for which weights are calculated.

        Returns:
            A dictionary mapping each Asset to its float weight. Sum of weights should be 1.0.
        """
        date_str = current_date.strftime('%Y-%m-%d')
        if not constituents:
            logger.warning(
                f"[{date_str}] Calculating weights for an empty list of "
                f"constituents for '{self.definition.index_name}'.")
            return {}

        logger.info(
            f"[{date_str}] Calculating weights for {len(constituents)} "
            f"constituents of '{self.definition.index_name}'.")

        try:
            weights = self.definition.weighting_scheme.calculate_weights(
                constituents, current_date, self.data
            )
        except Exception as e:
            logger.error(
                f"Error applying weighting scheme "
                f"{self.definition.weighting_scheme.scheme_name}: {e}")
            raise CalculationError(
                calculation_name=f"WeightingScheme-{self.definition.weighting_scheme.scheme_name}",
                details=str(e)) from e

        # Normalize weights to sum to 1, if not already
        #todo: check if normalisation is needed based on the weighting scheme's
        # output. Some schemes may guarantee this.
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-9 and weight_sum != 0:
            logger.warning(
                f"Weights from scheme {self.definition.weighting_scheme.scheme_name} "
                f"sum to {weight_sum}. Normalizing.")
            weights = {asset: w / weight_sum for asset, w in weights.items()}
        elif weight_sum == 0 and weights:
             logger.error(
                 f"Calculated weights sum to zero for {len(weights)} "
                 f"constituents. Cannot normalize.")

        logger.info(f"Weights calculated for '{self.definition.index_name}'.")
        return weights

    def cap_weights(self,
                    weights: dict[Asset, float]) -> tuple[dict[Asset, float], CapReport]:
        """Apply the definition's cap, returning the weights and a report.

        Capping happens here rather than inside a weighting scheme so that it
        composes with every scheme, and it returns its report rather than
        storing one so the calculator stays stateless and `run()` stays
        idempotent.

        Args:
            weights: Normalised weights keyed by Asset.

        Returns:
            tuple: The capped weights and a CapReport. With no cap configured
            the weights are returned unchanged and the report is empty.
        """
        cap = self.definition.max_constituent_weight
        if cap is None or not weights:
            return weights, CapReport(cap=cap)

        # apply_cap works on identifiers so its report can name constituents
        # without depending on the asset classes.
        by_id = {asset.asset_id: weight for asset, weight in weights.items()}
        capped, report = apply_cap(by_id, cap)

        return {asset: capped[asset.asset_id] for asset in weights}, report

    def initialize_divisor(self,
                           initial_total_market_value: float) -> float:
        """
        Calculates the initial divisor for the index on its base_date.
        Divisor = Initial Total Market Value / Base Index Value.

        Args:
            initial_total_market_value: The sum of (price * shares * fx_rate *
                free_float_if_applicable) for all base constituents on the
                base_date, expressed in index currency.

        Returns:
            The initial divisor as a float.
        """
        if initial_total_market_value <= 0:
            logger.error("Initial total market value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization",
                                    "Initial total market value is non-positive.")
        if self.definition.base_value <= 0:
            logger.error("Base index value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization", "Base index value is non-positive.")

        divisor = initial_total_market_value / self.definition.base_value
        logger.info(
            f"Divisor for '{self.definition.index_name}' initialized to: {divisor:.4f} "
            f"(Initial Market Value: {initial_total_market_value:.2f}, "
            f"Base Value: {self.definition.base_value})")
        return divisor

    @staticmethod
    def adjust_divisor_for_rebalance(old_divisor: float,
                                     old_market_value: float,
                                     new_market_value: float) -> float:
        """Adjust the divisor to maintain index level continuity across a rebalance.

        When index composition or weights change, the total market value shifts.
        To prevent an artificial jump in the index level the divisor is scaled:

            new_divisor = old_divisor * (new_market_value / old_market_value)

        This guarantees: level_before == level_after.

        Args:
            old_divisor: The divisor in effect before the rebalance.
            old_market_value: Aggregate market value under the **old** composition.
            new_market_value: Aggregate market value under the **new** composition.

        Returns:
            The adjusted divisor.

        Raises:
            ValueError: If *old_divisor*, *old_market_value* or *new_market_value*
                is zero or negative.
        """
        if old_divisor <= 0:
            raise ValueError(f"old_divisor must be positive, got {old_divisor}")
        if old_market_value <= 0:
            raise ValueError(f"old_market_value must be positive, got {old_market_value}")
        if new_market_value <= 0:
            raise ValueError(f"new_market_value must be positive, got {new_market_value}")

        new_divisor = old_divisor * (new_market_value / old_market_value)

        logger.info(
            f"Divisor adjusted for rebalance: {old_divisor:.6f} -> {new_divisor:.6f} "
            f"(old_mv={old_market_value:.2f}, new_mv={new_market_value:.2f})"
        )
        return new_divisor

    #todo: run() is currently iterating through all dates, this function should be
    # vectorised for efficiency.
    def run(self,
            start_date: str | None = None,
            end_date: str | None = None) -> IndexResult:
        """Run the full index calculation over a date range.

        Iterates through business days from *start_date* to *end_date*,
        handling three day types:

        1. **Base date** – resolve universe, select constituents, compute
           weights, initialise divisor, set level = base_value.
        2. **Rebalance date** – reconstitute (re-resolve universe, re-select,
           re-weight) and adjust divisor for continuity.
        3. **Regular day** – compute index level using current constituents
           and weights.

        The method is idempotent: it carries no state between calls.

        Args:
            start_date: First calculation date (YYYY-MM-DD).  Defaults to
                ``definition.base_date``.
            end_date: Last calculation date (YYYY-MM-DD).  Required.

        Returns:
            An :class:`IndexResult` containing index levels, divisor history,
            constituent snapshots, weight snapshots, and the daily weights
            panel — one row per constituent per day, recorded as the loop
            goes, since the state it holds each day is path-dependent and
            cannot be reconstructed from the rebalance snapshots afterwards.

        Raises:
            ValueError: If *end_date* is not provided or precedes the base date.
        """
        base_date = self.definition.base_date
        pd_start = pd.Timestamp(start_date) if start_date else base_date
        if end_date is None:
            raise ValueError("end_date must be provided.")
        pd_end = pd.Timestamp(end_date)

        if pd_end < base_date:
            raise ValueError(
                f"end_date ({pd_end.strftime('%Y-%m-%d')}) precedes "
                f"base_date ({base_date.strftime('%Y-%m-%d')})."
            )

        # Ensure start is not before base_date
        if pd_start < base_date:
            pd_start = base_date

        trading_days = pd.bdate_range(start=pd_start, end=pd_end)
        if trading_days.empty:
            logger.warning("No trading days in the requested range.")
            return IndexResult(
                index_id=self.definition.index_id,
                index_levels=pd.Series(dtype=float),
                divisor_history=pd.Series(dtype=float),
                constituent_snapshots={},
                weight_snapshots={},
            )

        # Pre-compute rebalance dates (excluding base date which is handled separately)
        rebalance_dates_list = self.definition.get_rebalance_dates(
            pd_start.strftime('%Y-%m-%d'),
            pd_end.strftime('%Y-%m-%d'),
        )
        # Announced on one date, in force on another. With no lag the two
        # coincide and this mapping is the identity, which is what keeps every
        # index defined before BN-126 producing identical levels.
        # The panel is built only when a lag actually applies. An index
        # without one does no calendar work at all, which keeps this change
        # free for every index defined before it.
        lag = self.definition.effective_lag_sessions
        panel = (sessions(pd_start, pd_end, self.definition.calendar)
                 if lag > 0 else pd.DatetimeIndex([]))
        effective_for = {announced: effective_date(announced, lag, panel)
                         for announced in rebalance_dates_list
                         if announced != base_date}

        # Keyed by the date the composition is *applied*, since that is the day
        # the loop has to act on. Two announcements landing on one effective
        # date would be a schedule shorter than its own lag; the later wins,
        # which is the one a reader would expect to be in force.
        announced_for = {effective: announced
                         for announced, effective in effective_for.items()}
        rebalance_dates = set(announced_for)
        rebalance_dates.discard(base_date)

        # Accumulators
        index_levels: dict[pd.Timestamp, float] = {}
        divisor_values: dict[pd.Timestamp, float] = {}
        constituent_snapshots: dict[pd.Timestamp, list[str]] = {}
        weight_snapshots: dict[pd.Timestamp, dict[str, float]] = {}
        # Only rebalances where the cap actually bound get an entry, so an
        # uncapped index carries an empty mapping rather than noise.
        cap_reports: dict[pd.Timestamp, CapReport] = {}
        # Effective date -> announcement date, populated only where the two
        # differ. An index with no lag carries an empty mapping, so its
        # presence is itself the signal that a lag applies.
        announcements: dict[pd.Timestamp, pd.Timestamp] = {}
        # One record per constituent per day. The loop already holds the true
        # state of the index on every date and used to discard it, keeping
        # only levels, divisors and the rebalance snapshots.
        #
        # Held as dicts until the run ends and converted once, which is the
        # pattern the backtest engine uses. The cost is at the peak rather
        # than at rest: a pending record measures ~240 bytes against ~19 in
        # the frame, so a 6,000-name decade would hold ~3.6 GB here before
        # collapsing to ~286 MB. Chunked conversion is the fix if that ever
        # binds; nothing in this repository runs at that size yet.
        daily_records: list[dict[str, object]] = []

        # Cash distributions, loaded once. A price index skips this entirely,
        # so it costs nothing and reads no action history — which is what keeps
        # every index defined before BN-125 producing identical levels.
        reinvesting = self.definition.return_type in REINVESTING
        distributions = self.cash_distribution_schedule() if reinvesting else {}

        # When each name stops being listed, resolved once. An index over a
        # universe where nothing is ever delisted gets an empty mapping and
        # pays for one `if` per day.
        delistings = self.delisting_schedule()
        previous_date = base_date
        withholding = withholding_for(self.definition.return_type,
                                      self.definition.withholding_tax_rate)

        # Running state. `units` is what the index actually holds: fixed
        # between rebalances, so weights drift with relative performance
        # instead of being reset every day.
        constituents: list[Asset] = []
        weights: dict[Asset, float] = {}
        units: dict[Asset, float] = {}
        divisor: float = 0.0
        level: float = self.definition.base_value

        for date in trading_days:
            # Today's holdings, valued. Empty on a day the index has no
            # holdings to value, which records no weights.
            values: dict[Asset, float] = {}

            if date == base_date:
                # --- Base date initialisation ---
                constituents_raw = self._get_universe(date)
                constituents = self.select_constituents(constituents_raw, date)
                weights = self.calculate_constituent_weights(constituents, date)
                weights, cap_report = self.cap_weights(weights)
                if cap_report.was_capped:
                    cap_reports[date] = cap_report

                # The aggregate the index represents is still the constituents'
                # total market value, which keeps the divisor's magnitude and
                # meaning unchanged. What changes is that the holdings are now
                # units derived from the weights, rather than shares
                # outstanding — so the methodology actually drives the level.
                mv_map = self._get_constituent_market_values(weights, date)
                total_mv = sum(mv_map.values())
                units = self.index_units(weights, total_mv, date)
                values = self.holding_values(units, date)

                if total_mv > 0:
                    divisor = self.initialize_divisor(total_mv)
                else:
                    logger.warning(
                        f"Zero market value on base date {date.strftime('%Y-%m-%d')}. "
                        "Setting divisor to 1.0."
                    )
                    divisor = 1.0

                level = self.definition.base_value

                # Record snapshots
                constituent_snapshots[date] = [a.asset_id for a in constituents]
                weight_snapshots[date] = {a.asset_id: w for a, w in weights.items()}

            elif date in rebalance_dates:
                # --- Rebalance date ---
                # Value the outgoing holdings at today's prices. This is the
                # level the new composition has to start from, which is what
                # the divisor adjustment preserves.
                old_aggregate = self.aggregate_value(units, date)

                # The outgoing holdings are the ones that went ex today, so the
                # reinvestment belongs to them and has to happen before the
                # composition changes. Adjusting the divisor here composes with
                # the continuity adjustment below: that one preserves whatever
                # level is in force, which now includes the distribution.
                if reinvesting:
                    paid = distributions.get(date, {})
                    divisor = self.reinvest(
                        divisor, old_aggregate,
                        self.distribution_received(
                            units, paid, withholding,
                            self.distribution_rates(
                                paid, units, date,
                                self.definition.currency)))

                # Reconstitute as of the *announcement*: the constituent list
                # and target weights are what was published, even though they
                # are implemented at today's prices. Selecting on the effective
                # date instead would let a name that qualified when the index
                # was announced be dropped by a price move in between, which is
                # not what a published composition means.
                announced_on = announced_for.get(date, date)

                constituents_raw = self._get_universe(announced_on)
                constituents = self.select_constituents(constituents_raw,
                                                        announced_on)
                weights = self.calculate_constituent_weights(constituents,
                                                             announced_on)
                weights, cap_report = self.cap_weights(weights)
                if cap_report.was_capped:
                    cap_reports[date] = cap_report

                if announced_on != date:
                    announcements[date] = announced_on

                # Rebuild the holdings to the new weights, scaled to the
                # constituents' total market value so the divisor keeps the
                # magnitude it has always had.
                new_mv_map = self._get_constituent_market_values(weights, date)
                new_total_mv = sum(new_mv_map.values())
                units = self.index_units(weights, new_total_mv, date)
                values = self.holding_values(units, date)
                new_aggregate = float(sum(values.values()))

                # Adjust divisor for continuity
                if old_aggregate > 0 and new_aggregate > 0:
                    divisor = self.adjust_divisor_for_rebalance(
                        divisor, old_aggregate, new_aggregate
                    )
                elif new_aggregate > 0:
                    divisor = new_aggregate / level if level > 0 else 1.0

                # Compute level with adjusted divisor
                level = new_aggregate / divisor if divisor > 0 else level

                # Record snapshots
                constituent_snapshots[date] = [a.asset_id for a in constituents]
                weight_snapshots[date] = {a.asset_id: w for a, w in weights.items()}

            else:
                # --- Regular trading day ---
                if not constituents or divisor <= 0:
                    # Before base date initialisation or no constituents
                    pass
                else:
                    # Before anything else: a holding that stopped being
                    # listed cannot be valued, and leaving it in would report
                    # its whole weight as a loss on the day it went.
                    units, divisor, deleted = self.apply_deletions(
                        units, divisor, date, delistings, previous_date)

                    if deleted:
                        constituents = [asset for asset in constituents
                                        if asset.asset_id not in set(deleted)]
                        weights = {asset: weight
                                   for asset, weight in weights.items()
                                   if asset.asset_id not in set(deleted)}

                    # Valued once, then used three times over: to reinvest
                    # into, to set the level, and to record the day's weights.
                    # A price lookup per holding is the run's dominant cost.
                    values = self.holding_values(units, date)
                    aggregate = float(sum(values.values()))

                    if reinvesting:
                        paid = distributions.get(date, {})
                        divisor = self.reinvest(
                            divisor,
                            aggregate,
                            self.distribution_received(
                                units, paid, withholding,
                                self.distribution_rates(
                                paid, units, date,
                                self.definition.currency)))

                    level = self.level_from_units(
                        units=units,
                        divisor=divisor,
                        current_date=date,
                        previous_index_level=level,
                        values=values,
                    )

            index_levels[date] = level
            divisor_values[date] = divisor
            daily_records.extend(weight_rows(date, units, values))

            # Deletions are valued on the last day the leaver still had a
            # price, so the loop has to remember which day that was.
            previous_date = date

        logger.info(
            f"run() completed for '{self.definition.index_name}': "
            f"{len(trading_days)} trading days, "
            f"{len(constituent_snapshots)} rebalance(s), "
            f"{len(daily_records)} daily weight record(s)."
        )

        return IndexResult(
            index_id=self.definition.index_id,
            index_levels=pd.Series(index_levels),
            divisor_history=pd.Series(divisor_values),
            constituent_snapshots=constituent_snapshots,
            weight_snapshots=weight_snapshots,
            cap_reports=cap_reports,
            announcement_dates=announcements,
            daily_weights=daily_weights_frame(daily_records),
        ).with_data(self.data)

    def run_daily_calculation(self,
                              current_date: pd.Timestamp,
                              constituents: list[Asset],
                              weights: dict[Asset, float],
                              previous_index_level: float,
                              previous_divisor: float) -> tuple[float, float]:
        """
        Runs a single day's index calculation process.

        Args:
            current_date: The date for which to perform calculations.
            constituents: Current index constituents.
            weights: Current constituent weights.
            previous_index_level: Index level from the previous period.
            previous_divisor: Divisor from the previous period.

        Returns:
            Tuple of (new_index_level, new_divisor).
        """
        divisor = previous_divisor

        if divisor is None or divisor <= 0:
            if current_date == self.definition.base_date:
                if not constituents:
                    raise ValueError(
                        "Base date calculation: Constituents not provided. "
                        "Cannot initialize divisor.")
                base_day_values = self._get_constituent_market_values(
                    constituents_with_weights=dict.fromkeys(constituents, 0),
                    current_date=current_date
                )
                initial_mv = sum(base_day_values.values())
                if initial_mv > 0:
                    divisor = self.initialize_divisor(initial_mv)
                else:
                    raise ValueError(
                        f"Cannot initialize divisor on base date {current_date} due to "
                        "zero or negative market value.")
            else:
                raise ValueError("Divisor not initialized for index calculation.")

        new_level, final_divisor = self.calculate_index_level(
            current_date=current_date,
            constituents=constituents,
            weights=weights,
            divisor=divisor,
            previous_index_level=previous_index_level,
        )

        return new_level, final_divisor
