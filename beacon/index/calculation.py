# beacon/index/calculation.py
"""
Module for the IndexCalculator, responsible for the logic of
constituent selection, weighting, index level calculation, and corporate action adjustments.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
import logging

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from .constructor import IndexDefinition
    from ..asset.base import Asset
    from ..data.fetcher import DataFetcher
    from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

class IndexCalculator:
    """
    Stateless index calculator. Accepts an IndexDefinition and DataFetcher,
    and provides methods for constituent selection, weighting, index level
    calculation, and corporate action adjustments. All state is passed
    through method parameters and return values.
    """
    def __init__(self,
                 index_definition: 'IndexDefinition',
                 data_provider: 'DataFetcher'):
        """
        Initializes the IndexCalculator.

        Args:
            index_definition: The IndexDefinition object that specifies the index rules.
            data_provider: A DataFetcher instance to access market and asset data.
        """
        if not index_definition:
            raise ValueError("index_definition must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")

        self.definition: 'IndexDefinition' = index_definition
        self.data: 'DataFetcher' = data_provider

        logger.info(f"IndexCalculator initialized for index '{self.definition.index_name}'.")


    def _get_universe(self, date: pd.Timestamp) -> List['Asset']:
        """Resolve universe_identifiers from the IndexDefinition into Asset objects.

        Uses ``self.data.fetch_reference_data`` to look up metadata for each
        identifier and constructs :class:`Equity` objects.  Identifiers that
        cannot be resolved are logged as warnings and skipped.

        Args:
            date: Point-in-time date for reference data lookup.

        Returns:
            A list of Asset objects corresponding to resolvable identifiers.
        """
        from ..asset.equity import Equity

        identifiers = self.definition.universe_identifiers
        if identifiers is None:
            logger.warning(
                f"universe_identifiers is None for index '{self.definition.index_name}'. "
                "Returning empty universe."
            )
            return []

        assets: List['Asset'] = []
        date_str = date.strftime('%Y-%m-%d')

        for identifier in identifiers:
            try:
                ref_df = self.data.fetch_reference_data(identifier, date_str)
                if ref_df.empty:
                    logger.warning(f"_get_universe: No reference data for '{identifier}' on {date_str}. Skipping.")
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

    def select_constituents(self, universe: List['Asset'], current_date: pd.Timestamp) -> List['Asset']:
        """
        Selects index constituents from a given universe based on eligibility rules.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
            current_date: The date for which selection is being made.

        Returns:
            A list of Asset objects that are eligible for the index.
        """
        logger.info(f"[{current_date.strftime('%Y-%m-%d')}] Selecting constituents for '{self.definition.index_name}'. Universe size: {len(universe)}")
        eligible_constituents: List['Asset'] = []
        if not universe:
            logger.warning("Constituent selection called with an empty universe.")
            return []

        for asset in universe:
            is_eligible_for_asset = True
            for rule in self.definition.eligibility_rules:
                try:
                    if not rule.is_eligible(asset, current_date, self.data):
                        is_eligible_for_asset = False
                        logger.debug(f"Asset {asset.asset_id} failed eligibility rule: {rule.rule_name}")
                        break
                except Exception as e:
                    logger.error(f"Error applying eligibility rule {rule.rule_name} to asset {asset.asset_id}: {e}")
                    is_eligible_for_asset = False
                    break

            if is_eligible_for_asset:
                eligible_constituents.append(asset)
                logger.debug(f"Asset {asset.asset_id} passed all eligibility rules.")

        logger.info(f"Selected {len(eligible_constituents)} constituents for '{self.definition.index_name}'.")
        return eligible_constituents

    def calculate_constituent_weights(self, constituents: List['Asset'], current_date: pd.Timestamp) -> Dict['Asset', float]:
        """
        Calculates the weights for the given constituents based on the index's weighting scheme.

        Args:
            constituents: A list of Asset objects that are part of the index.
            current_date: The date for which weights are calculated.

        Returns:
            A dictionary mapping each Asset to its float weight. Sum of weights should be 1.0.
        """
        if not constituents:
            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] Calculating weights for an empty list of constituents for '{self.definition.index_name}'.")
            return {}

        logger.info(f"[{current_date.strftime('%Y-%m-%d')}] Calculating weights for {len(constituents)} constituents of '{self.definition.index_name}'.")

        try:
            weights = self.definition.weighting_scheme.calculate_weights(
                constituents, current_date, self.data
            )
        except Exception as e:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error(f"Error applying weighting scheme {self.definition.weighting_scheme.scheme_name}: {e}")
            raise CalculationError(calculation_name=f"WeightingScheme-{self.definition.weighting_scheme.scheme_name}", details=str(e))

        # Normalize weights to sum to 1, if not already
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-9 and weight_sum != 0:
            logger.warning(f"Weights from scheme {self.definition.weighting_scheme.scheme_name} sum to {weight_sum}. Normalizing.")
            weights = {asset: w / weight_sum for asset, w in weights.items()}
        elif weight_sum == 0 and weights:
             logger.error(f"Calculated weights sum to zero for {len(weights)} constituents. Cannot normalize.")

        logger.info(f"Weights calculated for '{self.definition.index_name}'.")
        return weights

    def initialize_divisor(self, initial_total_market_value: float) -> float:
        """
        Calculates the initial divisor for the index on its base_date.
        Divisor = Initial Total Market Value / Base Index Value.

        Args:
            initial_total_market_value: The sum of (price * shares * fx_rate * free_float_if_applicable)
                                        for all base constituents on the base_date, expressed in index currency.

        Returns:
            The initial divisor as a float.
        """
        if initial_total_market_value <= 0:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error("Initial total market value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization", "Initial total market value is non-positive.")
        if self.definition.base_value <= 0:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error("Base index value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization", "Base index value is non-positive.")

        divisor = initial_total_market_value / self.definition.base_value
        logger.info(f"Divisor for '{self.definition.index_name}' initialized to: {divisor:.4f} "
                    f"(Initial Market Value: {initial_total_market_value:.2f}, Base Value: {self.definition.base_value})")
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

    def _get_constituent_market_values(self,
                                       constituents_with_weights: Dict['Asset', float],
                                       current_date: pd.Timestamp) -> Dict['Asset', float]:
        """
        Helper to get current market values for constituents.
        Market Value = Price * Shares * FX_Rate_to_Index_Currency * (FreeFloat if applicable)

        This method computes Sum(Price_t * Shares_t * [FF_t] * [FX_t])
        i.e. the "Adjusted Total Market Cap" of the index constituents.
        """
        from ..asset.equity import Equity # Specific check for equity attributes

        constituent_market_values: Dict['Asset', float] = {}

        for asset, weight in constituents_with_weights.items():
            if not isinstance(asset, Equity):
                logger.warning(f"Asset {asset.asset_id} is not Equity. Skipping market value calculation.")
                continue
            try:
                price_df = self.data.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                if price_df.empty or pd.isna(price_df['Adj Close'].iloc[0]):
                    logger.warning(f"_get_constituent_market_values: No price for {asset.ticker}. Value is 0.")
                    constituent_market_values[asset] = 0.0
                    continue
                current_price = price_df['Adj Close'].iloc[0]

                shares = self.data.fetch_shares_outstanding(asset.ticker, current_date.strftime('%Y-%m-%d'))
                if shares is None or shares <= 0:
                    logger.warning(f"_get_constituent_market_values: No shares for {asset.ticker}. Value is 0.")
                    constituent_market_values[asset] = 0.0
                    continue

                market_value_local_ccy = current_price * shares

                # Apply Free Float if used by weighting scheme
                if hasattr(self.definition.weighting_scheme, 'use_free_float') and \
                   self.definition.weighting_scheme.use_free_float:
                    ff_factor = self.data.fetch_free_float_factor(asset.ticker, current_date.strftime('%Y-%m-%d'))
                    if ff_factor is not None and 0.0 <= ff_factor <= 1.0:
                        market_value_local_ccy *= ff_factor
                    else:
                        logger.warning(f"Missing or invalid free-float for {asset.ticker}, not applying to market value.")

                # FX Conversion to Index Currency
                fx_rate = 1.0
                if asset.currency.upper() != self.definition.currency.upper():
                    fx_series = self.data.fetch_fx_rates(asset.currency, self.definition.currency,
                                                         current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                    if not fx_series.empty:
                        fx_rate = fx_series.iloc[0]
                    else:
                        logger.warning(f"No FX rate found for {asset.currency}/{self.definition.currency} on {current_date}. Using 1.0.")
                        constituent_market_values[asset] = 0.0
                        continue

                adj_market_value_index_ccy = market_value_local_ccy * fx_rate
                constituent_market_values[asset] = adj_market_value_index_ccy

            except Exception as e:
                logger.error(f"Error calculating market value for {asset.ticker}: {e}")
                constituent_market_values[asset] = 0.0

        return constituent_market_values


    def calculate_index_level(self,
                              current_date: pd.Timestamp,
                              constituents: List['Asset'],
                              weights: Dict['Asset', float],
                              divisor: float,
                              previous_index_level: float
                             ) -> Tuple[float, float]:
        """
        Calculates the current index level using a Laspeyres-type formula:
        Index Level = Sum of Current Market Values of Constituents / Current Divisor.

        Args:
            current_date: The date for which to calculate the index level.
            constituents: Current index constituents.
            weights: Current constituent weights.
            divisor: The current index divisor.
            previous_index_level: The index level from the previous calculation period.

        Returns:
            A tuple of (new_index_level, divisor).
        """
        if divisor <= 0:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error(f"Invalid divisor: {divisor}. Cannot calculate index level.")
            raise CalculationError("IndexLevelCalculation", f"Invalid divisor: {divisor}")

        if not constituents:
            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] No current constituents to calculate index level for '{self.definition.index_name}'. Returning previous level.")
            return previous_index_level, divisor

        constituent_values_map = self._get_constituent_market_values(
            constituents_with_weights=weights,
            current_date=current_date
        )
        current_total_adjusted_market_value = sum(constituent_values_map.values())

        if current_total_adjusted_market_value < 0:
            logger.warning(f"Total adjusted market value is negative: {current_total_adjusted_market_value}. Using 0.")
            current_total_adjusted_market_value = 0.0

        new_index_level = current_total_adjusted_market_value / divisor

        logger.debug(f"[{current_date.strftime('%Y-%m-%d')}] Index '{self.definition.index_name}': "
                     f"Total Market Value = {current_total_adjusted_market_value:.2f}, Divisor = {divisor:.4f}, "
                     f"New Level = {new_index_level:.4f}")
        return new_index_level, divisor


    def handle_corporate_action(self,
                                action: Dict[str, Any],
                                constituents: List['Asset'],
                                current_total_market_value_before_ca: float,
                                current_divisor_before_ca: float
                               ) -> float:
        """
        Adjusts the index divisor for specific corporate actions to maintain index continuity.

        Args:
            action: A dictionary describing the corporate action. Must include 'type', 'asset',
                    'value', and 'ex_date'.
            constituents: Current index constituents.
            current_total_market_value_before_ca: The sum of market values of all constituents
                                                  just before the CA's impact.
            current_divisor_before_ca: The divisor in effect before this corporate action.

        Returns:
            The new adjusted divisor (float).
        """
        action_type = action.get('type', '').upper()
        asset_involved = action.get('asset')
        value = action.get('value')
        ex_date = pd.Timestamp(action.get('ex_date'))

        logger.info(f"[{ex_date.strftime('%Y-%m-%d')}] Handling CA: {action_type} for asset {asset_involved.asset_id if asset_involved else 'N/A'} "
                    f"for index '{self.definition.index_name}'.")

        if not all([asset_involved, value is not None, ex_date]):
            logger.warning(f"Insufficient information for corporate action: {action}. No divisor adjustment.")
            return current_divisor_before_ca

        if asset_involved not in constituents:
            logger.info(f"Asset {asset_involved.asset_id} affected by CA is not currently an index constituent. No divisor adjustment.")
            return current_divisor_before_ca

        change_in_market_value_due_to_ca = 0.0

        if action_type == "SPECIAL_DIVIDEND":
            from ..asset.equity import Equity
            if not isinstance(asset_involved, Equity): return current_divisor_before_ca

            shares = self.data.fetch_shares_outstanding(asset_involved.ticker, ex_date.strftime('%Y-%m-%d'))
            if shares is None or shares <= 0:
                logger.warning(f"CA Handle: No shares for {asset_involved.ticker}. Cannot adjust divisor for special dividend.")
                return current_divisor_before_ca

            special_dividend_amount_per_share = float(value)
            value_reduction_local_ccy = special_dividend_amount_per_share * shares

            if hasattr(self.definition.weighting_scheme, 'use_free_float') and \
               self.definition.weighting_scheme.use_free_float:
                ff_factor = self.data.fetch_free_float_factor(asset_involved.ticker, ex_date.strftime('%Y-%m-%d'))
                if ff_factor is not None: value_reduction_local_ccy *= ff_factor

            fx_rate = 1.0
            if asset_involved.currency.upper() != self.definition.currency.upper():
                fx_series = self.data.fetch_fx_rates(asset_involved.currency, self.definition.currency,
                                                     ex_date.strftime('%Y-%m-%d'), ex_date.strftime('%Y-%m-%d'))
                if not fx_series.empty: fx_rate = fx_series.iloc[0]
                else:
                    logger.warning(f"CA Handle: No FX for {asset_involved.currency}/{self.definition.currency}. Cannot adjust precisely.")
                    return current_divisor_before_ca

            change_in_market_value_due_to_ca = value_reduction_local_ccy * fx_rate
            logger.debug(f"Special Dividend: Asset {asset_involved.asset_id}, reduction value (index ccy): {change_in_market_value_due_to_ca:.2f}")

        elif action_type == "RIGHTS_ISSUE":
            logger.warning(f"Divisor adjustment for {action_type} is complex and placeholder.")

        elif action_type == "SPIN_OFF":
            logger.warning(f"Divisor adjustment for {action_type} is complex and placeholder.")

        if abs(change_in_market_value_due_to_ca) > 1e-9:
            if current_total_market_value_before_ca <= 0:
                 logger.warning(f"CA Handle: Market value before CA is {current_total_market_value_before_ca}. Cannot adjust divisor.")
                 return current_divisor_before_ca

            market_value_after_ca_effect = current_total_market_value_before_ca - change_in_market_value_due_to_ca

            if market_value_after_ca_effect < 0:
                logger.error(f"CA Handle: Calculated market value after CA effect is negative ({market_value_after_ca_effect}). This is unusual. Not adjusting divisor.")
                return current_divisor_before_ca

            new_divisor = current_divisor_before_ca * (market_value_after_ca_effect / current_total_market_value_before_ca)

            logger.info(f"Divisor adjusted due to {action_type} for {asset_involved.asset_id}. "
                        f"Old Divisor: {current_divisor_before_ca:.4f}, New Divisor: {new_divisor:.4f}. "
                        f"MV Before: {current_total_market_value_before_ca:.2f}, MV After Effect: {market_value_after_ca_effect:.2f}")
            return new_divisor
        else:
            logger.debug(f"No significant market value change from CA {action_type} for {asset_involved.asset_id}. Divisor not changed.")
            return current_divisor_before_ca

    def run(self,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> 'IndexResult':
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
            constituent snapshots and weight snapshots.

        Raises:
            ValueError: If *end_date* is not provided or precedes the base date.
        """
        from .result import IndexResult

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
        rebalance_dates = set(rebalance_dates_list)
        rebalance_dates.discard(base_date)

        # Accumulators
        index_levels: Dict[pd.Timestamp, float] = {}
        divisor_values: Dict[pd.Timestamp, float] = {}
        constituent_snapshots: Dict[pd.Timestamp, List[str]] = {}
        weight_snapshots: Dict[pd.Timestamp, Dict[str, float]] = {}

        # Running state
        constituents: List['Asset'] = []
        weights: Dict['Asset', float] = {}
        divisor: float = 0.0
        level: float = self.definition.base_value

        for date in trading_days:
            if date == base_date:
                # --- Base date initialisation ---
                constituents_raw = self._get_universe(date)
                constituents = self.select_constituents(constituents_raw, date)
                weights = self.calculate_constituent_weights(constituents, date)

                mv_map = self._get_constituent_market_values(weights, date)
                total_mv = sum(mv_map.values())
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
                # Capture pre-rebalance market value for divisor adjustment
                old_mv_map = self._get_constituent_market_values(weights, date)
                old_total_mv = sum(old_mv_map.values())

                # Reconstitute
                constituents_raw = self._get_universe(date)
                constituents = self.select_constituents(constituents_raw, date)
                weights = self.calculate_constituent_weights(constituents, date)

                # New market value under new composition
                new_mv_map = self._get_constituent_market_values(weights, date)
                new_total_mv = sum(new_mv_map.values())

                # Adjust divisor for continuity
                if old_total_mv > 0 and new_total_mv > 0:
                    divisor = self.adjust_divisor_for_rebalance(
                        divisor, old_total_mv, new_total_mv
                    )
                elif new_total_mv > 0:
                    divisor = new_total_mv / level if level > 0 else 1.0

                # Compute level with adjusted divisor
                level = new_total_mv / divisor if divisor > 0 else level

                # Record snapshots
                constituent_snapshots[date] = [a.asset_id for a in constituents]
                weight_snapshots[date] = {a.asset_id: w for a, w in weights.items()}

            else:
                # --- Regular trading day ---
                if not constituents or divisor <= 0:
                    # Before base date initialisation or no constituents
                    pass
                else:
                    level, divisor = self.calculate_index_level(
                        current_date=date,
                        constituents=constituents,
                        weights=weights,
                        divisor=divisor,
                        previous_index_level=level,
                    )

            index_levels[date] = level
            divisor_values[date] = divisor

        logger.info(
            f"run() completed for '{self.definition.index_name}': "
            f"{len(trading_days)} trading days, "
            f"{len(constituent_snapshots)} rebalance(s)."
        )

        return IndexResult(
            index_id=self.definition.index_id,
            index_levels=pd.Series(index_levels),
            divisor_history=pd.Series(divisor_values),
            constituent_snapshots=constituent_snapshots,
            weight_snapshots=weight_snapshots,
        ).with_data(self.data)

    def run_daily_calculation(self,
                              current_date: pd.Timestamp,
                              constituents: List['Asset'],
                              weights: Dict['Asset', float],
                              previous_index_level: float,
                              previous_divisor: float
                             ) -> Tuple[float, float]:
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
        logger.debug(f"--- Daily Calculation for {self.definition.index_name} on {current_date.strftime('%Y-%m-%d')} ---")

        divisor = previous_divisor

        if divisor is None or divisor <= 0:
            if current_date == self.definition.base_date:
                if not constituents:
                    raise ValueError("Base date calculation: Constituents not provided. Cannot initialize divisor.")
                base_day_values = self._get_constituent_market_values(
                    constituents_with_weights={asset: 0 for asset in constituents},
                    current_date=current_date
                )
                initial_mv = sum(base_day_values.values())
                if initial_mv > 0:
                    divisor = self.initialize_divisor(initial_mv)
                else:
                    raise ValueError(f"Cannot initialize divisor on base date {current_date} due to zero or negative market value.")
            else:
                raise ValueError("Divisor not initialized for index calculation.")

        new_level, final_divisor = self.calculate_index_level(
            current_date=current_date,
            constituents=constituents,
            weights=weights,
            divisor=divisor,
            previous_index_level=previous_index_level,
        )

        logger.debug(f"--- End Daily Calculation for {self.definition.index_name} on {current_date.strftime('%Y-%m-%d')} --- Level: {new_level:.4f}")
        return new_level, final_divisor
