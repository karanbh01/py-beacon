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
