# beacon/index/calculation.py
"""
Module for the IndexCalculationAgent, responsible for the logic of
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

class IndexCalculationAgent:
    """
    Manages the ongoing calculation of an index based on its definition.
    This includes selecting constituents, calculating weights, computing index levels,
    and handling corporate action adjustments.
    """
    def __init__(self,
                 index_definition: 'IndexDefinition',
                 data_provider: 'DataFetcher'):
        """
        Initializes the IndexCalculationAgent.

        Args:
            index_definition: The IndexDefinition object that specifies the index rules.
            data_provider: A DataFetcher instance to access market and asset data.
        """
        if not index_definition:
            raise ValueError("index_definition must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")
            
        self.index_definition: 'IndexDefinition' = index_definition
        self.data_provider: 'DataFetcher' = data_provider

        # State variables for ongoing calculation (would be managed by a backtest or live engine)
        self.current_constituents: List['Asset'] = []
        self.current_weights: Dict['Asset', float] = {}
        self.current_index_level: float = index_definition.base_value
        self.current_divisor: Optional[float] = None # Will be initialized

        logger.info(f"IndexCalculationAgent initialized for index '{self.index_definition.index_name}'.")
        # Initialize divisor if on base date or if starting fresh
        # self.current_divisor = self.initialize_divisor_on_base_date()


    def select_constituents(self, universe: List['Asset'], current_date: pd.Timestamp) -> List['Asset']:
        """
        Selects index constituents from a given universe based on eligibility rules.

        Args:
            universe: A list of potential Asset objects to consider for inclusion.
                      This universe should be broad enough (e.g., all stocks on an exchange).
            current_date: The date for which selection is being made.

        Returns:
            A list of Asset objects that are eligible for the index.
        """
        logger.info(f"[{current_date.strftime('%Y-%m-%d')}] Selecting constituents for '{self.index_definition.index_name}'. Universe size: {len(universe)}")
        eligible_constituents: List['Asset'] = []
        if not universe:
            logger.warning("Constituent selection called with an empty universe.")
            return []

        for asset in universe:
            is_eligible_for_asset = True
            for rule in self.index_definition.eligibility_rules:
                try:
                    if not rule.is_eligible(asset, current_date, self.data_provider):
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

        self.current_constituents = eligible_constituents
        logger.info(f"Selected {len(eligible_constituents)} constituents for '{self.index_definition.index_name}'.")
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
            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] Calculating weights for an empty list of constituents for '{self.index_definition.index_name}'.")
            self.current_weights = {}
            return {}

        logger.info(f"[{current_date.strftime('%Y-%m-%d')}] Calculating weights for {len(constituents)} constituents of '{self.index_definition.index_name}'.")
        
        try:
            weights = self.index_definition.weighting_scheme.calculate_weights(
                constituents, current_date, self.data_provider
            )
        except Exception as e:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error(f"Error applying weighting scheme {self.index_definition.weighting_scheme.scheme_name}: {e}")
            raise CalculationError(calculation_name=f"WeightingScheme-{self.index_definition.weighting_scheme.scheme_name}", details=str(e))


        # Normalize weights to sum to 1, if not already
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-9 and weight_sum != 0 : # Check for non-zero sum before normalizing
            logger.warning(f"Weights from scheme {self.index_definition.weighting_scheme.scheme_name} sum to {weight_sum}. Normalizing.")
            normalized_weights = {asset: w / weight_sum for asset, w in weights.items()}
            self.current_weights = normalized_weights
        elif weight_sum == 0 and weights: # Has assets but sum is zero (e.g. all market caps were zero)
             logger.error(f"Calculated weights sum to zero for {len(weights)} constituents. Cannot normalize.")
             self.current_weights = weights # Keep zero weights
        else:
            self.current_weights = weights
            
        logger.info(f"Weights calculated for '{self.index_definition.index_name}'.")
        return self.current_weights

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
        if self.index_definition.base_value <= 0:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error("Base index value must be positive to initialize divisor.")
            raise CalculationError("DivisorInitialization", "Base index value is non-positive.")

        divisor = initial_total_market_value / self.index_definition.base_value
        self.current_divisor = divisor
        logger.info(f"Divisor for '{self.index_definition.index_name}' initialized to: {divisor:.4f} "
                    f"(Initial Market Value: {initial_total_market_value:.2f}, Base Value: {self.index_definition.base_value})")
        return divisor

    def _get_constituent_market_values(self,
                                       constituents_with_weights: Dict['Asset', float],
                                       current_date: pd.Timestamp) -> Dict['Asset', float]:
        """
        Helper to get current market values for constituents.
        Market Value = Price * Shares * FX_Rate_to_Index_Currency * (FreeFloat if applicable)
        This calculation needs to be robust. For this example, we assume weights already reflect
        the "value" part that the divisor works against (e.g. for market cap weighted, this is market cap).
        The `calculate_index_level` expects `current_constituents_market_values` which is the numerator
        for the Laspeyres formula (Sum of P_t * Q_base).

        This method should calculate Sum(CurrentPrice_i * BaseQuantity_i * FX_i)
        where BaseQuantity_i is the fixed quantity from the last rebalance.
        For a divisor-based index, what's often used is:
        Index_t = Sum(Price_it * Shares_it * FF_it * FX_it) / Divisor_t
        So, this method should compute the numerator part.
        """
        from ..asset.equity import Equity # Specific check for equity attributes

        total_adj_market_value_numerator = 0.0
        constituent_market_values: Dict['Asset', float] = {} # This will store the P_t * Q_0 term effectively

        # This is tricky because Q_0 (base quantities) are implicit.
        # A common way: use current prices and shares, and the divisor handles continuity.
        # So, we calculate Sum (Price_t * Shares_t * [FF_t] * [FX_t])
        # This sum is the "Adjusted Total Market Cap" of the index constituents.

        for asset, weight in constituents_with_weights.items(): # Weight not directly used here, but asset is
            if not isinstance(asset, Equity): # Assuming Equities for now
                logger.warning(f"Asset {asset.asset_id} is not Equity. Skipping market value calculation.")
                continue
            try:
                price_df = self.data_provider.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                if price_df.empty or pd.isna(price_df['Adj Close'].iloc[0]):
                    logger.warning(f"_get_constituent_market_values: No price for {asset.ticker}. Value is 0.")
                    constituent_market_values[asset] = 0.0
                    continue
                current_price = price_df['Adj Close'].iloc[0]

                shares = self.data_provider.fetch_shares_outstanding(asset.ticker, current_date.strftime('%Y-%m-%d'))
                if shares is None or shares <= 0:
                    logger.warning(f"_get_constituent_market_values: No shares for {asset.ticker}. Value is 0.")
                    constituent_market_values[asset] = 0.0
                    continue
                
                market_value_local_ccy = current_price * shares

                # Apply Free Float if used by weighting scheme (consistency is key)
                # This check should ideally be tied to the index_definition's properties
                if hasattr(self.index_definition.weighting_scheme, 'use_free_float') and \
                   self.index_definition.weighting_scheme.use_free_float:
                    ff_factor = self.data_provider.fetch_free_float_factor(asset.ticker, current_date.strftime('%Y-%m-%d'))
                    if ff_factor is not None and 0.0 <= ff_factor <= 1.0:
                        market_value_local_ccy *= ff_factor
                    else:
                        logger.warning(f"Missing or invalid free-float for {asset.ticker}, not applying to market value.")
                
                # FX Conversion to Index Currency
                fx_rate = 1.0
                if asset.currency.upper() != self.index_definition.currency.upper():
                    fx_series = self.data_provider.fetch_fx_rates(asset.currency, self.index_definition.currency,
                                                                  current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                    if not fx_series.empty:
                        fx_rate = fx_series.iloc[0]
                    else:
                        logger.warning(f"No FX rate found for {asset.currency}/{self.index_definition.currency} on {current_date}. Using 1.0.")
                        # This could be a critical failure depending on policy.
                        constituent_market_values[asset] = 0.0 # Or handle error more strictly
                        continue
                
                adj_market_value_index_ccy = market_value_local_ccy * fx_rate
                constituent_market_values[asset] = adj_market_value_index_ccy
                
            except Exception as e:
                logger.error(f"Error calculating market value for {asset.ticker}: {e}")
                constituent_market_values[asset] = 0.0 # Assign 0 if error
        
        return constituent_market_values


    def calculate_index_level(self,
                              current_date: pd.Timestamp,
                              # current_constituents_market_values: Dict['Asset', float], -> This should be calculated inside
                              previous_index_level: float, # Or self.current_index_level
                              current_divisor: float # Or self.current_divisor
                             ) -> Tuple[float, float]:
        """
        Calculates the current index level using a Laspeyres-type formula:
        Index Level = Sum of Current Market Values of Constituents / Current Divisor.
        The 'current_constituents_market_values' should be the sum of (Price_t * Shares_t * FF_t * FX_t)
        for all constituents, correctly adjusted for index currency.

        Args:
            current_date: The date for which to calculate the index level.
            previous_index_level: The index level from the previous calculation period.
                                  (Note: For pure Laspeyres with divisor, this isn't directly used in level formula
                                   but is state for continuity).
            current_divisor: The current index divisor.

        Returns:
            A tuple containing:
                - The newly calculated index level (float).
                - The divisor used for this calculation (float) - typically unchanged unless a CA occurred.
        """
        if self.current_divisor is None:
            # This should happen ideally only on the first day IF not base day.
            # If it's base day, divisor should be set by initialize_divisor.
            # If it's after base day and still None, it's an issue.
            # For robustness, try to initialize if base date.
            if current_date == self.index_definition.base_date:
                logger.info(f"Attempting to initialize divisor on base date: {current_date.strftime('%Y-%m-%d')}")
                # This requires constituents and their market values on base_date
                base_constituents = self.select_constituents(
                    universe=[], # This is problematic, need base universe. Assume current_constituents are base.
                    current_date=current_date
                ) # This needs to be the *base day* constituents
                
                # This whole block is tricky. Base day initialization is a special first step.
                # Let's assume `initialize_divisor` was called by an outer loop (e.g. backtester start).
                # If we reach here and divisor is None, it's a state error.
                from ..beacon_exceptions import CalculationError # Local import
                logger.error("Divisor is not initialized. Cannot calculate index level.")
                raise CalculationError("IndexLevelCalculation", "Divisor not initialized.")

        if current_divisor <= 0:
            from ..beacon_exceptions import CalculationError # Local import
            logger.error(f"Invalid divisor: {current_divisor}. Cannot calculate index level.")
            raise CalculationError("IndexLevelCalculation", f"Invalid divisor: {current_divisor}")

        # Get the sum of market values (numerator for Laspeyres)
        # The constituents here should be the ones currently in the index from the last rebalance.
        if not self.current_constituents: # current_constituents should be set by select_constituents
            logger.warning(f"[{current_date.strftime('%Y-%m-%d')}] No current constituents to calculate index level for '{self.index_definition.index_name}'. Returning previous level.")
            return previous_index_level, current_divisor # Or 0, or handle as error

        # The market values are Sum(Price_t * Shares_t * FF_t * FX_t)
        constituent_values_map = self._get_constituent_market_values(
            constituents_with_weights=self.current_weights, # Pass current_weights to identify constituents
            current_date=current_date
        )
        current_total_adjusted_market_value = sum(constituent_values_map.values())

        if current_total_adjusted_market_value < 0: # Can be 0 if all prices are 0
            logger.warning(f"Total adjusted market value is negative: {current_total_adjusted_market_value}. Using 0.")
            current_total_adjusted_market_value = 0.0

        new_index_level = current_total_adjusted_market_value / current_divisor
        self.current_index_level = new_index_level # Update state

        logger.debug(f"[{current_date.strftime('%Y-%m-%d')}] Index '{self.index_definition.index_name}': "
                     f"Total Market Value = {current_total_adjusted_market_value:.2f}, Divisor = {current_divisor:.4f}, "
                     f"New Level = {new_index_level:.4f}")
        return new_index_level, current_divisor


    def handle_corporate_action(self,
                                action: Dict[str, Any], # e.g. {'type': 'SPECIAL_DIVIDEND', 'asset': Asset, 'value': X, 'ex_date': YYYY-MM-DD}
                                current_total_market_value_before_ca: float, # Sum(P_old * Q)
                                current_divisor_before_ca: float
                               ) -> float:
        """
        Adjusts the index divisor for specific corporate actions to maintain index continuity.
        Standard price adjustments for splits/regular dividends are typically handled by 'Adjusted Close' prices
        from the data provider. This method is for actions that require explicit divisor adjustment,
        like special cash dividends, rights issues, spin-offs, etc.

        Divisor_new = Divisor_old * (MarketValue_after_CA / MarketValue_before_CA_at_ex_date_prices)

        Args:
            action: A dictionary describing the corporate action. Must include 'type', 'asset',
                    'value', and 'ex_date'. For 'SPECIAL_DIVIDEND', value is per share.
                    For 'RIGHTS_ISSUE', value might be discount or theoretical ex-rights price drop.
            current_total_market_value_before_ca: The sum of market values of all constituents
                                                  (Price * Shares * FF * FX) just before the CA's impact
                                                  (typically using close prices on day before ex-date).
            current_divisor_before_ca: The divisor in effect before this corporate action.

        Returns:
            The new adjusted divisor (float).
        """
        action_type = action.get('type', '').upper()
        asset_involved = action.get('asset')
        value = action.get('value') # Meaning depends on action_type
        ex_date = pd.Timestamp(action.get('ex_date'))

        logger.info(f"[{ex_date.strftime('%Y-%m-%d')}] Handling CA: {action_type} for asset {asset_involved.asset_id if asset_involved else 'N/A'} "
                    f"for index '{self.index_definition.index_name}'.")

        if not all([asset_involved, value is not None, ex_date]):
            logger.warning(f"Insufficient information for corporate action: {action}. No divisor adjustment.")
            return current_divisor_before_ca

        if asset_involved not in self.current_constituents:
            logger.info(f"Asset {asset_involved.asset_id} affected by CA is not currently an index constituent. No divisor adjustment.")
            return current_divisor_before_ca
        
        # This needs careful implementation per action type based on index provider rules.
        # General principle: change in index market value due to CA not reflecting market movement
        # must be offset by a divisor change.

        change_in_market_value_due_to_ca = 0.0

        if action_type == "SPECIAL_DIVIDEND":
            # For a special cash dividend, the stock price drops by the dividend amount on ex-date.
            # This drop in component's market value needs to be offset in the divisor.
            # Change = Dividend_per_share * Shares_outstanding * FreeFloat * FX_Rate
            
            # Need shares, FF, FX for the specific asset
            from ..asset.equity import Equity
            if not isinstance(asset_involved, Equity): return current_divisor_before_ca # Skip if not equity

            shares = self.data_provider.fetch_shares_outstanding(asset_involved.ticker, ex_date.strftime('%Y-%m-%d'))
            if shares is None or shares <= 0:
                logger.warning(f"CA Handle: No shares for {asset_involved.ticker}. Cannot adjust divisor for special dividend.")
                return current_divisor_before_ca
            
            special_dividend_amount_per_share = float(value)
            value_reduction_local_ccy = special_dividend_amount_per_share * shares

            # Apply Free Float if applicable for the index
            if hasattr(self.index_definition.weighting_scheme, 'use_free_float') and \
               self.index_definition.weighting_scheme.use_free_float:
                ff_factor = self.data_provider.fetch_free_float_factor(asset_involved.ticker, ex_date.strftime('%Y-%m-%d'))
                if ff_factor is not None: value_reduction_local_ccy *= ff_factor
            
            # Convert to index currency
            fx_rate = 1.0
            if asset_involved.currency.upper() != self.index_definition.currency.upper():
                fx_series = self.data_provider.fetch_fx_rates(asset_involved.currency, self.index_definition.currency,
                                                              ex_date.strftime('%Y-%m-%d'), ex_date.strftime('%Y-%m-%d'))
                if not fx_series.empty: fx_rate = fx_series.iloc[0]
                else:
                    logger.warning(f"CA Handle: No FX for {asset_involved.currency}/{self.index_definition.currency}. Cannot adjust precisely.")
                    return current_divisor_before_ca # Cannot make accurate adjustment

            change_in_market_value_due_to_ca = value_reduction_local_ccy * fx_rate
            logger.debug(f"Special Dividend: Asset {asset_involved.asset_id}, reduction value (index ccy): {change_in_market_value_due_to_ca:.2f}")

        elif action_type == "RIGHTS_ISSUE":
            # This is more complex. The value drop depends on rights terms.
            # Often, index providers use the drop in theoretical ex-rights price (TERP).
            # 'value' might represent this drop per share, or parameters to calculate it.
            # Simplified: assume 'value' is the per-share market cap reduction in asset's currency.
            # This requires specific methodology.
            logger.warning(f"Divisor adjustment for {action_type} is complex and placeholder.")
            # Similar calculation to special dividend if 'value' is direct market cap change per share
            pass
        
        elif action_type == "SPIN_OFF":
            # Spin-off: market cap of parent decreases. Sometimes spun-off entity added to index temporarily.
            # Requires specific rules. Change in market value is market cap of spun-off part.
            logger.warning(f"Divisor adjustment for {action_type} is complex and placeholder.")
            pass
            
        # Add other CA types: stock dividend, mergers, etc.

        if abs(change_in_market_value_due_to_ca) > 1e-9: # If there's a meaningful change
            if current_total_market_value_before_ca <= 0: # Avoid division by zero if market value is already zero
                 logger.warning(f"CA Handle: Market value before CA is {current_total_market_value_before_ca}. Cannot adjust divisor.")
                 return current_divisor_before_ca

            # Market value after CA's non-market impact = MarketValue_before - Change_due_to_CA
            market_value_after_ca_effect = current_total_market_value_before_ca - change_in_market_value_due_to_ca
            
            if market_value_after_ca_effect < 0: # Should not happen with typical CAs like special divs
                logger.error(f"CA Handle: Calculated market value after CA effect is negative ({market_value_after_ca_effect}). This is unusual. Not adjusting divisor.")
                return current_divisor_before_ca

            new_divisor = current_divisor_before_ca * (market_value_after_ca_effect / current_total_market_value_before_ca)
            
            logger.info(f"Divisor adjusted due to {action_type} for {asset_involved.asset_id}. "
                        f"Old Divisor: {current_divisor_before_ca:.4f}, New Divisor: {new_divisor:.4f}. "
                        f"MV Before: {current_total_market_value_before_ca:.2f}, MV After Effect: {market_value_after_ca_effect:.2f}")
            self.current_divisor = new_divisor # Update agent's state
            return new_divisor
        else:
            logger.debug(f"No significant market value change from CA {action_type} for {asset_involved.asset_id}. Divisor not changed.")
            return current_divisor_before_ca

    def run_daily_calculation(self, current_date: pd.Timestamp, previous_index_level: float, previous_divisor: float) -> Tuple[float, float, List['Asset'], Dict['Asset', float]]:
        """
        Runs a single day's index calculation process.
        This is a high-level method that would be called by a backtester or live engine.
        It assumes rebalancing/reconstitution decisions are handled externally by checking
        the index_definition.rebalancing_frequency.

        This method focuses on calculating the index level for current_date given
        a set of constituents and a divisor from 'previous_day_or_rebalance'.

        Args:
            current_date: The date for which to perform calculations.
            previous_index_level: Index level from the previous period.
            previous_divisor: Divisor from the previous period.

        Returns:
            Tuple: (new_index_level, new_divisor, current_constituents, current_weights)
                   The constituents and weights are returned for context/logging but might be unchanged
                   if it's not a rebalance day. The new_divisor might change if CAs occurred.
        """
        logger.debug(f"--- Daily Calculation for {self.index_definition.index_name} on {current_date.strftime('%Y-%m-%d')} ---")

        # 0. Set initial state for the day from previous day's end state
        self.current_index_level = previous_index_level
        self.current_divisor = previous_divisor
        # self.current_constituents and self.current_weights should be those from end of previous day / last rebalance.
        # These would typically be passed in or managed as persistent state by the calling engine.

        # 1. Check for and handle corporate actions affecting the divisor *before* price-based calculation.
        #    This requires fetching CAs for current_constituents with ex_date = current_date.
        #    The `current_total_market_value_before_ca` would be based on *previous day's close prices*
        #    for the constituents. This is a simplification point.
        #    Accurate CA handling needs precise timing (e.g., apply before market open on ex-date).

        #    Simplified: Sum of market values using previous day's close for current constituents.
        #    This calculation needs the prices from T-1.
        #    Let's assume for this daily run, the divisor passed (previous_divisor) is already adjusted for T-1 CAs.
        #    Any CA on *today's* ex-date needs to be handled with *today's opening prices* or a model.
        #    This is complex. For now, we'll assume CAs are handled by an outer loop or by adjusted prices.
        #    The `handle_corporate_action` method is there, but its integration into a daily loop needs care.
        
        #    A common model:
        #    Divisor_t_morning = Divisor_t-1_close
        #    For each CA on ex-date_t:
        #       Divisor_t_morning = handle_corporate_action(CA, MV_t-1_close_constits, Divisor_t_morning)
        #    IndexLevel_t = Sum(Price_t_close * Shares_t) / Divisor_t_morning
        #    Divisor_t_close = Divisor_t_morning (unless rebalance changes portfolio structure)

        # For now, let the divisor passed be the one to use, assuming CA adjustments are done elsewhere or reflected in adjusted prices.
        # If explicit CA handling for divisor is done here:
        # Fetch CAs for self.current_constituents for current_date
        # For each CA: self.current_divisor = self.handle_corporate_action(...)

        # 2. Calculate index level with current day's prices and the (potentially CA-adjusted) divisor.
        if self.current_divisor is None:
            # Attempt to initialize if it's the base date (first run scenario)
            if current_date == self.index_definition.base_date:
                if not self.current_constituents: # Should have been set by select_constituents if it's base date
                     logger.error("Base date calculation: Constituents not set. Cannot initialize divisor.")
                     # This path indicates an issue in the calling logic / setup.
                     raise Exception("Constituent setup error on base date for divisor initialization.")

                base_day_constituent_values = self._get_constituent_market_values(
                    constituents_with_weights= {asset: 0 for asset in self.current_constituents}, # Weights not needed for sum of MV
                    current_date=current_date
                )
                initial_mv = sum(base_day_constituent_values.values())
                if initial_mv > 0 :
                    self.current_divisor = self.initialize_divisor(initial_mv)
                else:
                    logger.error(f"Base date MV is {initial_mv}. Cannot initialize divisor robustly for {self.index_definition.index_name}.")
                    # Fallback, though problematic:
                    # self.current_divisor = 1.0
                    # self.current_index_level = self.index_definition.base_value
                    # return self.current_index_level, self.current_divisor, self.current_constituents, self.current_weights
                    raise ValueError(f"Cannot initialize divisor on base date {current_date} due to zero or negative market value.")
            else: # Not base date and divisor is None - error state
                logger.error(f"Divisor is None on non-base date {current_date} for {self.index_definition.index_name}. Cannot calculate index.")
                raise ValueError("Divisor not initialized for index calculation.")


        new_level, final_divisor_for_day = self.calculate_index_level(
            current_date=current_date,
            previous_index_level=self.current_index_level, # Current state before this calc
            current_divisor=self.current_divisor # Current state before this calc
        )
        self.current_index_level = new_level
        self.current_divisor = final_divisor_for_day # Divisor might change if CA was handled inside calculate_index_level

        logger.debug(f"--- End Daily Calculation for {self.index_definition.index_name} on {current_date.strftime('%Y-%m-%d')} --- Level: {self.current_index_level:.4f}")
        return self.current_index_level, self.current_divisor, self.current_constituents, self.current_weights