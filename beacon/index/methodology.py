# beacon/index/methodology.py
"""
Module defining base classes and examples for index methodology rules,
such as eligibility criteria and weighting schemes.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, TYPE_CHECKING, Optional
import logging

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..asset.base import Asset
    from ..asset.equity import Equity # For specific checks if needed
    from ..data.fetcher import DataFetcher


logger = logging.getLogger(__name__)

class MethodologyRule(ABC):
    """
    Base class for a generic methodology rule.
    Could be used to categorize different types of rules if needed,
    though specific base classes like EligibilityRuleBase are more direct.
    """
    def __init__(self, rule_type: str):
        self.rule_type = rule_type

    @abstractmethod
    def apply(self, *args: Any, **kwargs: Any) -> Any:
        """Generic apply method to be implemented by subclasses."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule_type='{self.rule_type}')"


class EligibilityRuleBase(MethodologyRule):
    """
    Abstract base class for an eligibility rule.
    Eligibility rules determine if an asset can be part of an index.
    """
    def __init__(self, rule_name: str):
        super().__init__(rule_type="ELIGIBILITY")
        self.rule_name = rule_name

    @abstractmethod
    def is_eligible(self,
                    asset: 'Asset',
                    current_date: pd.Timestamp,
                    market_data_provider: 'DataFetcher',
                    # Optional: context from index definition (e.g., universe constraints)
                    context: Optional[Dict[str, Any]] = None
                   ) -> bool:
        """
        Checks if a given asset is eligible based on this rule.

        Args:
            asset: The asset to check.
            current_date: The date on which eligibility is being assessed.
            market_data_provider: A DataFetcher instance to get necessary market data
                                  (e.g., market cap, trading volume).
            context: Optional dictionary for additional context from the index or global settings.

        Returns:
            True if the asset is eligible, False otherwise.
        """
        pass

    # Override apply to match specific use if needed, or remove if is_eligible is preferred interface
    def apply(self, asset: 'Asset', current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> bool:
        return self.is_eligible(asset, current_date, market_data_provider, context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule_name='{self.rule_name}')"


# --- Example Eligibility Rules ---

class MarketCapRule(EligibilityRuleBase):
    """
    Eligibility rule based on market capitalization.
    """
    def __init__(self, min_market_cap: Optional[float] = None, max_market_cap: Optional[float] = None):
        super().__init__(rule_name="MarketCapRule")
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        if min_market_cap is not None and max_market_cap is not None and min_market_cap > max_market_cap:
            raise ValueError("min_market_cap cannot be greater than max_market_cap.")

    def is_eligible(self, asset: 'Asset', current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> bool:
        # Requires fetching market cap data for the asset on current_date
        # Market Cap = Price * Shares Outstanding
        # This logic is simplified. Real market cap data might be directly available or need careful calculation.
        from ..asset.equity import Equity # Specific check for equity attributes
        if not isinstance(asset, Equity):
            logger.debug(f"MarketCapRule: Asset {asset.asset_id} is not Equity type, skipping.")
            return True # Or False, depending on how non-equities should be handled by this rule

        try:
            price_df = market_data_provider.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
            if price_df.empty or 'Adj Close' not in price_df.columns or pd.isna(price_df['Adj Close'].iloc[0]):
                logger.warning(f"MarketCapRule: Could not fetch price for {asset.ticker} on {current_date.strftime('%Y-%m-%d')}.")
                return False
            current_price = price_df['Adj Close'].iloc[0]

            shares_outstanding = market_data_provider.fetch_shares_outstanding(asset.ticker, current_date.strftime('%Y-%m-%d'))
            if shares_outstanding is None or shares_outstanding <= 0:
                logger.warning(f"MarketCapRule: Could not fetch valid shares outstanding for {asset.ticker} on {current_date.strftime('%Y-%m-%d')}.")
                return False

            market_cap = current_price * shares_outstanding

            if self.min_market_cap is not None and market_cap < self.min_market_cap:
                logger.debug(f"MarketCapRule: {asset.ticker} (MCap: {market_cap:.2f}) below min_market_cap {self.min_market_cap:.2f}")
                return False
            if self.max_market_cap is not None and market_cap > self.max_market_cap:
                logger.debug(f"MarketCapRule: {asset.ticker} (MCap: {market_cap:.2f}) above max_market_cap {self.max_market_cap:.2f}")
                return False
            logger.debug(f"MarketCapRule: {asset.ticker} (MCap: {market_cap:.2f}) is eligible.")
            return True
        except Exception as e:
            logger.error(f"MarketCapRule: Error checking eligibility for {asset.ticker}: {e}")
            return False


class LiquidityRule(EligibilityRuleBase):
    """
    Eligibility rule based on trading liquidity (e.g., average daily volume or value).
    """
    def __init__(self, min_avg_daily_volume: Optional[int] = None, min_avg_daily_value: Optional[float] = None, lookback_days: int = 60):
        super().__init__(rule_name="LiquidityRule")
        self.min_avg_daily_volume = min_avg_daily_volume
        self.min_avg_daily_value = min_avg_daily_value
        self.lookback_days = lookback_days
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")

    def is_eligible(self, asset: 'Asset', current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> bool:
        from ..asset.equity import Equity
        if not isinstance(asset, Equity):
            return True # Or False

        start_lookback = (current_date - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d') # Fetch more to ensure enough trading days
        end_lookback = current_date.strftime('%Y-%m-%d')
        
        try:
            price_df = market_data_provider.fetch_prices(asset.ticker, start_lookback, end_lookback)
            if price_df.empty or price_df.shape[0] < (self.lookback_days / 2): # Ensure some data
                 logger.warning(f"LiquidityRule: Insufficient historical price data for {asset.ticker} for period ending {end_lookback}.")
                 return False
            
            # Ensure we have data up to current_date or shortly before
            price_df = price_df[price_df['Date'] <= current_date].tail(self.lookback_days)
            if price_df.shape[0] < (self.lookback_days * 0.8): # Heuristic: need at least 80% of lookback days
                logger.warning(f"LiquidityRule: Not enough trading days ({price_df.shape[0]}/{self.lookback_days}) for {asset.ticker} for ADV calc.")
                return False


            if self.min_avg_daily_volume is not None:
                if 'Volume' not in price_df.columns or price_df['Volume'].isnull().all():
                    logger.warning(f"LiquidityRule: Volume data missing for {asset.ticker}.")
                    return False
                avg_daily_volume = price_df['Volume'].mean()
                if avg_daily_volume < self.min_avg_daily_volume:
                    logger.debug(f"LiquidityRule: {asset.ticker} (ADV: {avg_daily_volume:.0f}) below min volume {self.min_avg_daily_volume:.0f}")
                    return False

            if self.min_avg_daily_value is not None:
                if 'Adj Close' not in price_df.columns or 'Volume' not in price_df.columns or \
                   price_df['Adj Close'].isnull().all() or price_df['Volume'].isnull().all():
                    logger.warning(f"LiquidityRule: Price or Volume data missing for ADTV calculation for {asset.ticker}.")
                    return False
                avg_daily_value = (price_df['Adj Close'] * price_df['Volume']).mean()
                if avg_daily_value < self.min_avg_daily_value:
                    logger.debug(f"LiquidityRule: {asset.ticker} (ADTV: {avg_daily_value:.2f}) below min value {self.min_avg_daily_value:.2f}")
                    return False
            
            logger.debug(f"LiquidityRule: {asset.ticker} is eligible.")
            return True
        except Exception as e:
            logger.error(f"LiquidityRule: Error checking eligibility for {asset.ticker}: {e}")
            return False

# Other example stubs:
# class FreeFloatRule(EligibilityRuleBase): ...
# class ListingLocationRule(EligibilityRuleBase): ...


class WeightingSchemeBase(MethodologyRule):
    """
    Abstract base class for a weighting scheme.
    Weighting schemes determine the proportion of each constituent in an index.
    """
    def __init__(self, scheme_name: str):
        super().__init__(rule_type="WEIGHTING")
        self.scheme_name = scheme_name

    @abstractmethod
    def calculate_weights(self,
                          constituents: List['Asset'],
                          current_date: pd.Timestamp,
                          market_data_provider: 'DataFetcher',
                          # Optional context (e.g., from index definition)
                          context: Optional[Dict[str, Any]] = None
                         ) -> Dict['Asset', float]:
        """
        Calculates the weight for each constituent asset.

        Args:
            constituents: A list of assets that are eligible for the index.
            current_date: The date for which weights are being calculated.
            market_data_provider: A DataFetcher instance.
            context: Optional dictionary for additional context.

        Returns:
            A dictionary mapping each Asset object to its calculated weight (float).
            The sum of weights should typically be 1.0.
        """
        pass

    def apply(self, constituents: List['Asset'], current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> Dict['Asset', float]:
        return self.calculate_weights(constituents, current_date, market_data_provider, context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scheme_name='{self.scheme_name}')"


# --- Example Weighting Schemes ---

class MarketCapWeighted(WeightingSchemeBase):
    """
    Market capitalization weighting scheme.
    Optionally supports free-float adjustment.
    """
    def __init__(self, use_free_float: bool = False):
        super().__init__(scheme_name="MarketCapWeighted")
        self.use_free_float = use_free_float

    def calculate_weights(self, constituents: List['Asset'], current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> Dict['Asset', float]:
        weights: Dict['Asset', float] = {}
        market_caps: Dict['Asset', float] = {}
        total_market_cap = 0.0

        from ..asset.equity import Equity
        for asset in constituents:
            if not isinstance(asset, Equity):
                logger.warning(f"MarketCapWeighted: Asset {asset.asset_id} is not Equity. Skipping.")
                continue
            try:
                price_df = market_data_provider.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
                if price_df.empty or pd.isna(price_df['Adj Close'].iloc[0]):
                    logger.warning(f"MarketCapWeighted: No price for {asset.ticker} on {current_date.strftime('%Y-%m-%d')}. Market cap will be 0.")
                    market_caps[asset] = 0.0
                    continue
                current_price = price_df['Adj Close'].iloc[0]

                shares_outstanding = market_data_provider.fetch_shares_outstanding(asset.ticker, current_date.strftime('%Y-%m-%d'))
                if shares_outstanding is None or shares_outstanding <=0:
                    logger.warning(f"MarketCapWeighted: No shares for {asset.ticker} on {current_date.strftime('%Y-%m-%d')}. Market cap will be 0.")
                    market_caps[asset] = 0.0
                    continue
                
                asset_market_cap = current_price * shares_outstanding

                if self.use_free_float:
                    free_float_factor = market_data_provider.fetch_free_float_factor(asset.ticker, current_date.strftime('%Y-%m-%d'))
                    if free_float_factor is not None and 0.0 <= free_float_factor <= 1.0:
                        asset_market_cap *= free_float_factor
                    else:
                        logger.warning(f"MarketCapWeighted: Invalid or missing free-float for {asset.ticker} on {current_date.strftime('%Y-%m-%d')}. Using full market cap.")
                
                market_caps[asset] = asset_market_cap
                total_market_cap += asset_market_cap
            except Exception as e:
                logger.error(f"MarketCapWeighted: Error calculating market cap for {asset.ticker}: {e}. Market cap will be 0.")
                market_caps[asset] = 0.0


        if total_market_cap > 0:
            for asset, cap in market_caps.items():
                weights[asset] = cap / total_market_cap
        else:
            # Handle case with no valid market caps (e.g. assign equal weight if any assets, or empty if none)
            if constituents:
                logger.warning("MarketCapWeighted: Total market cap is zero. Assigning equal weights as fallback.")
                equal_weight = 1.0 / len(constituents) if constituents else 0.0
                for asset in constituents:
                     if isinstance(asset, Equity): # Only for those processed
                        weights[asset] = equal_weight
            # else weights remains empty

        return weights


class EqualWeighted(WeightingSchemeBase):
    """
    Equal weighting scheme.
    """
    def __init__(self):
        super().__init__(scheme_name="EqualWeighted")

    def calculate_weights(self, constituents: List['Asset'], current_date: pd.Timestamp, market_data_provider: 'DataFetcher', context: Optional[Dict[str, Any]] = None) -> Dict['Asset', float]:
        weights: Dict['Asset', float] = {}
        num_constituents = len(constituents)

        if num_constituents > 0:
            weight_per_constituent = 1.0 / num_constituents
            for asset in constituents:
                weights[asset] = weight_per_constituent
        else:
            logger.warning("EqualWeighted: No constituents provided. Returning empty weights.")
            
        return weights

# class CorporateActionRule(MethodologyRule): ... (For specific handling if not covered by divisor)