# src/beacon/index/methodology.py
"""
Module defining base classes and examples for index methodology rules,
such as eligibility criteria and weighting schemes.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ..asset.base import Asset
from ..asset.equity import Equity
from ..data.fetcher import DataFetcher

logger = logging.getLogger(__name__)

# Market-data column names read by the rules/schemes below.
_PRICE_COLUMN = "CLOSE"
_VOLUME_COLUMN = "VOLUME"

class EligibilityRuleBase(ABC):
    """
    Abstract base class for an eligibility rule.
    Eligibility rules determine if an asset can be part of an index.
    """
    def __init__(self,
                 rule_name: str):
        self.rule_name = rule_name

    @abstractmethod
    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: dict[str, Any] | None = None) -> bool:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule_name='{self.rule_name}')"


# --- Example Eligibility Rules ---

class MarketCapRule(EligibilityRuleBase):
    """
    Eligibility rule based on market capitalization.
    """
    def __init__(self,
                 min_market_cap: float | None = None,
                 max_market_cap: float | None = None):
        super().__init__(rule_name="MarketCapRule")
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap

        if (min_market_cap is not None and max_market_cap is not None
                and min_market_cap > max_market_cap):
            raise ValueError("min_market_cap cannot be greater than max_market_cap.")

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: dict[str, Any] | None = None) -> bool:
        # Requires fetching market cap data for the asset on current_date
        # Market Cap = Price * Shares Outstanding
        # This logic is simplified. Real market cap data might be directly available or
        # need careful calculation.
        if not isinstance(asset, Equity):
            logger.debug(f"MarketCapRule: Asset {asset.asset_id} is not Equity type, skipping.")
            return True # Or False, depending on how non-equities should be handled by this rule

        date_str = current_date.strftime('%Y-%m-%d')
        try:
            price_df = market_data_provider.fetch_market_data(asset.ticker, date_str, date_str)
            if (price_df.empty or _PRICE_COLUMN not in price_df.columns
                    or pd.isna(price_df[_PRICE_COLUMN].iloc[0])):
                logger.warning(
                    f"MarketCapRule: Could not fetch price for {asset.ticker} "
                    f"on {date_str}.")
                return False
            current_price = price_df[_PRICE_COLUMN].iloc[0]

            shares_outstanding = market_data_provider.fetch_shares_outstanding(
                asset.ticker, date_str)
            if shares_outstanding is None or shares_outstanding <= 0:
                logger.warning(
                    f"MarketCapRule: Could not fetch valid shares outstanding for "
                    f"{asset.ticker} on {date_str}.")
                return False

            market_cap = current_price * shares_outstanding

            if self.min_market_cap is not None and market_cap < self.min_market_cap:
                logger.debug(
                    f"MarketCapRule: {asset.ticker} (MCap: {market_cap:.2f}) below "
                    f"min_market_cap {self.min_market_cap:.2f}")
                return False
            if self.max_market_cap is not None and market_cap > self.max_market_cap:
                logger.debug(
                    f"MarketCapRule: {asset.ticker} (MCap: {market_cap:.2f}) above "
                    f"max_market_cap {self.max_market_cap:.2f}")
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
    def __init__(self,
                 min_avg_daily_volume: int | None = None,
                 min_avg_daily_value: float | None = None,
                 lookback_days: int = 60):
        super().__init__(rule_name="LiquidityRule")
        self.min_avg_daily_volume = min_avg_daily_volume
        self.min_avg_daily_value = min_avg_daily_value
        self.lookback_days = lookback_days

        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: dict[str, Any] | None = None) -> bool:
        if not isinstance(asset, Equity):
            return True # Or False

        # Fetch more to ensure enough trading days
        start_lookback = (
            current_date - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d')
        end_lookback = current_date.strftime('%Y-%m-%d')

        try:
            price_df = market_data_provider.fetch_market_data(
                asset.ticker, start_lookback, end_lookback)
            if price_df.empty or price_df.shape[0] < (self.lookback_days / 2): # Ensure some data
                 logger.warning(
                     f"LiquidityRule: Insufficient historical price data for "
                     f"{asset.ticker} for period ending {end_lookback}.")
                 return False

            # Ensure we have data up to current_date or shortly before
            # (single-identifier market data is indexed by date).
            price_df = price_df[price_df.index <= current_date].tail(self.lookback_days)
            # Heuristic: need at least 80% of lookback days
            if price_df.shape[0] < (self.lookback_days * 0.8):
                logger.warning(
                    f"LiquidityRule: Not enough trading days "
                    f"({price_df.shape[0]}/{self.lookback_days}) for {asset.ticker} "
                    f"for ADV calc.")
                return False


            if self.min_avg_daily_volume is not None:
                if (_VOLUME_COLUMN not in price_df.columns
                        or price_df[_VOLUME_COLUMN].isnull().all()):
                    logger.warning(f"LiquidityRule: Volume data missing for {asset.ticker}.")
                    return False
                avg_daily_volume = price_df[_VOLUME_COLUMN].mean()
                if avg_daily_volume < self.min_avg_daily_volume:
                    logger.debug(
                        f"LiquidityRule: {asset.ticker} (ADV: {avg_daily_volume:.0f}) below "
                        f"min volume {self.min_avg_daily_volume:.0f}")
                    return False

            if self.min_avg_daily_value is not None:
                if (_PRICE_COLUMN not in price_df.columns
                        or _VOLUME_COLUMN not in price_df.columns
                        or price_df[_PRICE_COLUMN].isnull().all()
                        or price_df[_VOLUME_COLUMN].isnull().all()):
                    logger.warning(
                        f"LiquidityRule: Price or Volume data missing for ADTV "
                        f"calculation for {asset.ticker}.")
                    return False
                avg_daily_value = (price_df[_PRICE_COLUMN] * price_df[_VOLUME_COLUMN]).mean()
                if avg_daily_value < self.min_avg_daily_value:
                    logger.debug(
                        f"LiquidityRule: {asset.ticker} (ADTV: {avg_daily_value:.2f}) below "
                        f"min value {self.min_avg_daily_value:.2f}")
                    return False

            logger.debug(f"LiquidityRule: {asset.ticker} is eligible.")
            return True
        except Exception as e:
            logger.error(f"LiquidityRule: Error checking eligibility for {asset.ticker}: {e}")
            return False

# Other example stubs:
# class FreeFloatRule(EligibilityRuleBase): ...
# class ListingLocationRule(EligibilityRuleBase): ...


class WeightingSchemeBase(ABC):
    """
    Abstract base class for a weighting scheme.
    Weighting schemes determine the proportion of each constituent in an index.
    """
    def __init__(self,
                 scheme_name: str):
        self.scheme_name = scheme_name

    @abstractmethod
    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: dict[str, Any] | None = None) -> dict[Asset, float]:
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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scheme_name='{self.scheme_name}')"


# --- Example Weighting Schemes ---

class MarketCapWeighted(WeightingSchemeBase):
    """
    Market capitalization weighting scheme.
    Optionally supports free-float adjustment.
    """
    def __init__(self,
                 use_free_float: bool = False):
        super().__init__(scheme_name="MarketCapWeighted")
        self.use_free_float = use_free_float

    def _asset_market_cap(self,
                          asset: Equity,
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher) -> float:
        """
        Computes a single asset's market cap (price * shares outstanding),
        applying the free-float adjustment when enabled. Returns 0.0 and
        logs a warning when required price/shares data is missing or invalid.

        Only equities are priced this way; :meth:`calculate_weights` filters
        non-equity constituents out before calling this helper.
        """
        date_str = current_date.strftime('%Y-%m-%d')
        price_df = market_data_provider.fetch_market_data(asset.ticker, date_str, date_str)
        if (price_df.empty or _PRICE_COLUMN not in price_df.columns
                or pd.isna(price_df[_PRICE_COLUMN].iloc[0])):
            logger.warning(
                f"MarketCapWeighted: No price for {asset.ticker} on {date_str}. "
                "Market cap will be 0.")
            return 0.0
        current_price = float(price_df[_PRICE_COLUMN].iloc[0])

        shares_outstanding = market_data_provider.fetch_shares_outstanding(asset.ticker, date_str)
        if shares_outstanding is None or shares_outstanding <=0:
            logger.warning(
                f"MarketCapWeighted: No shares for {asset.ticker} on {date_str}. "
                "Market cap will be 0.")
            return 0.0

        asset_market_cap = current_price * shares_outstanding

        if not self.use_free_float:
            return asset_market_cap

        free_float_factor = market_data_provider.fetch_free_float_factor(asset.ticker, date_str)
        if free_float_factor is not None and 0.0 <= free_float_factor <= 1.0:
            return asset_market_cap * free_float_factor

        logger.warning(
            f"MarketCapWeighted: Invalid or missing free-float for {asset.ticker} "
            f"on {date_str}. Using full market cap.")
        return asset_market_cap

    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: dict[str, Any] | None = None) -> dict[Asset, float]:
        weights: dict[Asset, float] = {}
        market_caps: dict[Asset, float] = {}
        total_market_cap = 0.0

        for asset in constituents:
            if not isinstance(asset, Equity):
                logger.warning(
                    f"MarketCapWeighted: Asset {asset.asset_id} is not Equity. Skipping.")
                continue

            try:
                market_caps[asset] = self._asset_market_cap(
                    asset, current_date, market_data_provider)
                total_market_cap += market_caps[asset]
            except Exception as e:
                logger.error(
                    f"MarketCapWeighted: Error calculating market cap for {asset.ticker}: "
                    f"{e}. Market cap will be 0.")
                market_caps[asset] = 0.0

        if total_market_cap > 0:
            for asset, cap in market_caps.items():
                weights[asset] = cap / total_market_cap
            return weights

        # Handle case with no valid market caps (e.g. assign equal weight if any
        # assets, or empty if none)
        if not constituents:
            # else weights remains empty
            return weights

        logger.warning(
            "MarketCapWeighted: Total market cap is zero. Assigning equal weights as fallback.")
        equal_weight = 1.0 / len(constituents) if constituents else 0.0
        for asset in constituents:
             if isinstance(asset, Equity): # Only for those processed
                weights[asset] = equal_weight

        return weights


class EqualWeighted(WeightingSchemeBase):
    """
    Equal weighting scheme.
    """
    def __init__(self) -> None:
        super().__init__(scheme_name="EqualWeighted")

    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: dict[str, Any] | None = None) -> dict[Asset, float]:
        weights: dict[Asset, float] = {}
        num_constituents = len(constituents)

        if num_constituents > 0:
            weight_per_constituent = 1.0 / num_constituents
            for asset in constituents:
                weights[asset] = weight_per_constituent
        else:
            logger.warning("EqualWeighted: No constituents provided. Returning empty weights.")

        return weights

# class CorporateActionRule: ... (For specific handling if not covered by divisor)
