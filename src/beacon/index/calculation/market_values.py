# src/beacon/index/calculation/market_values.py
"""
Module for MarketValuesMixin, responsible for computing constituent
market values and the aggregate index level.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity
from ...data.fetcher import DataFetcher
from ...exceptions import CalculationError
from ..constructor import IndexDefinition

logger = logging.getLogger(__name__)

class MarketValuesMixin:
    """Market-value and index-level calculation logic, mixed into IndexCalculator."""

    # Provided by the IndexCalculator that mixes this in.
    data: DataFetcher
    definition: IndexDefinition
    price_column: str

    def _get_constituent_market_values(self,
                                       constituents_with_weights: dict[Asset, float],
                                       current_date: pd.Timestamp) -> dict[Asset, float]:
        """
        Helper to get current market values for constituents.
        Market Value = Price * Shares * FX_Rate_to_Index_Currency * (FreeFloat if applicable)

        This method computes Sum(Price_t * Shares_t * [FF_t] * [FX_t])
        i.e. the "Adjusted Total Market Cap" of the index constituents.
        """
        constituent_market_values: dict[Asset, float] = {}

        for asset in constituents_with_weights:
            if not isinstance(asset, Equity):
                logger.warning(
                    f"Asset {asset.asset_id} is not Equity. Skipping market value "
                    "calculation.")
                continue
            constituent_market_values[asset] = self._asset_market_value(asset, current_date)

        return constituent_market_values

    def _asset_market_value(self,
                            asset: Equity,
                            current_date: pd.Timestamp) -> float:
        """Compute the FX/free-float-adjusted market value for a single Equity asset.

        Returns 0.0 (with a warning/error logged) whenever price, shares, or
        FX data is missing or invalid, matching the previous inline behaviour.

        Only equities carry the ticker used for the market-data lookups;
        :meth:`_get_constituent_market_values` filters non-equities out first.
        """
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            price_df = self.data.fetch_market_data(asset.ticker, date_str, date_str)
            if price_df.empty or self.price_column not in price_df.columns \
                    or pd.isna(price_df[self.price_column].iloc[0]):
                logger.warning(
                    f"_get_constituent_market_values: No price for {asset.ticker}. "
                    "Value is 0.")
                return 0.0
            current_price = float(price_df[self.price_column].iloc[0])

            shares = self.data.fetch_shares_outstanding(asset.ticker, date_str)
            if shares is None or shares <= 0:
                logger.warning(
                    f"_get_constituent_market_values: No shares for {asset.ticker}. "
                    "Value is 0.")
                return 0.0

            market_value_local_ccy = current_price * shares

            # Apply Free Float if used by weighting scheme
            if hasattr(self.definition.weighting_scheme, 'use_free_float') and \
               self.definition.weighting_scheme.use_free_float:
                ff_factor = self.data.fetch_free_float_factor(asset.ticker, date_str)
                if ff_factor is not None and 0.0 <= ff_factor <= 1.0:
                    market_value_local_ccy *= ff_factor
                else:
                    logger.warning(
                        f"Missing or invalid free-float for {asset.ticker}, not "
                        "applying to market value.")

            # FX Conversion to Index Currency
            fx_rate = 1.0
            if asset.currency.upper() != self.definition.currency.upper():
                fx_series = self.data.fetch_fx_rates(asset.currency, self.definition.currency,
                                                     date_str, date_str)
                if not fx_series.empty:
                    fx_rate = float(fx_series.iloc[0])
                else:
                    logger.warning(
                        f"No FX rate found for {asset.currency}/{self.definition.currency} "
                        f"on {current_date}. Using 1.0.")
                    return 0.0

            adj_market_value_index_ccy = market_value_local_ccy * fx_rate
            return adj_market_value_index_ccy

        except Exception as e:
            logger.error(f"Error calculating market value for {asset.ticker}: {e}")
            return 0.0

    def calculate_index_level(self,
                              current_date: pd.Timestamp,
                              constituents: list[Asset],
                              weights: dict[Asset, float],
                              divisor: float,
                              previous_index_level: float) -> tuple[float, float]:
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
            logger.error(f"Invalid divisor: {divisor}. Cannot calculate index level.")
            raise CalculationError("IndexLevelCalculation", f"Invalid divisor: {divisor}")

        if not constituents:
            logger.warning(
                f"[{current_date.strftime('%Y-%m-%d')}] No current constituents to "
                f"calculate index level for '{self.definition.index_name}'. "
                "Returning previous level.")
            return previous_index_level, divisor

        constituent_values_map = self._get_constituent_market_values(
            constituents_with_weights=weights,
            current_date=current_date
        )
        current_total_adjusted_market_value = sum(constituent_values_map.values())

        if current_total_adjusted_market_value < 0:
            logger.warning(
                f"Total adjusted market value is negative: "
                f"{current_total_adjusted_market_value}. Using 0.")
            current_total_adjusted_market_value = 0.0

        new_index_level = current_total_adjusted_market_value / divisor

        return new_index_level, divisor
