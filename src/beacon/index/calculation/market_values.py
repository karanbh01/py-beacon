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


    def rate_on(self,
                from_currency: str,
                to_currency: str,
                date: pd.Timestamp) -> float | None:
        """An FX rate on a date, from a series fetched once per pair.

        The calculator converts every foreign holding on every day, so this
        used to make one `fetch_fx_rates` call per foreign name per date --
        each of which slices the whole market frame. Against the
        single-currency universes that existed before BN-128 it never fired;
        against a global one an eighty-name index took longer than the rest of
        the suite put together.

        Returns:
            float | None: The rate as of `date`, carried forward over gaps, or
            None when the pair is unknown -- which callers already treat as
            "cannot convert" rather than as a rate of one.
        """
        if from_currency.upper() == to_currency.upper():
            return 1.0

        pair = (from_currency.upper(), to_currency.upper())
        cache = getattr(self, "_fx_cache", None)

        if cache is None:
            cache = {}
            self._fx_cache = cache

        if pair not in cache:
            cache[pair] = self.data.fetch_fx_rates(
                from_currency, to_currency).sort_index()

        series = cache[pair]

        if series.empty:
            return None

        position = series.index.searchsorted(date, side="right") - 1

        return float(series.iloc[max(position, 0)])

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
                rate = self.rate_on(asset.currency,
                                    self.definition.currency, current_date)
                if rate is not None:
                    fx_rate = rate
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

    def asset_unit_value(self,
                         asset: Asset,
                         current_date: pd.Timestamp) -> float:
        """Value of one unit of *asset* in the index currency: price times FX.

        Distinct from :meth:`_asset_market_value`, which multiplies by shares
        outstanding and free float. Those belong to the *weighting* of the
        index; this is what one unit is worth, which is what the index's
        holdings are valued at day to day.

        Args:
            asset: The constituent. Must be an Equity.
            current_date: Valuation date.

        Returns:
            float: Price in index currency, or 0.0 when price or FX is
            missing — matching the behaviour of the market-value path.
        """
        if not isinstance(asset, Equity):
            logger.warning(f"Asset {asset.asset_id} is not Equity. Unit value is 0.")
            return 0.0

        try:
            date_str = current_date.strftime('%Y-%m-%d')
            price_df = self.data.fetch_market_data(asset.ticker, date_str, date_str)

            if (price_df.empty or self.price_column not in price_df.columns
                    or pd.isna(price_df[self.price_column].iloc[0])):
                logger.warning(f"asset_unit_value: No price for {asset.ticker}. Value is 0.")
                return 0.0

            price = float(price_df[self.price_column].iloc[0])

            return price * self._fx_rate(asset, current_date, date_str)

        except Exception as e:
            logger.error(f"Error calculating unit value for {asset.asset_id}: {e}")
            return 0.0

    def _fx_rate(self,
                 asset: Asset,
                 current_date: pd.Timestamp,
                 date_str: str) -> float:
        """FX rate converting *asset*'s currency into the index currency.

        Returns 0.0 rather than 1.0 when a needed rate is missing, so a
        constituent whose rate cannot be found drops out of the aggregate
        instead of being silently valued as though no conversion were needed.
        """
        if asset.currency.upper() == self.definition.currency.upper():
            return 1.0

        rate = self.rate_on(asset.currency, self.definition.currency,
                            current_date)

        if rate is None:
            logger.warning(
                f"No FX rate found for {asset.currency}/{self.definition.currency} "
                f"on {current_date}. Excluding from the aggregate.")
            return 0.0

        return rate

    def index_units(self,
                    weights: dict[Asset, float],
                    aggregate: float,
                    current_date: pd.Timestamp) -> dict[Asset, float]:
        """Units of each constituent the index holds to realise *weights*.

        The index holds a fixed number of units of each constituent between
        rebalances, which is what makes weights *drift* with relative
        performance rather than being silently reset every day.

        Units are set so that ``unit_value * units`` is ``weight`` of
        *aggregate*, which makes the weights exactly right on the rebalance
        date and lets them move from there.

        For a market-capitalisation weighting this reduces to shares
        outstanding — the weight is itself the share of aggregate market value
        — so that methodology produces exactly the levels it did before units
        existed.

        Args:
            weights: Target weight per constituent, summing to 1.
            aggregate: Total value the index represents on this date.
            current_date: Rebalance date.

        Returns:
            dict: Units per constituent. A constituent with no unit value gets
            zero units rather than an infinite position.
        """
        units: dict[Asset, float] = {}

        for asset, weight in weights.items():
            unit_value = self.asset_unit_value(asset, current_date)

            if unit_value <= 0.0:
                logger.warning(
                    f"No unit value for {asset.asset_id} on {current_date}; it "
                    "holds zero units and contributes nothing.")
                units[asset] = 0.0
                continue

            units[asset] = weight * aggregate / unit_value

        return units

    def holding_values(self,
                       units: dict[Asset, float],
                       current_date: pd.Timestamp) -> dict[Asset, float]:
        """What each holding is worth today: units times unit value.

        Split out of :meth:`aggregate_value` because the daily weights panel
        needs the parts as well as the total, and a part costs a market-data
        lookup — computing them twice would double the lookups a run makes,
        which is its dominant cost.

        Args:
            units: What the index holds, asset to unit count.
            current_date: Valuation date.

        Returns:
            dict: Value per holding in the index currency. A name with no
            price today is worth 0.0 and still appears, because it is still
            held.
        """
        return {asset: count * self.asset_unit_value(asset, current_date)
                for asset, count in units.items()}

    def aggregate_value(self,
                        units: dict[Asset, float],
                        current_date: pd.Timestamp) -> float:
        """Total value of the index's holdings: units times unit value."""
        return float(sum(self.holding_values(units, current_date).values()))

    def level_from_units(self,
                         units: dict[Asset, float],
                         divisor: float,
                         current_date: pd.Timestamp,
                         previous_index_level: float,
                         values: dict[Asset, float] | None = None) -> float:
        """Index level on an ordinary day: holdings value over the divisor.

        Args:
            units: What the index holds, fixed since the last rebalance.
            divisor: Current divisor.
            current_date: Valuation date.
            previous_index_level: Carried forward when the index cannot be
                valued today, so a missing price shows as a flat day rather
                than a collapse to zero.
            values: Holdings already valued for *current_date*, from
                :meth:`holding_values`. Passed by the run loop, which needs
                them anyway to record the day's weights; omitting it values
                the holdings here instead.

        Returns:
            float: The index level.

        Raises:
            CalculationError: If the divisor is not positive.
        """
        if divisor <= 0:
            logger.error(f"Invalid divisor: {divisor}. Cannot calculate index level.")
            raise CalculationError("IndexLevelCalculation", f"Invalid divisor: {divisor}")

        if not units:
            logger.warning(
                f"[{current_date.strftime('%Y-%m-%d')}] No holdings for "
                f"'{self.definition.index_name}'. Returning previous level.")
            return previous_index_level

        aggregate = (float(sum(values.values())) if values is not None
                     else self.aggregate_value(units, current_date))

        if aggregate <= 0.0:
            logger.warning(
                f"[{current_date.strftime('%Y-%m-%d')}] Holdings are worth "
                f"{aggregate}. Returning previous level.")
            return previous_index_level

        return aggregate / divisor

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
