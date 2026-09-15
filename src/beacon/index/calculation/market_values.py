# src/beacon/index/calculation/market_values.py
"""
Module for MarketValuesMixin, responsible for computing constituent
market values and the aggregate index level.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity, require_equity
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
        """An FX rate on a date. Kept as the calculator's name for one lookup.

        The caching, carry-forward and None-on-unknown behaviour this method
        defined moved to :meth:`DataFetcher.fx_rate_on` in BN-188, because
        three parts of the library were converting currency three different
        ways and one of them -- the market-cap weighting -- was not converting
        at all. This is now the calculator's spelling of that one lookup, so
        the levels, the weights and the reference display cannot drift apart
        again.

        Returns:
            float | None: The rate as of `date`, carried forward over gaps, or
            None when the pair is unknown -- which callers already treat as
            "cannot convert" rather than as a rate of one.
        """
        return self.data.fx_rate_on(from_currency, to_currency, date)

    def _get_constituent_market_values(self,
                                       constituents_with_weights: dict[Asset, float],
                                       current_date: pd.Timestamp) -> dict[Asset, float]:
        """
        Helper to get current market values for constituents.
        Market Value = Price * Shares * FX_Rate_to_Index_Currency * (FreeFloat if applicable)

        This method computes Sum(Price_t * Shares_t * [FF_t] * [FX_t])
        i.e. the "Adjusted Total Market Cap" of the index constituents.

        Raises:
            CalculationError: If any constituent is not an equity (BN-185).
                This used to skip it with a warning, which quietly aggregated
                the index over a subset of its own constituents.
        """
        constituent_market_values: dict[Asset, float] = {}

        for asset in constituents_with_weights:
            equity = require_equity(asset, "ConstituentMarketValues", "be valued")
            constituent_market_values[asset] = self._asset_market_value(equity, current_date)

        return constituent_market_values

    def _asset_market_value(self,
                            asset: Equity,
                            current_date: pd.Timestamp) -> float:
        """Compute the FX/free-float-adjusted market value for a single Equity asset.

        Returns 0.0 (with a warning logged) whenever price, shares, or FX data
        is missing, matching the previous inline behaviour.

        Only equities carry the ticker used for the market-data lookups;
        :meth:`_get_constituent_market_values` refuses a non-equity before it
        ever reaches here.

        Errors are not caught (BN-184). This used to sit under a bare
        ``except Exception`` that logged and returned 0.0, so a fetcher that
        *failed* and a name that is genuinely worth nothing were spelled the
        same — and any refusal raised beneath it, including the free-float one
        below, would have been absorbed here and never reached a caller.

        Raises:
            CalculationError: If the weighting scheme is float-adjusted and no
                usable free-float factor exists for *asset* (BN-184).
        """
        date_str = current_date.strftime('%Y-%m-%d')
        price_df = self.data.fetch_market_data(asset.ticker, date_str, date_str)

        # The two zero-value branches below, and the FX one further down, are
        # left as they are on purpose: they are the chain demonstrated with a
        # live wrong output in #204 (an exact-date read, so a market holiday
        # zeroes the whole book) and they are fixed there, together with
        # `_fx_rate` and `index_units`. Fixing one of them here would leave
        # the others producing the same wrong answer by a different route.
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
        market_value_local_ccy *= self._free_float(asset, date_str)

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

        return market_value_local_ccy * fx_rate

    def _free_float(self,
                    asset: Equity,
                    date_str: str) -> float:
        """The free-float factor to scale *asset*'s market cap by.

        1.0 when the weighting scheme is not float-adjusted, and otherwise the
        factor the data carries. A missing or out-of-range factor used to fall
        back to 1.0 with a warning — the full market cap, which is the twin of
        the fallback BN-179 removed from the weighting and still live on this
        path until BN-184. ``use_free_float`` is a statement about *what index
        this is*: an index weighted by full market cap where a float-adjusted
        one was specified is a different index, not a rounding error.

        Raises:
            CalculationError: If the scheme is float-adjusted and no usable
                factor exists.
        """
        scheme = self.definition.weighting_scheme

        if not getattr(scheme, "use_free_float", False):
            return 1.0

        factor = self.data.fetch_free_float_factor(asset.ticker, date_str)

        if factor is None or not 0.0 <= factor <= 1.0:
            raise CalculationError(
                calculation_name="ConstituentMarketValues",
                details=(f"'{scheme.scheme_name}' is float-adjusted but "
                         f"{asset.ticker} has no usable free-float factor on "
                         f"{date_str} (got {factor!r}). Using its full market "
                         f"cap would weight it as though every share were "
                         f"freely traded, which is a different index from the "
                         f"one specified."))

        return factor

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

        Raises:
            CalculationError: If *asset* is not an equity (BN-185). Valuing it
                at 0.0 made a constituent the index still holds contribute
                nothing, which is a silent restatement of the index rather
                than a missing price.

        Errors are not caught (BN-184). A bare ``except Exception`` used to
        turn any failure below into a unit value of 0.0, which feeds straight
        into the "holds zero units" path — so a fetcher that raised and a name
        that is genuinely unvaluable were indistinguishable, and any refusal
        added beneath this would have been absorbed before reaching a caller.
        """
        equity = require_equity(asset, "AssetUnitValue", "be valued")

        date_str = current_date.strftime('%Y-%m-%d')
        price_df = self.data.fetch_market_data(equity.ticker, date_str, date_str)

        if (price_df.empty or self.price_column not in price_df.columns
                or pd.isna(price_df[self.price_column].iloc[0])):
            logger.warning(f"asset_unit_value: No price for {equity.ticker}. Value is 0.")
            return 0.0

        price = float(price_df[self.price_column].iloc[0])

        return price * self._fx_rate(equity, current_date, date_str)

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

        # BN-184, triaged as leave. Unlike the rest of the catalogue this
        # substitutes nothing in practice: every term summed above is
        # price * shares * free_float * fx with each factor guarded
        # non-negative, so the total cannot be negative and this branch is
        # unreachable by construction. It is a floor against a future term
        # that could go negative (a short or a liability leg), and a floor is
        # the right answer there — a market-cap aggregate below zero has no
        # level to express. Kept, deliberately, rather than deleted as dead:
        # it costs one comparison a day and documents the invariant.
        if current_total_adjusted_market_value < 0:
            logger.warning(
                f"Total adjusted market value is negative: "
                f"{current_total_adjusted_market_value}. Using 0.")
            current_total_adjusted_market_value = 0.0

        new_index_level = current_total_adjusted_market_value / divisor

        return new_index_level, divisor
