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
from ..catalogue import (
    SELECTION,
    WEIGHTING,
    Display,
    register,
)
from ..data.fetcher import DataFetcher
from ..exceptions import CalculationError

logger = logging.getLogger(__name__)

# Market-data column names read by the rules/schemes below.
_PRICE_COLUMN = "CLOSE"
_VOLUME_COLUMN = "VOLUME"


def _has_close(frame: pd.DataFrame) -> bool:
    """Whether *frame*'s first row carries a usable close."""
    return not (frame.empty or _PRICE_COLUMN not in frame.columns
                or pd.isna(frame[_PRICE_COLUMN].iloc[0]))


def _resolve_session(calculation_name: str,
                     action: str,
                     current_date: pd.Timestamp,
                     market_data_provider: DataFetcher) -> pd.Timestamp:
    """The session *current_date* reads from, or a refusal (BN-179, BN-182).

    One primitive for every part of a methodology that has to turn a requested
    date into the day the market was actually open. Selection and weighting
    calling the same function is the point: an index whose rules resolve one
    way and whose weights resolve another is two methodologies sharing a
    heading, which is the fault this was built to end.

    Args:
        calculation_name: What is refusing — a rule or scheme name.
        action: The verb phrase for the refusal message ("weight",
            "assess eligibility"), so the message says what could not be done.
        current_date: The date asked about.
        market_data_provider: The data source to resolve against.

    Returns:
        pd.Timestamp: The last session on or before *current_date*.

    Raises:
        CalculationError: If *current_date* falls outside the data's coverage.
            The message names both the date that was asked for and the data's
            actual end, because "ask for an earlier date or refresh the store"
            is the action and neither half of it is discoverable otherwise.
    """
    session = market_data_provider.resolve_session(current_date)

    if session is not None:
        return session

    requested = current_date.strftime('%Y-%m-%d')
    first, last = market_data_provider.date_range

    raise CalculationError(
        calculation_name=calculation_name,
        details=(f"cannot {action} at {requested}: the market data runs "
                 f"{first:%Y-%m-%d} to {last:%Y-%m-%d}, so that date lies "
                 f"outside it. Inside the range a date with no bar is a "
                 f"closed market and resolves back to the last session on "
                 f"or before it; outside it nothing is known, and carrying "
                 f"a price forward would answer a different question in "
                 f"this one's date. Ask for a date on or before "
                 f"{last:%Y-%m-%d}, or refresh the store."))


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

@register(SELECTION, "Market capitalisation",
          fields={
              "min_market_cap": Display("Minimum market cap", order=1,
                                        help="In the index currency. Blank for no floor."),
              "max_market_cap": Display("Maximum market cap", order=2,
                                        help="In the index currency. Blank for no ceiling."),
          })
class MarketCapRule(EligibilityRuleBase):
    """Eligibility by market capitalisation, read from a resolved session.

    **Dates resolve backwards into the data (BN-182).** A weekend, a holiday
    or any date inside the data's coverage that carries no bar is read at the
    last session on or before it, because that is the universe the index
    actually held through the closure. Reading the exact date instead
    excluded *every* name on a closed day: the universe emptied, the weighting
    was handed nothing, and an index of no constituents computed a coherent
    level of zero.

    That is the same resolution :class:`MarketCapWeighted` performs, through
    the same primitive and by design. Selection running on one calendar and
    weighting on another is two methodologies under one heading.

    **Past the last bar it refuses rather than excluding.** A rule that cannot
    evaluate has not found the asset ineligible, it has failed, and the two
    must not share an answer — "not in the index" is a published fact about a
    name, while "the data does not reach that date" is a fact about the store.
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
        """Whether *asset*'s market cap at the resolved session clears the bounds.

        Raises:
            CalculationError: If *current_date* lies outside the data's
                coverage, so the rule cannot be evaluated at all. Nothing here
                turns a failure into an exclusion — see the class docstring.
        """
        if not isinstance(asset, Equity):
            logger.debug(f"MarketCapRule: Asset {asset.asset_id} is not Equity type, skipping.")
            return True # Or False, depending on how non-equities should be handled by this rule

        session = _resolve_session(self.rule_name, "assess eligibility",
                                   current_date, market_data_provider)
        date_str = session.strftime('%Y-%m-%d')

        price_df = market_data_provider.fetch_market_data(asset.ticker, date_str, date_str)

        if not _has_close(price_df):
            logger.warning(
                f"MarketCapRule: Could not fetch price for {asset.ticker} "
                f"on {date_str}.")
            return False

        current_price = float(price_df[_PRICE_COLUMN].iloc[0])

        # Read on the same session as the price, so the cap is one coherent
        # observation rather than a current share count against an older close.
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


@register(SELECTION, "Liquidity",
          fields={
              "min_avg_daily_volume": Display("Minimum average daily volume", order=1,
                                              help="Shares traded per day, averaged "
                                                   "over the lookback."),
              "min_avg_daily_value": Display("Minimum average daily value", order=2,
                                             help="Traded value per day, in the "
                                                  "index currency."),
              "lookback_days": Display("Lookback", order=3,
                                       help="Trading days the averages are taken over."),
          })
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
        """Whether *asset*'s traded volume and value over the lookback qualify.

        No session resolution here, and none needed: this reads a *window*
        ending at *current_date*, so a closed day is already spanned by the
        days around it rather than being the single day everything hangs on.

        Errors are not caught (BN-182). A rule that throws has not said the
        asset is ineligible, and the two answers must not be spelled the same.
        """
        if not isinstance(asset, Equity):
            return True # Or False

        # Fetch more to ensure enough trading days
        start_lookback = (
            current_date - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d')
        end_lookback = current_date.strftime('%Y-%m-%d')

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

@register(WEIGHTING, "Market capitalisation weighted",
          fields={
              "use_free_float": Display("Free-float adjusted", order=1,
                                        help="Weight by the freely traded portion "
                                             "rather than by full market cap."),
          })
class MarketCapWeighted(WeightingSchemeBase):
    """Market capitalization weighting, optionally free-float adjusted.

    **Every path either weights by real market caps or refuses (BN-179).**
    There is no equal-weight fallback: an index that comes out equal-weighted
    because the caps could not be read is not a degraded market-cap index, it
    is a different index published under the same heading, and nothing
    downstream looks wrong enough for anyone to ask — the levels are right,
    the weights sum, the backtest tracks.

    **Dates resolve backwards into the data.** A request for a weekend, a
    holiday, or any date inside the data's coverage that carries no bar reads
    the last session on or before it, because that is the composition the
    index actually held through the closure rather than an approximation of
    one. Past the last bar it refuses, since there the same read would be a
    stale print presented as the current one. The bound is the data's own
    coverage, not a day count, which cannot tell those two apart.
    """
    def __init__(self,
                 use_free_float: bool = False):
        super().__init__(scheme_name="MarketCapWeighted")
        self.use_free_float = use_free_float

    def _session_for(self,
                     current_date: pd.Timestamp,
                     market_data_provider: DataFetcher) -> pd.Timestamp:
        """The session this rebalance reads from.

        The shared primitive, which `MarketCapRule` also calls: one definition
        of the session in force, so selection and weighting cannot resolve a
        closed day differently.

        Raises:
            CalculationError: If *current_date* falls outside the data's
                coverage. The message names both the date that was asked for
                and the data's actual end, because "ask for an earlier date
                or refresh the store" is the action and neither half of it is
                discoverable from "market cap is zero".
        """
        return _resolve_session(self.scheme_name, "weight",
                                current_date, market_data_provider)

    def _asset_price(self,
                     asset: Equity,
                     session: pd.Timestamp,
                     market_data_provider: DataFetcher) -> tuple[pd.Timestamp, float]:
        """*asset*'s last close at or before *session*, and the day it traded.

        Walked back per name rather than taken from *session* alone, since a
        name can be quiet on a day the market was open.

        Raises:
            CalculationError: If the data carries no close for the name on or
                before *session*. Refusing rather than returning 0.0 is
                deliberate: a zero cap is not a small weight, it is the rest
                of the universe absorbing this name's share, and a cap
                weighting computed over part of its universe is a different
                index — the same fault as the fallback, in miniature.
        """
        session_str = session.strftime('%Y-%m-%d')
        same_day = market_data_provider.fetch_market_data(
            asset.ticker, session_str, session_str)

        if _has_close(same_day):
            return session, float(same_day[_PRICE_COLUMN].iloc[0])

        history = market_data_provider.fetch_market_data(
            asset.ticker, None, session_str)

        closes = (history[_PRICE_COLUMN].dropna().sort_index()
                  if not history.empty and _PRICE_COLUMN in history.columns
                  else pd.Series(dtype=float))

        if closes.empty:
            raise CalculationError(
                calculation_name=self.scheme_name,
                details=(f"{asset.ticker} has no {_PRICE_COLUMN} on or before "
                         f"{session_str}, so it cannot be priced. Weighting the "
                         f"rest of the universe without it would publish a "
                         f"cap-weighted index over a subset of its own "
                         f"constituents. Load the name's prices, or remove it "
                         f"from the universe."))

        return pd.Timestamp(closes.index[-1]), float(closes.iloc[-1])

    def _asset_market_cap(self,
                          asset: Equity,
                          session: pd.Timestamp,
                          market_data_provider: DataFetcher) -> float:
        """One constituent's market cap, read from the day it last traded.

        Shares outstanding and the free-float factor are read on that same
        day rather than on the requested one, so the cap is one coherent
        observation rather than a current share count against an older price.

        Raises:
            CalculationError: If the name cannot be priced, has no positive
                shares outstanding, or — on a free-float index — has no
                usable free-float factor.
        """
        priced_on, price = self._asset_price(asset, session, market_data_provider)
        priced_str = priced_on.strftime('%Y-%m-%d')

        shares_outstanding = market_data_provider.fetch_shares_outstanding(
            asset.ticker, priced_str)

        if shares_outstanding is None or shares_outstanding <= 0:
            raise CalculationError(
                calculation_name=self.scheme_name,
                details=(f"{asset.ticker} has no positive SHARES_OUTSTANDING on "
                         f"{priced_str}, so its market cap is unknown. That "
                         f"column is what this scheme weights by; without it "
                         f"there is no market-cap index to publish."))

        asset_market_cap = price * shares_outstanding

        if not self.use_free_float:
            return asset_market_cap

        free_float_factor = market_data_provider.fetch_free_float_factor(
            asset.ticker, priced_str)

        if free_float_factor is None or not 0.0 <= free_float_factor <= 1.0:
            raise CalculationError(
                calculation_name=self.scheme_name,
                details=(f"{asset.ticker} has no usable FREE_FLOAT on "
                         f"{priced_str}, and this index is free-float adjusted. "
                         f"Using its full market cap instead would weight one "
                         f"name on a different basis from the rest."))

        return asset_market_cap * free_float_factor

    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: dict[str, Any] | None = None) -> dict[Asset, float]:
        """Weights proportional to market cap, or a refusal.

        Raises:
            CalculationError: If *current_date* lies outside the data's
                coverage, if any constituent is unpriceable or is not an
                equity, or if the caps sum to nothing. Nothing here falls
                back to another methodology — see the class docstring.
        """
        if not constituents:
            return {}

        session = self._session_for(current_date, market_data_provider)
        market_caps: dict[Asset, float] = {}

        for asset in constituents:
            if not isinstance(asset, Equity):
                raise CalculationError(
                    calculation_name=self.scheme_name,
                    details=(f"constituent '{asset.asset_id}' is not an equity, "
                             f"so it has no market cap to weight by. Skipping it "
                             f"would publish a cap-weighted index over a subset "
                             f"of its own constituents."))

            market_caps[asset] = self._asset_market_cap(
                asset, session, market_data_provider)

        total_market_cap = sum(market_caps.values())

        if total_market_cap <= 0:
            raise CalculationError(
                calculation_name=self.scheme_name,
                details=(f"the {len(market_caps)} constituents priced at "
                         f"{session:%Y-%m-%d} have a total market cap of "
                         f"{total_market_cap}, so there is nothing to weight "
                         f"by."))

        return {asset: cap / total_market_cap
                for asset, cap in market_caps.items()}


@register(WEIGHTING, "Equal weighted")
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
