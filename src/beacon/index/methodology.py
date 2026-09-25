# src/beacon/index/methodology.py
"""
Index methodology: eligibility rules and weighting schemes.

`EligibilityRuleBase` and `WeightingSchemeBase` are the interfaces a rule or
scheme implements. `MarketCapRule` and `LiquidityRule` screen constituents;
`MarketCapWeighted` and `EqualWeighted` weight them.
"""
import logging
from abc import ABC, abstractmethod

import pandas as pd

from ..asset.base import Asset
from ..asset.equity import Equity, require_equity
from ..catalogue import (
    SELECTION,
    WEIGHTING,
    Display,
    register,
)
from ..data.fetcher import DataFetcher
from ..data.free_float import require_free_float
from ..exceptions import CalculationError
from .context import IndexContext

logger = logging.getLogger(__name__)

# Market-data column names read by the rules/schemes below.
_PRICE_COLUMN = "CLOSE"
_VOLUME_COLUMN = "VOLUME"
_SHARES_COLUMN = "SHARES_OUTSTANDING"
_FREE_FLOAT_COLUMN = "FREE_FLOAT"


def _equity_tickers(assets: list[Asset]) -> list[str]:
    """The tickers among *assets*, skipping anything that is not an equity.

    For warming a session read only. Nothing is decided here, so a non-equity
    is passed over rather than refused — the refusal belongs where the asset is
    actually being assessed or weighted, and `require_equity` makes it there
    with the name of what could not be done.
    """
    return [asset.ticker for asset in assets if isinstance(asset, Equity)]


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


def _rate_into(calculation_name: str,
               asset: Equity,
               to_currency: str,
               date: pd.Timestamp,
               market_data_provider: DataFetcher) -> float:
    """FX converting *asset*'s quote currency into *to_currency*, or a refusal.

    The one place a methodology converts money, over the one rate lookup the
    library has (:meth:`DataFetcher.fx_rate_on`). A market cap is a price
    times a share count in whatever currency the name trades in; comparing two
    of them — to rank, to weight, or to screen against a bound — is only
    meaningful once they are the same money.

    Raises:
        CalculationError: If the pair is unknown, naming it, the date and the
            name. Falling back to the local number is the substitution that
            produced BN-188: a yen cap outranked a dollar one on magnitude
            alone and every weight in the index was wrong, with nothing
            downstream looking odd enough for anyone to ask.
    """
    rate = market_data_provider.fx_rate_on(asset.currency, to_currency, date)

    if rate is not None:
        return rate

    raise CalculationError(
        calculation_name=calculation_name,
        details=(f"no {asset.currency.upper()}/{to_currency.upper()} rate on "
                 f"or before {date:%Y-%m-%d}, so {asset.ticker}'s market cap "
                 f"cannot be expressed in {to_currency.upper()}. Using its "
                 f"local number instead would compare it with the rest of the "
                 f"universe on magnitude alone, which is a different index "
                 f"under the same heading. Load the pair, or drop the name "
                 f"from the universe."))


class EligibilityRuleBase(ABC):
    """Abstract base class for an eligibility rule.

    Eligibility rules determine whether an asset can be part of an index.
    Subclasses implement `is_eligible`, and may override `required_columns`
    and `prepare`.

    Args:
        rule_name: The rule's name, used in logs and error messages.
    """
    def __init__(self,
                 rule_name: str):
        self.rule_name = rule_name

    def required_columns(self) -> frozenset[str]:
        """The market-data columns this rule reads, declared up front.

        Checked against the dataset before a run does any work, so a store
        with no SHARES_OUTSTANDING column is refused at the start as "this
        rule needs SHARES_OUTSTANDING and the dataset has none", rather than
        at the first rebalance as "N0 has no SHARES_OUTSTANDING on
        2024-01-02", which is true, and sends a reader to inspect one company
        whose data is fine.

        Empty by default rather than abstract, so a rule written outside this
        package keeps working. It is then simply not checked up front, and
        fails at the first read instead. Every rule shipped here declares its
        own.
        """
        # The up-front column check is BN-217 (`beacon.index.requirements`).
        return frozenset()

    # The prepare hook is BN-190.
    def prepare(self,  # noqa: B027 — an optional hook, not part of the interface
                candidates: list[Asset],
                current_date: pd.Timestamp,
                market_data_provider: DataFetcher,
                context: IndexContext | None = None) -> None:
        """Read in one go whatever this rule is about to read per name.

        Called once with the whole candidate set before `is_eligible` is asked
        about any of them. It decides nothing and returns nothing: a rule must
        give exactly the same answers whether or not it prepared, because this
        is a hint about *how* to read rather than about *what* is eligible.
        Doing nothing is therefore the right default, and it is the base
        implementation.

        It exists because `is_eligible` is a predicate over one asset, so a
        rule reading market data reads it a name at a time, and each read
        slices a frame whose size is the whole store: the cost of one lookup
        grows with the universe around it rather than with the row it wants.

        Args:
            candidates: Everything that reached this rung, in order. A rule
                that ranks rather than screens would want this set too; that is
                not what this is for, but it is the same set.
            current_date: The date selection is being made at.
            market_data_provider: The data source the reads will go to.
            context: What the index settles for its rules, as for
                :meth:`is_eligible`.
        """

    @abstractmethod
    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: IndexContext | None = None) -> bool:
        """
        Checks if a given asset is eligible based on this rule.

        Args:
            asset: The asset to check.
            current_date: The date on which eligibility is being assessed.
            market_data_provider: A DataFetcher instance to get necessary market data
                                  (e.g., market cap, trading volume).
            context: What the index the rule is running inside reports in and
                settles. None when the rule is evaluated outside an index, in
                which case nothing here may assume a currency it was not told.

        Returns:
            True if the asset is eligible, False otherwise.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(rule_name='{self.rule_name}')"


# --- Example Eligibility Rules ---

# BN-182: reading the exact date instead of resolving back excluded *every*
# name on a closed day, so the universe emptied, the weighting was handed
# nothing, and an index of no constituents computed a coherent level of zero.
#
# BN-188: the published help text always said the bounds were in the index
# currency, while the code compared a name's local number against them, so a
# 5bn floor admitted a name whose yen cap read 5.2bn and excluded a genuinely
# larger one quoted in a strong currency.
@register(SELECTION, "Market capitalisation",
          fields={
              "min_market_cap": Display("Minimum market cap", order=1,
                                        help="In the index currency, converted "
                                             "at the rate on the day. Blank for "
                                             "no floor."),
              "max_market_cap": Display("Maximum market cap", order=2,
                                        help="In the index currency, converted "
                                             "at the rate on the day. Blank for "
                                             "no ceiling."),
          })
class MarketCapRule(EligibilityRuleBase):
    """Eligibility by market capitalisation, read from a resolved session.

    The cap is price times shares outstanding, both read on the same session.
    A name with no price or no positive share count on that session is
    excluded (and logged as a warning).

    **Dates resolve backwards into the data.** A weekend, a holiday or any
    date inside the data's coverage that carries no bar is read at the last
    session on or before it, because that is the universe the index actually
    held through the closure. This is the same resolution
    :class:`MarketCapWeighted` performs, through the same primitive, so
    selection and weighting cannot resolve a closed day differently.

    **The bounds are in the index's currency.** The cap is converted at the
    session's rate before it meets either bound; a missing FX pair refuses
    rather than falling back to the local figure. Outside an index there is no
    context and so no currency to convert into, and the bounds are then read
    in the asset's own currency. Inside an index the calculator always
    supplies one.

    **Past the last bar it refuses rather than excluding.** A rule that cannot
    evaluate has not found the asset ineligible, it has failed, and the two
    must not share an answer: "not in the index" is a published fact about a
    name, while "the data does not reach that date" is a fact about the store.

    Args:
        min_market_cap: Lowest cap admitted, in the index currency. None for
            no floor.
        max_market_cap: Highest cap admitted, in the index currency. None for
            no ceiling.

    Raises:
        ValueError: If *min_market_cap* is greater than *max_market_cap*.
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

    def required_columns(self) -> frozenset[str]:
        """A cap is price times shares, so both, whichever bound is set."""
        return frozenset({_PRICE_COLUMN, _SHARES_COLUMN})

    def prepare(self,
                candidates: list[Asset],
                current_date: pd.Timestamp,
                market_data_provider: DataFetcher,
                context: IndexContext | None = None) -> None:
        """Read the whole candidate set's session in one slice.

        Every name this rule is about to be asked about is read on the same
        session, for the same two columns. Warming that session turns the
        per-name reads into dictionary lookups and, because the weighting
        scheme then reads the survivors on the same session, makes the second
        pricing of every surviving name free.

        Raises:
            CalculationError: If *current_date* lies outside the data's
                coverage. That is the same refusal `is_eligible` makes over the
                same date, arriving one call earlier; a rule that cannot
                resolve its session cannot assess anything.
        """
        if not candidates:
            return

        session = _resolve_session(self.rule_name, "assess eligibility",
                                   current_date, market_data_provider)

        market_data_provider.warm_session(_equity_tickers(candidates), session)

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: IndexContext | None = None) -> bool:
        """Whether *asset*'s market cap at the resolved session clears the bounds.

        Raises:
            CalculationError: If *asset* is not an equity, if *current_date*
                lies outside the data's coverage, or if the cap cannot be
                converted into the index currency, so the rule cannot be
                evaluated at all. Nothing here turns a failure into an
                exclusion (see the class docstring).
        """
        equity = require_equity(asset, self.rule_name,
                                "be assessed for market-cap eligibility")

        session = _resolve_session(self.rule_name, "assess eligibility",
                                   current_date, market_data_provider)
        date_str = session.strftime('%Y-%m-%d')

        current_price = market_data_provider.fetch_price(equity.ticker, date_str,
                                                         _PRICE_COLUMN)

        if current_price is None:
            logger.warning(
                f"MarketCapRule: Could not fetch price for {equity.ticker} "
                f"on {date_str}.")
            return False

        # Read on the same session as the price, so the cap is one coherent
        # observation rather than a current share count against an older close.
        shares_outstanding = market_data_provider.fetch_shares_outstanding(
            equity.ticker, date_str)

        if shares_outstanding is None or shares_outstanding <= 0:
            logger.warning(
                f"MarketCapRule: Could not fetch valid shares outstanding for "
                f"{equity.ticker} on {date_str}.")
            return False

        market_cap = current_price * shares_outstanding

        # Into the index's money before it meets a bound stated in that money.
        if context is not None:
            market_cap *= _rate_into(self.rule_name, equity, context.currency,
                                     session, market_data_provider)

        if self.min_market_cap is not None and market_cap < self.min_market_cap:
            logger.debug(
                f"MarketCapRule: {equity.ticker} (MCap: {market_cap:.2f}) below "
                f"min_market_cap {self.min_market_cap:.2f}")
            return False

        if self.max_market_cap is not None and market_cap > self.max_market_cap:
            logger.debug(
                f"MarketCapRule: {equity.ticker} (MCap: {market_cap:.2f}) above "
                f"max_market_cap {self.max_market_cap:.2f}")
            return False

        logger.debug(f"MarketCapRule: {equity.ticker} (MCap: {market_cap:.2f}) is eligible.")

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
    """Eligibility by trading liquidity: average daily volume or value.

    The averages are taken over the last *lookback_days* rows of market data
    on or before the date asked about. A name with fewer than 80% of that many
    rows, or with the needed column missing or empty, is excluded (and logged
    as a warning). Values are in the currency the name trades in, not
    converted.

    Args:
        min_avg_daily_volume: Lowest average shares traded per day. None for
            no volume floor.
        min_avg_daily_value: Lowest average close times volume per day. None
            for no value floor.
        lookback_days: Trading days the averages are taken over.

    Raises:
        ValueError: If *lookback_days* is not positive.
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

    def required_columns(self) -> frozenset[str]:
        """Volume for either threshold, and the close too for a value one.

        Declared from the thresholds actually set, because they read different
        things. The declaration matters here: `is_eligible` treats a missing
        VOLUME column as "not liquid enough" and excludes the name, so without
        the up-front check a store lacking the column would exclude **every**
        name and the run would fail as "index holds nothing on its base date"
        with no mention of volume.
        """
        needed: set[str] = set()

        if self.min_avg_daily_volume is not None:
            needed.add(_VOLUME_COLUMN)

        if self.min_avg_daily_value is not None:
            needed.update({_PRICE_COLUMN, _VOLUME_COLUMN})

        return frozenset(needed)

    def is_eligible(self,
                    asset: Asset,
                    current_date: pd.Timestamp,
                    market_data_provider: DataFetcher,
                    context: IndexContext | None = None) -> bool:
        """Whether *asset*'s traded volume and value over the lookback qualify.

        No session resolution here, and none needed: this reads a *window*
        ending at *current_date*, so a closed day is already spanned by the
        days around it rather than being the single day everything hangs on.

        Errors are not caught. A rule that throws has not said the asset is
        ineligible, and the two answers must not be spelled the same.

        Raises:
            CalculationError: If *asset* is not an equity, so there is no
                ticker to read volume against.
        """
        # Errors propagate since BN-182; the equity requirement is BN-185.
        equity = require_equity(asset, self.rule_name,
                                "be assessed for liquidity")

        # Fetch more to ensure enough trading days
        start_lookback = (
            current_date - pd.Timedelta(days=self.lookback_days * 2)).strftime('%Y-%m-%d')
        end_lookback = current_date.strftime('%Y-%m-%d')

        price_df = market_data_provider.fetch_market_data(
            equity.ticker, start_lookback, end_lookback)

        if price_df.empty or price_df.shape[0] < (self.lookback_days / 2): # Ensure some data
            logger.warning(
                f"LiquidityRule: Insufficient historical price data for "
                f"{equity.ticker} for period ending {end_lookback}.")
            return False

        # Ensure we have data up to current_date or shortly before
        # (single-identifier market data is indexed by date).
        price_df = price_df[price_df.index <= current_date].tail(self.lookback_days)

        # Heuristic: need at least 80% of lookback days
        if price_df.shape[0] < (self.lookback_days * 0.8):
            logger.warning(
                f"LiquidityRule: Not enough trading days "
                f"({price_df.shape[0]}/{self.lookback_days}) for {equity.ticker} "
                f"for ADV calc.")
            return False

        if self.min_avg_daily_volume is not None:
            if (_VOLUME_COLUMN not in price_df.columns
                    or price_df[_VOLUME_COLUMN].isnull().all()):
                logger.warning(f"LiquidityRule: Volume data missing for {equity.ticker}.")
                return False
            avg_daily_volume = price_df[_VOLUME_COLUMN].mean()
            if avg_daily_volume < self.min_avg_daily_volume:
                logger.debug(
                    f"LiquidityRule: {equity.ticker} (ADV: {avg_daily_volume:.0f}) below "
                    f"min volume {self.min_avg_daily_volume:.0f}")
                return False

        if self.min_avg_daily_value is not None:
            if (_PRICE_COLUMN not in price_df.columns
                    or _VOLUME_COLUMN not in price_df.columns
                    or price_df[_PRICE_COLUMN].isnull().all()
                    or price_df[_VOLUME_COLUMN].isnull().all()):
                logger.warning(
                    f"LiquidityRule: Price or Volume data missing for ADTV "
                    f"calculation for {equity.ticker}.")
                return False
            avg_daily_value = (price_df[_PRICE_COLUMN] * price_df[_VOLUME_COLUMN]).mean()
            if avg_daily_value < self.min_avg_daily_value:
                logger.debug(
                    f"LiquidityRule: {equity.ticker} (ADTV: {avg_daily_value:.2f}) below "
                    f"min value {self.min_avg_daily_value:.2f}")
                return False

        logger.debug(f"LiquidityRule: {equity.ticker} is eligible.")

        return True

# Other example stubs:
# class FreeFloatRule(EligibilityRuleBase): ...
# class ListingLocationRule(EligibilityRuleBase): ...


class WeightingSchemeBase(ABC):
    """Abstract base class for a weighting scheme.

    Weighting schemes determine the proportion of each constituent in an
    index. Subclasses implement `calculate_weights`, and may override
    `required_columns`.

    Args:
        scheme_name: The scheme's name, used in logs and error messages.
    """
    def __init__(self,
                 scheme_name: str):
        self.scheme_name = scheme_name

    def required_columns(self) -> frozenset[str]:
        """The market-data columns this scheme reads, declared up front.

        The same contract as :meth:`EligibilityRuleBase.required_columns`, and
        derived from the scheme's own inputs where they change what it reads:
        a scheme's parameters *are* the declaration, so nothing asks a store
        for a column the configured scheme does not use.
        """
        return frozenset()

    @abstractmethod
    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: IndexContext | None = None) -> dict[Asset, float]:
        """
        Calculates the weight for each constituent asset.

        Args:
            constituents: A list of assets that are eligible for the index.
            current_date: The date for which weights are being calculated.
            market_data_provider: A DataFetcher instance.
            context: What the index the scheme is running inside reports in
                and settles. None when it is invoked outside an index.

        Returns:
            A dictionary mapping each Asset object to its calculated weight (float).
            The sum of weights should typically be 1.0.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scheme_name='{self.scheme_name}')"


# --- Example Weighting Schemes ---

# BN-179 removed the equal-weight fallback. BN-188: this used to weight
# ``price x shares`` in whatever money the name traded in, so a yen name
# entered the sum as though a thousand billion yen were a thousand billion
# dollars, a fifteen-fold error on its own weight and a wrong weight on every
# other constituent with it.
@register(WEIGHTING, "Market capitalisation weighted",
          fields={
              "use_free_float": Display("Free-float adjusted", order=1,
                                        help="Weight by the freely traded portion "
                                             "rather than by full market cap."),
          })
class MarketCapWeighted(WeightingSchemeBase):
    """Market capitalisation weighting, optionally free-float adjusted.

    Each constituent's weight is its cap (price times shares outstanding,
    times the free-float factor when *use_free_float* is set) over the sum of
    all caps. A name is priced at its last close on or before the rebalance
    session, and its shares, free float and FX rate are read on that same day.

    **Every path either weights by real market caps or refuses.** There is no
    equal-weight fallback: an index that comes out equal-weighted because the
    caps could not be read is not a degraded market-cap index, it is a
    different index published under the same heading, and nothing downstream
    looks wrong enough for anyone to ask (the levels are right, the weights
    sum, the backtest tracks).

    **Dates resolve backwards into the data.** A request for a weekend, a
    holiday, or any date inside the data's coverage that carries no bar reads
    the last session on or before it, because that is the composition the
    index actually held through the closure rather than an approximation of
    one. Past the last bar it refuses, since there the same read would be a
    stale print presented as the current one. The bound is the data's own
    coverage, not a day count, which cannot tell those two apart.

    **Caps are compared in one currency.** A universe spanning currencies is
    converted into the index currency before the caps are summed, and a
    missing FX pair refuses. A universe quoted in a single currency needs no
    conversion at all, since converting every cap by the same rate cannot move
    a weight. Called outside an index (no context) over several currencies,
    it refuses, since there is no currency to compare in.

    Args:
        use_free_float: Weight by the freely traded portion of each cap
            rather than the full cap. Requires a FREE_FLOAT column.
    """
    def __init__(self,
                 use_free_float: bool = False):
        super().__init__(scheme_name="MarketCapWeighted")
        self.use_free_float = use_free_float

    def required_columns(self) -> frozenset[str]:
        """Price and shares, and free float only when this scheme uses it.

        `use_free_float` is the declaration: every free-float read in a run is
        behind it, so a scheme that is not float-adjusted never asks the store
        for the column and must not be refused for lacking it.
        """
        needed = {_PRICE_COLUMN, _SHARES_COLUMN}

        if self.use_free_float:
            needed.add(_FREE_FLOAT_COLUMN)

        return frozenset(needed)

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
        same_day = market_data_provider.fetch_price(asset.ticker, session_str,
                                                    _PRICE_COLUMN)

        if same_day is not None:
            return session, same_day

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

    def _target_currency(self,
                         constituents: list[Asset],
                         context: IndexContext | None) -> str | None:
        """The currency the caps are compared in, or None when none is needed.

        A weight is a ratio, so converting every cap by the *same* rate cannot
        move it: a universe quoted in one currency weights identically however
        the index reports, and this answers None there. That is not a
        shortcut — it is what stops a single-currency index refusing over an
        FX pair that could not have changed its answer, and it is why BN-188
        moves no existing result.

        Across currencies the conversion is the whole point. Mathematically
        any base gives the same weights, since a change of base is another
        common factor; the index's own is the one to use in practice, because
        it is the base a store actually holds pairs into and the one the
        index's levels are already quoted in.

        Raises:
            CalculationError: If the universe spans currencies and no context
                was supplied, so there is no currency to compare in. Summing
                the local numbers is the arithmetic this exists to stop, and a
                scheme silently picking a base of its own would be the same
                fault wearing a better-looking implementation.
        """
        currencies = {asset.currency.upper() for asset in constituents}

        if len(currencies) <= 1:
            return None

        if context is not None:
            return context.currency

        raise CalculationError(
            calculation_name=self.scheme_name,
            details=(f"the universe is quoted in {len(currencies)} currencies "
                     f"({', '.join(sorted(currencies))}) and no index context "
                     f"was supplied, so there is no currency to compare their "
                     f"market caps in. Adding them as they stand would weight "
                     f"a name by the size of its currency's unit. Call this "
                     f"through an IndexCalculator, or pass an IndexContext."))

    def _asset_market_cap(self,
                          asset: Equity,
                          session: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          to_currency: str | None = None) -> float:
        """One constituent's market cap, read from the day it last traded.

        Shares outstanding and the free-float factor are read on that same
        day rather than on the requested one, so the cap is one coherent
        observation rather than a current share count against an older price.

        Args:
            asset: The constituent.
            session: The session the rebalance reads from.
            market_data_provider: The data source.
            to_currency: Currency to express the cap in. None leaves it in the
                name's own, which :meth:`_target_currency` asks for only when
                every constituent shares it.

        Raises:
            CalculationError: If the name cannot be priced, has no positive
                shares outstanding, cannot be converted into *to_currency*,
                or — on a free-float index — has no usable free-float factor.
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

        # Converted on the day the name was priced, so the cap stays one
        # coherent observation: a rate from the requested date against a close
        # from an earlier one would mix two days into a single number.
        if to_currency is not None:
            asset_market_cap *= _rate_into(self.scheme_name, asset, to_currency,
                                           priced_on, market_data_provider)

        if not self.use_free_float:
            return asset_market_cap

        return asset_market_cap * require_free_float(market_data_provider,
                                                     asset.ticker,
                                                     priced_str,
                                                     self.scheme_name)

    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: IndexContext | None = None) -> dict[Asset, float]:
        """Weights proportional to market cap, or a refusal.

        Raises:
            CalculationError: If *current_date* lies outside the data's
                coverage, if any constituent is unpriceable, unconvertible or
                is not an equity, has no positive shares outstanding or (when
                free-float adjusted) no usable free-float factor, if the
                universe spans currencies with no context to compare them in,
                or if the caps sum to nothing. Nothing here falls back to
                another methodology (see the class docstring).
        """
        if not constituents:
            return {}

        session = self._session_for(current_date, market_data_provider)
        to_currency = self._target_currency(constituents, context)

        # The same session a selection rule has just read, over a subset of the
        # names it read it for, so this keeps that panel rather than building
        # another — which is what stops every surviving name being priced twice
        # in one rebalance (BN-190). Where no rule read anything, it is still
        # one slice for the whole constituent list rather than one per name.
        market_data_provider.warm_session(_equity_tickers(constituents), session)

        market_caps: dict[Asset, float] = {}

        for asset in constituents:
            equity = require_equity(asset, self.scheme_name,
                                    "be weighted by market cap, having none")

            market_caps[asset] = self._asset_market_cap(
                equity, session, market_data_provider, to_currency)

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
    """Equal weighting: every constituent gets ``1 / n``.

    Reads no market data. An empty constituent list gets empty weights.
    """
    def __init__(self) -> None:
        super().__init__(scheme_name="EqualWeighted")

    def required_columns(self) -> frozenset[str]:
        """Nothing: equal weights are decided without reading the market.

        Stated rather than inherited, so the absence is a decision a reader
        can see. The index still needs a price column to value its holdings
        daily, but that is the calculator's requirement, not this scheme's.
        """
        return frozenset()

    def calculate_weights(self,
                          constituents: list[Asset],
                          current_date: pd.Timestamp,
                          market_data_provider: DataFetcher,
                          context: IndexContext | None = None) -> dict[Asset, float]:
        weights: dict[Asset, float] = {}
        num_constituents = len(constituents)

        # BN-184, triaged as leave. Empty weights for an empty constituent
        # list is the answer, not a substitute for one: there is nothing to
        # weight, and any non-empty result would be invented. The emptiness
        # itself is not swallowed — `calculate_constituent_weights` never
        # reaches a scheme with no constituents, and an index that holds
        # nothing on its base date is refused outright by the calculator's
        # `_require_a_base_composition`, so the condition surfaces where it
        # can actually be acted on.
        if num_constituents > 0:
            weight_per_constituent = 1.0 / num_constituents
            for asset in constituents:
                weights[asset] = weight_per_constituent
        else:
            logger.warning("EqualWeighted: No constituents provided. Returning empty weights.")

        return weights

# class CorporateActionRule: ... (For specific handling if not covered by divisor)
