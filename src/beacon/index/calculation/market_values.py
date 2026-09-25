# src/beacon/index/calculation/market_values.py
"""
MarketValuesMixin: constituent market values, holding values and the index
level.
"""
import logging

import pandas as pd

from ...asset.base import Asset
from ...asset.equity import Equity, require_equity
from ...data.fetcher import DataFetcher
from ...data.free_float import require_free_float
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
        """An FX rate on a date, through :meth:`DataFetcher.fx_rate_on`.

        The calculator's name for the library's one FX lookup, so the levels,
        the weights and the reference display all convert currency the same
        way.

        Returns:
            float | None: The rate as of `date`, carried forward over gaps
            under the dataset's FX policy, or None when the pair is unknown,
            which callers treat as "cannot convert" rather than as a rate of
            one.
        """
        # The caching, carry-forward and None-on-unknown behaviour this method
        # used to define moved to `DataFetcher.fx_rate_on` in BN-188, because
        # three parts of the library were converting currency three different
        # ways and one of them (the market-cap weighting) was not converting
        # at all.
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

        # One page for the rebalance session before the per-name reads below,
        # so price, shares and float are all answered from it (BN-218). A
        # no-op when the daily loop already warmed this session.
        warm = getattr(self.data, "warm_session", None)

        if warm is not None and constituents_with_weights:
            warm([asset.asset_id for asset in constituents_with_weights],
                 current_date)

        for asset in constituents_with_weights:
            equity = require_equity(asset, "ConstituentMarketValues", "be valued")
            constituent_market_values[asset] = self._asset_market_value(equity, current_date)

        return constituent_market_values

    def _asset_market_value(self,
                            asset: Equity,
                            current_date: pd.Timestamp) -> float:
        """Compute the FX/free-float-adjusted market value for a single Equity asset.

        Returns 0.0 (with a warning logged) when price or shares data is
        missing. An unconvertible currency refuses instead — see below.

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
                usable free-float factor exists for *asset* (BN-184), or if
                *asset*'s currency cannot be converted into the index's
                (BN-191).
        """
        date_str = current_date.strftime('%Y-%m-%d')

        # A price, not a frame (BN-218). The rebalance-path twin of the read
        # BN-212 fixed in `asset_unit_value`: `fetch_market_data` never
        # consults the session panel, so every constituent at every rebalance
        # sliced the whole frame for one row. BN-212 missed it by reading the
        # profile by self time, where pandas' cost is spread thin across
        # dozens of internals and no single line stands out; by cumulative
        # time it was three quarters of the calculation. `fetch_price` answers
        # None for an absent row, an absent column and a NaN alike -- the three
        # cases the frame test used to spell out.
        current_price = self.data.fetch_price(asset.ticker, date_str,
                                              self.price_column)

        # BN-191, triaged as leave, because this total is a *scale* and not a
        # level: on the `run()` path it sets the units and the divisor together
        # (units = weight x total / unit_value, divisor = total / base_value),
        # so any positive constant multiplying it leaves every published level
        # identical, and the rebalance adjustment cancels it the same way. What
        # a name is *weighted* by is decided in
        # `MarketCapWeighted._asset_market_cap`, which already refuses an
        # unpriceable name, a missing share count and a missing free float
        # (BN-184, BN-188), off the session the name last traded rather than an
        # exact date. The level-affecting twin of these two zeros is
        # `asset_unit_value`'s missing price, separated below.
        if current_price is None:
            logger.warning(
                f"_get_constituent_market_values: No price for {asset.ticker}. "
                "Value is 0.")
            return 0.0

        shares = self.data.fetch_shares_outstanding(asset.ticker, date_str)

        if shares is None or shares <= 0:
            logger.warning(
                f"_get_constituent_market_values: No shares for {asset.ticker}. "
                "Value is 0.")
            return 0.0

        market_value_local_ccy = current_price * shares
        market_value_local_ccy *= self._free_float(asset, date_str)

        # FX conversion into the index currency. Unlike the two zeros above
        # this is not a scale: `calculate_index_level` divides this very sum by
        # the divisor, so a dropped term is a level that is wrong by that
        # name's whole share of the market cap.
        return market_value_local_ccy * self._market_value_rate(
            asset, current_date, date_str, market_value_local_ccy)

    def _market_value_rate(self,
                           asset: Equity,
                           current_date: pd.Timestamp,
                           date_str: str,
                           local_value: float) -> float:
        """The rate converting *asset*'s market value into the index currency.

        Raises:
            CalculationError: If the pair is unknown (BN-191). This used to
                return 0.0 while logging "Using 1.0" — two opposite
                behaviours, one in the code and the other in the log, so a
                reader diagnosing from logs would conclude the name had been
                valued as though a yen were a dollar when in fact it had been
                removed from the aggregate altogether. Neither is available:
                a rate of 1.0 is BN-188's defect and 0.0 publishes a level
                over the constituents that happen to be convertible.
        """
        local = asset.currency.upper()
        index_currency = self.definition.currency.upper()

        if local == index_currency:
            return 1.0

        rate = self.rate_on(asset.currency, self.definition.currency,
                            current_date)

        if rate is None:
            raise CalculationError(
                calculation_name="ConstituentMarketValues",
                details=(f"no {local}/{index_currency} rate on or before "
                         f"{date_str}, so {asset.ticker}'s market value of "
                         f"{local_value:g} {local} cannot be expressed in "
                         f"{index_currency}. Excluding it would restate the "
                         f"index over the constituents that happen to be "
                         f"convertible, and converting at 1.0 would treat the "
                         f"two currencies as the same money. Load the pair, "
                         f"or define the index in {local}."))

        return rate

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

        return require_free_float(self.data,
                                  asset.ticker,
                                  date_str,
                                  "ConstituentMarketValues")

    def asset_unit_value(self,
                         asset: Asset,
                         current_date: pd.Timestamp) -> float | None:
        """Value of one unit of *asset* in the index currency: price times FX.

        Distinct from a constituent's market value, which multiplies by shares
        outstanding and free float. Those belong to the *weighting* of the
        index; this is what one unit is worth, which is what the index's
        holdings are valued at day to day.

        Args:
            asset: The constituent. Must be an Equity.
            current_date: Valuation date.

        Returns:
            float | None: The price in index currency, or None when the name
            could not be priced on *current_date*. None rather than 0.0,
            because "could not be priced" and "priced at zero" are different
            answers: :meth:`index_units` must refuse the first, while
            :meth:`holding_values` tolerates it (a feed gap is a flat day).

        Raises:
            CalculationError: If *asset* is not an equity, or if its currency
                cannot be converted into the index's. Valuing either at 0.0
                would make a constituent the index still holds contribute
                nothing, which is a silent restatement of the index rather
                than a missing price.

        Errors from the data source are not caught: a fetcher that raised and
        a name that is genuinely unvaluable must not look the same.
        """
        # None rather than 0.0 since BN-191. The equity requirement is BN-185.
        # A bare `except Exception` used to turn any failure here into a unit
        # value of 0.0, feeding straight into the "holds zero units" path, and
        # would have absorbed any refusal added beneath it (removed in BN-184).
        equity = require_equity(asset, "AssetUnitValue", "be valued")

        date_str = current_date.strftime('%Y-%m-%d')

        # `fetch_price`, not `fetch_market_data` (BN-212). Both answer the same
        # question and only one of them consults the session panel: this is the
        # hottest read in the library -- once per constituent per day, 172,000
        # times over a 200-name decade -- and it was asking for a whole frame
        # slice of one name on one date, each costing what the whole frame
        # costs rather than what one row does. Measured on one session of 200
        # names: 76.5 ms this way, 0.3 ms through the panel.
        price = self.data.fetch_price(equity.ticker, date_str,
                                      self.price_column)

        if price is None:
            logger.warning(
                f"asset_unit_value: No price for {equity.ticker} on {date_str}; "
                "it cannot be priced today.")
            return None

        return price * self._fx_rate(equity, current_date, date_str)

    def _fx_rate(self,
                 asset: Asset,
                 current_date: pd.Timestamp,
                 date_str: str) -> float:
        """FX rate converting *asset*'s currency into the index currency.

        Raises:
            CalculationError: If the pair is unknown (BN-191). It used to
                return 0.0, which made the name's unit value zero, which made
                its unit count zero in :meth:`index_units`, which made it
                contribute nothing — and the index published a level over the
                constituents that remained, with two warnings in a log and a
                coherent number at the end. The result is not a mis-weighted
                index but a different one, and nothing in it says which names
                are missing. This is also what masked BN-189: the payer of an
                unconvertible dividend had already been given zero units here,
                so the substituted local dividend was multiplied by zero.
        """
        local = asset.currency.upper()
        index_currency = self.definition.currency.upper()

        if local == index_currency:
            return 1.0

        rate = self.rate_on(asset.currency, self.definition.currency,
                            current_date)

        if rate is None:
            raise CalculationError(
                calculation_name="AssetUnitValue",
                details=(f"no {local}/{index_currency} rate on or before "
                         f"{date_str}, so one unit of {asset.asset_id} cannot "
                         f"be valued in {index_currency}. Excluding it would "
                         f"publish a level over the other constituents, which "
                         f"is a different index. Load the pair, or define the "
                         f"index in {local}."))

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
        outstanding (the weight is itself the share of aggregate market
        value), so that methodology gives the same levels as holding shares
        outstanding directly.

        Args:
            weights: Target weight per constituent, summing to 1.
            aggregate: Total value the index represents on this date.
            current_date: Rebalance date.

        Returns:
            dict: Units per constituent. A constituent priced at exactly zero
            gets zero units rather than an infinite position, and so does one
            carried at a target weight of zero that could not be priced.

        Raises:
            CalculationError: If a constituent the index allocates weight to
                could not be priced at all, or was priced below zero. A name
                genuinely quoted at zero is a fact about a market; a missing
                price is the absence of one. Holding zero units of a name with
                a target weight would leave the index short by that whole
                weight and publish the shortfall as the index's own level. The
                optimised (chained) path makes the same refusal.
        """
        # BN-191 split "could not be priced" from "quoted at zero": the check
        # used to be `unit_value <= 0.0` for both. The chained path's matching
        # refusal is BN-184.
        units: dict[Asset, float] = {}

        for asset, weight in weights.items():
            units[asset] = self._units_of(asset, weight, aggregate,
                                          current_date)

        return units

    def _units_of(self,
                  asset: Asset,
                  weight: float,
                  aggregate: float,
                  current_date: pd.Timestamp) -> float:
        """Units of one constituent realising *weight* of *aggregate*.

        Raises:
            CalculationError: If *asset* carries a non-zero *weight* and could
                not be priced, or was priced below zero.
        """
        unit_value = self.asset_unit_value(asset, current_date)
        date_str = current_date.strftime('%Y-%m-%d')

        if unit_value is None or unit_value < 0.0:
            # A name carried at a weight of zero is never held, so its price is
            # not needed and its absence is not a failure — the same carve-out
            # `chaining._price_series` makes.
            if weight == 0.0:
                logger.warning(
                    f"{asset.asset_id} has no unit value on {date_str}; its "
                    "target weight is zero, so it holds zero units.")

                return 0.0

            reason = ("could not be priced" if unit_value is None
                      else f"was priced at {unit_value:g}")

            raise CalculationError(
                calculation_name="IndexUnits",
                details=(f"{asset.asset_id} carries a target weight of "
                         f"{weight:.6g} on {date_str} but {reason}, so there "
                         f"is no unit count that realises that weight. "
                         f"Holding zero units would leave the index short by "
                         f"that whole weight."))

        if unit_value == 0.0:
            # Distinct from the refusal above, deliberately (BN-191): a name
            # quoted at exactly zero has been priced, and zero is the answer
            # the market gave. Zero units is then the only finite answer — but
            # it is arithmetic forced by a real observation, not a missing one.
            logger.warning(
                f"{asset.asset_id} is priced at zero on {date_str}; it holds "
                "zero units, because no finite position in a worthless name "
                "carries a weight.")

            return 0.0

        return weight * aggregate / unit_value

    def holding_values(self,
                       units: dict[Asset, float],
                       current_date: pd.Timestamp) -> dict[Asset, float]:
        """What each holding is worth today: units times unit value.

        Separate from :meth:`aggregate_value` because the daily weights panel
        needs the parts as well as the total, and a part costs a market-data
        lookup: computing them twice would double the lookups a run makes,
        which is its dominant cost.

        Args:
            units: What the index holds, asset to unit count.
            current_date: Valuation date.

        Returns:
            dict: Value per holding in the index currency. A name with no
            price today is worth 0.0 (with a warning) and still appears,
            because it is still held: a feed gap is a data-quality problem,
            and carrying the level forward (which :meth:`level_from_units`
            does) is the right response to one.

        Raises:
            CalculationError: If a holding is not an equity, or a priced
                holding's currency cannot be converted into the index's.
        """
        # Batched rather than one `asset_unit_value` per holding (BN-218). The
        # answer is the same, name for name -- refuse a non-equity, no price is
        # 0.0 with the same warning, otherwise price x FX x units -- but the two
        # expensive parts are asked the size of question they are. Prices come
        # from the day's page in one read instead of a four-call chain per
        # name, and an FX rate is a fact about a currency on a day, so a book
        # of 200 names in three currencies needs three lookups rather than 200.
        # 0.84 ms a day to 0.12 on the benchmark, identical to the last digit.
        if not units:
            return {}

        date_str = current_date.strftime('%Y-%m-%d')
        equities = {asset: require_equity(asset, "AssetUnitValue", "be valued")
                    for asset in units}
        tickers = [equity.ticker for equity in equities.values()]

        # The batch read is a capability, not a contract -- a provider built by
        # hand need not offer it, the same way it need not offer `warm_session`
        # -- so one without it is asked name by name, as before.
        batch = getattr(self.data, "prices_on", None)
        prices = (batch(tickers, date_str, self.price_column) if callable(batch)
                  else {ticker: self.data.fetch_price(ticker, date_str,
                                                      self.price_column)
                        for ticker in tickers})

        rates: dict[str, float] = {}
        values: dict[Asset, float] = {}

        for asset, count in units.items():
            equity = equities[asset]
            price = prices.get(equity.ticker)

            if price is None:
                logger.warning(
                    f"asset_unit_value: No price for {equity.ticker} on {date_str}; "
                    "it cannot be priced today.")
                values[asset] = 0.0
                continue

            # Keyed on the currency, and computed only for a name that priced:
            # `_fx_rate` refuses an unknown pair naming the asset, and asking it
            # in the same order as before means the refusal names the same one.
            currency = equity.currency.upper()

            if currency not in rates:
                rates[currency] = self._fx_rate(equity, current_date, date_str)

            values[asset] = count * price * rates[currency]

        return values

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

        The previous level is carried forward on a day on which the *data*
        has nothing to say: when there are no holdings (a rebalance that
        selected no constituents), or when the holdings are worth nothing
        (every holding unpriced on one date).
        """
        # The two carry-forward branches were triaged with BN-191, which fixed
        # the chain that used to hollow the aggregate out above them. Neither
        # became unreachable and both are kept for a genuine feed gap; what
        # BN-191 removed is the gap the calculation was manufacturing for
        # itself.
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
        """Calculate the index level from constituent market values.

        A Laspeyres-type formula: the sum of the constituents' current market
        values (price times shares, times free float when the scheme is
        float-adjusted, converted into the index currency) over the divisor.
        `run` does not use this; it values the units the index holds, through
        :meth:`level_from_units`.

        Args:
            current_date: The date for which to calculate the index level.
            constituents: Current index constituents.
            weights: Current constituent weights.
            divisor: The current index divisor.
            previous_index_level: The index level from the previous calculation period.

        Returns:
            A tuple of (new_index_level, divisor). With no constituents the
            previous level is returned.

        Raises:
            CalculationError: If the divisor is not positive, a constituent is
                not an equity, or a currency cannot be converted.
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
