# src/beacon/backtest/engine.py
"""
BacktestEngine — simulates portfolio execution against a target weight schedule.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from ..portfolio.base import CASH_TOLERANCE as PORTFOLIO_CASH_TOLERANCE

# TradeInstruction lives with the ledger that accepts it (BN-151). Imported
# here as well so `from beacon.backtest.engine import TradeInstruction` keeps
# working -- and because the engine is its main producer.
from ..portfolio.base import Holding, Portfolio, TradeInstruction
from .result import BacktestResult, Book, UnfilledOrder
from .rules import BacktestModifier

# Reused from the portfolio rather than redefined: the engine decides whether
# to size an order down, the portfolio decides whether to accept it, and the
# two must agree on where "affordable" ends or an order sized to the boundary
# would be rejected on arrival.
CASH_TOLERANCE = PORTFOLIO_CASH_TOLERANCE

# Below this notional a reduced order is not worth placing; the position would
# be noise and the cost would dominate it.
MIN_TRADE_VALUE = 0.01

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Simulates portfolio execution against a target weight schedule.

    The engine consumes target weights from an ``IndexResult`` or a
    custom weight dictionary, and simulates trading over a date range
    using prices from a ``DataFetcher``.

    Args:
        start_date: The start date of the backtest (YYYY-MM-DD).
        end_date: The end date of the backtest (YYYY-MM-DD).
        initial_capital: The starting capital for the backtest.
        data_provider: Data source for market prices.
        index_result: An IndexResult whose weight_snapshots provide
            the rebalance schedule and target weights. Mutually exclusive
            with *target_weights*.
        target_weights: Custom weight schedule as a mapping of
            ``pd.Timestamp -> Dict[str, float]``. Mutually exclusive with
            *index_result*.
        price_column: Column name to read from market data. Defaults to
            ``"CLOSE"``.
        transaction_cost_bps: Transaction cost in basis points applied to
            each trade's notional value. Defaults to 0 (no cost).
        modifiers: Optional hooks that can skip rebalances or adjust trades.
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float,
                 data_provider: DataFetcher,
                 index_result: IndexResult | None = None,
                 target_weights: dict[pd.Timestamp, dict[str, float]] | None = None,
                 price_column: str = "CLOSE",
                 currency: str = "USD",
                 transaction_cost_bps: float = 0.0,
                 modifiers: list[BacktestModifier] | None = None,
                 benchmark: IndexResult | pd.Series | None = None,
                 target_index: IndexResult | None = None):
        if index_result is not None and target_weights is not None:
            raise ValueError(
                "Provide either index_result or target_weights, not both."
            )
        if index_result is None and target_weights is None:
            raise ValueError(
                "One of index_result or target_weights must be provided."
            )

        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.initial_capital: float = initial_capital
        self.data_provider: DataFetcher = data_provider
        self.index_result: IndexResult | None = index_result

        # The comparators of record (decision 13). The engine trades on
        # neither; it stores them so the run states what it was measured
        # against, and every reader quotes the same numbers.
        self.benchmark: IndexResult | pd.Series | None = benchmark
        self.target_index: IndexResult | None = target_index
        self.price_column: str = price_column
        self.currency: str = currency.upper()

        # Listing currency per identifier, resolved lazily and once. Prices
        # are quoted where the company lists; a portfolio has one currency.
        self._currencies: dict[str, str] = {}
        self._rates: dict[tuple[str, str], pd.Series] = {}

        self.transaction_cost_bps: float = transaction_cost_bps
        self.modifiers: list[BacktestModifier] = modifiers or []

        # Normalise weight schedule to a dict
        if target_weights is not None:
            self._weight_schedule: dict[pd.Timestamp, dict[str, float]] = target_weights
        elif index_result is not None:
            self._weight_schedule = index_result.weight_snapshots
        else:
            raise ValueError(
                "One of index_result or target_weights must be provided."
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_price(self,
                     asset_id: str,
                     date: pd.Timestamp) -> float | None:
        """One closing price for *asset_id* on *date*, in the book's currency.

        The conversion is the point. Prices are stored as the company is
        quoted -- yen in Tokyo, sterling in London -- while a portfolio has a
        single currency, and `IndexCalculator` has always converted its market
        values before comparing them. Returning the raw close here made the
        engine value a 300 yen share as 300 dollars: every non-domestic weight
        was wrong by its exchange rate, and against the single-currency
        universe that existed until BN-128 the error was invisible because
        every rate was 1.0.
        """
        date_str = date.strftime("%Y-%m-%d")
        try:
            df = self.data_provider.fetch_market_data(asset_id, date_str, date_str)
            if df.empty or self.price_column not in df.columns:
                return None

            val = df[self.price_column].iloc[0]
            if not pd.notna(val):
                return None

            return float(val) * self._rate_for(asset_id, date)
        except Exception as e:
            logger.error(f"Error fetching price for {asset_id} on {date_str}: {e}")
        return None

    def _rate_for(self,
                  asset_id: str,
                  date: pd.Timestamp) -> float:
        """FX from an asset's listing currency into the book's, on *date*.

        The whole series is fetched once per **pair** and then indexed by
        date, which is what makes a per-day rate affordable: seven lookups for
        a global universe rather than one per holding per day.

        Using a single fixed rate instead would be worse than it sounds. A
        constant scale factor cancels out of the weight arithmetic entirely --
        the engine sizes a position by value, so `quantity x price x rate` is
        the target value whatever the rate is -- and the conversion would look
        correct while changing nothing. It is the *drift* in the rate that a
        foreign holding actually experiences, and that only appears if the
        rate moves.
        """
        currency = self._currency_of(asset_id)

        if currency is None or currency == self.currency:
            return 1.0

        pair = (currency, self.currency)

        if pair not in self._rates:
            series = self.data_provider.fetch_fx_rates(currency, self.currency)

            if series.empty:
                logger.warning("No %s/%s rate; %s is valued unconverted.",
                               currency, self.currency, asset_id)

            self._rates[pair] = series.sort_index()

        series = self._rates[pair]

        if series.empty:
            return 1.0

        # As of the date, carried forward: a holiday in one market is not a
        # reason to stop converting a position held in another.
        position = series.index.searchsorted(date, side="right") - 1

        if position < 0:
            return float(series.iloc[0])

        return float(series.iloc[position])

    def _currency_of(self,
                     asset_id: str) -> str | None:
        """The currency an identifier is quoted in, from reference data."""
        if asset_id in self._currencies:
            return self._currencies[asset_id]

        resolved: str | None = None

        try:
            frame = self.data_provider.fetch_reference_data(asset_id)

            if not frame.empty and "CURRENCY" in frame.columns:
                value = frame["CURRENCY"].iloc[0]

                if pd.notna(value):
                    resolved = str(value).upper()
        except Exception as error:
            logger.error("Could not resolve the currency of %s: %s",
                         asset_id, error)

        self._currencies[asset_id] = resolved or self.currency

        return self._currencies[asset_id]

    def _update_portfolio_prices(self,
                                 portfolio: Portfolio,
                                 date: pd.Timestamp) -> None:
        """Fetch prices for all holdings and push into the portfolio."""
        prices: dict[str, float] = {}
        for asset_id in portfolio.holdings:
            price = self._fetch_price(asset_id, date)
            if price is not None:
                prices[asset_id] = price

        # Dated, so the history row lands on the simulated day rather than
        # at wall-clock time -- an undated mark would make the recorded NAV
        # panel useless (flagged in BN-152, resolved here).
        portfolio.update_prices(prices, date)

    def _delisting_dates(self) -> dict[str, pd.Timestamp]:
        """When each holding stops being listed, or an empty mapping.

        Defensive about what comes back because the provider is an interface,
        not a class: a fetcher assembled by hand or stood in for by a double
        need not implement this, and a backtest over a universe where nothing
        is ever delisted should not require it to.
        """
        getter = getattr(self.data_provider, "delisting_dates", None)

        if getter is None:
            return {}

        try:
            dates = getter()
        except Exception as error:
            logger.warning("Could not resolve delistings: %s", error)

            return {}

        return dates if isinstance(dates, dict) else {}

    def _dispose_delisted(self,
                          portfolio: Portfolio,
                          date: pd.Timestamp,
                          delistings: dict[str, pd.Timestamp]) -> None:
        """Settle any holding whose listing has ended, into cash.

        Without this the position is held forever. `_fetch_price` returns None
        once the rows stop, so `_update_portfolio_prices` leaves the holding
        marked at its last close and `_sell_instruction` returns None rather
        than a trade -- the NAV keeps carrying a company that no longer
        exists, and its weight is never released to anything that does.

        Settled at the last price the portfolio saw, and **without** a
        transaction cost. That is the modelling decision, and it is
        deliberate: an acquisition pays cash to the holder and a failure pays
        nothing, but neither is a trade crossed in a market that is by then
        closed. Charging brokerage on it would invent a fee nobody was
        billed.

        Args:
            portfolio: Mutated in place.
            date: Today.
            delistings: identifier -> last listed date.
        """
        if not delistings:
            return

        for asset_id in list(portfolio.holdings):
            last_listed = delistings.get(asset_id)

            if last_listed is None or date <= last_listed:
                continue

            holding = portfolio.holdings[asset_id]
            price = holding.current_price

            if price is None or price <= 0 or holding.quantity <= 0:
                logger.warning(
                    "[%s] %s delisted with no usable last price; the holding "
                    "is dropped and its value written off.", date, asset_id)
                portfolio.holdings.pop(asset_id, None)
                continue

            portfolio.execute_sell(asset_id, holding.quantity, price,
                                   cost=0.0, date=date)

            logger.info("[%s] Settled %.4f of %s at %.4f after delisting.",
                        date, holding.quantity, asset_id, price)

    def _get_target_weights_for_date(self,
                                     date: pd.Timestamp) -> dict[str, float] | None:
        """Return target weights if *date* is a rebalance date, else None."""
        return self._weight_schedule.get(date)

    def _sell_instruction(self,
                          asset_id: str,
                          holding: Holding,
                          target_weights: dict[str, float],
                          current_value: float,
                          cost_rate: float,
                          date: pd.Timestamp) -> TradeInstruction | None:
        """Return a SELL instruction for *asset_id* if not in target or overweight.

        Returns None if the asset should not be sold (no price, in-target
        and not overweight, or below the sell-quantity threshold).
        """
        price = self._fetch_price(asset_id, date)
        if price is None:
            return None

        target_w = target_weights.get(asset_id, 0.0)
        if target_w == 0:
            notional = holding.quantity * price
            cost = notional * cost_rate
            return TradeInstruction(asset_id, "SELL", holding.quantity, price, cost)

        target_value = current_value * target_w
        current_asset_value = holding.quantity * price
        if current_asset_value <= target_value + 1e-6:
            return None

        excess_value = current_asset_value - target_value
        qty_to_sell = excess_value / price
        if qty_to_sell <= 1e-9:
            return None

        notional = qty_to_sell * price
        cost = notional * cost_rate
        return TradeInstruction(asset_id, "SELL", qty_to_sell, price, cost)

    def _buy_instruction(self,
                         asset_id: str,
                         target_w: float,
                         portfolio: Portfolio,
                         current_value: float,
                         cost_rate: float,
                         date: pd.Timestamp) -> TradeInstruction | None:
        """Return a BUY instruction for *asset_id* if new or underweight.

        Returns None if the asset should not be bought (non-positive target
        weight, no price, or below the buy-deficit threshold).
        """
        if target_w <= 0:
            return None

        price = self._fetch_price(asset_id, date)
        if price is None or price <= 0:
            return None

        target_value = current_value * target_w
        current_holding_value = 0.0
        if asset_id in portfolio.holdings:
            current_holding_value = portfolio.holdings[asset_id].quantity * price

        deficit = target_value - current_holding_value
        if deficit <= 1e-6:
            return None

        qty_to_buy = deficit / price
        notional = qty_to_buy * price
        cost = notional * cost_rate
        return TradeInstruction(asset_id, "BUY", qty_to_buy, price, cost)

    def _generate_trades(self,
                         portfolio: Portfolio,
                         target_weights: dict[str, float],
                         date: pd.Timestamp) -> list[TradeInstruction]:
        """Calculate trades needed to move *portfolio* to *target_weights*.

        Returns a list of :class:`TradeInstruction` objects ordered with
        sells first, then buys. Transaction costs are calculated from
        :attr:`transaction_cost_bps`.

        Args:
            portfolio: The current portfolio state.
            target_weights: Mapping of asset_id to target weight (0 ≤ w ≤ 1).
            date: The trade date (used for price look-ups).

        Returns:
            list of TradeInstruction: Sells followed by buys.
        """
        current_value = portfolio.get_total_value()
        if current_value <= 0:
            return []

        cost_rate = self.transaction_cost_bps / 10_000.0
        sells: list[TradeInstruction] = []
        buys: list[TradeInstruction] = []

        # --- Sells: assets not in target, or overweight ---
        for asset_id, holding in portfolio.holdings.items():
            instruction = self._sell_instruction(asset_id, holding, target_weights,
                                                 current_value, cost_rate, date)
            if instruction is not None:
                sells.append(instruction)

        # --- Buys: new or underweight ---
        for asset_id, target_w in target_weights.items():
            instruction = self._buy_instruction(asset_id, target_w, portfolio,
                                                current_value, cost_rate, date)
            if instruction is not None:
                buys.append(instruction)

        return sells + buys

    def _rebalance(self,
                   portfolio: Portfolio,
                   target_weights: dict[str, float],
                   date: pd.Timestamp) -> list[UnfilledOrder]:
        """Adjust *portfolio* to match *target_weights* using :meth:`_generate_trades`.

        Modifiers may veto the rebalance or adjust the trade list.

        Returns:
            list of UnfilledOrder: Buys that could not be filled in full.
            Empty when every leg executed.
        """
        current_value = portfolio.get_total_value()
        if current_value <= 0:
            logger.warning(f"[{date}] Portfolio value is {current_value:.2f}. Skipping rebalance.")
            return []

        # Check modifiers for skip
        for modifier in self.modifiers:
            if modifier.should_skip_rebalance(date, portfolio, target_weights):
                logger.info(f"[{date}] Rebalance skipped by {modifier.__class__.__name__}.")
                return []

        logger.info(f"[{date}] Rebalancing to target weights: {target_weights}")
        trades = self._generate_trades(portfolio, target_weights, date)

        # Let modifiers adjust the trade list
        for modifier in self.modifiers:
            trades = modifier.adjust_trades(trades, date, portfolio)

        unfilled: list[UnfilledOrder] = []

        for trade in trades:
            if trade.side == "SELL":
                portfolio.apply(trade, date)
                logger.debug(f"[{date}] Sold {trade.quantity:.4f} of {trade.asset_id}")
            elif trade.side == "BUY":
                shortfall = self._execute_buy(portfolio, trade, date)
                if shortfall is not None:
                    unfilled.append(shortfall)

        return unfilled

    def _execute_buy(self,
                     portfolio: Portfolio,
                     trade: TradeInstruction,
                     date: pd.Timestamp) -> UnfilledOrder | None:
        """Buy as much of *trade* as the available cash supports.

        A rebalance sells before it buys, so the final buy is *expected* to
        consume almost exactly the proceeds. Comparing cash against the
        required amount exactly therefore fails routinely on sub-cent floating
        point noise, and dropping the whole order when cash falls a little
        short distorts the simulation far more than a slightly smaller
        position would: the freed weight silently accrues to whatever else is
        held.

        Args:
            portfolio: Portfolio to buy into. Mutated.
            trade: The requested buy.
            date: Trade date.

        Returns:
            UnfilledOrder or None: A record when the order could not be filled
            in full, otherwise None.
        """
        required = trade.quantity * trade.price + trade.cost

        # Affordable, allowing for accumulated float error on the cash balance.
        if portfolio.cash_balance >= required * (1 - CASH_TOLERANCE):
            portfolio.apply(trade, date)
            logger.debug(f"[{date}] Bought {trade.quantity:.4f} of {trade.asset_id}")
            return None

        # Size down instead of abandoning the leg. Cash has to cover the
        # notional *and* the cost charged on it, so the affordable quantity
        # solves cash = q * price * (1 + cost_rate).
        cost_rate = self.transaction_cost_bps / 10_000.0
        affordable = min(portfolio.cash_balance / (trade.price * (1 + cost_rate)),
                         trade.quantity)

        if affordable * trade.price < MIN_TRADE_VALUE:
            logger.warning(
                f"[{date}] Cannot buy {trade.asset_id}: need "
                f"{required:.2f}, have {portfolio.cash_balance:.2f}, and the "
                f"affordable quantity is below the minimum trade value.")
            return UnfilledOrder(date=date,
                                 asset_id=trade.asset_id,
                                 requested_quantity=trade.quantity,
                                 filled_quantity=0.0,
                                 price=trade.price,
                                 shortfall_value=trade.quantity * trade.price)

        available = portfolio.cash_balance
        reduced_cost = affordable * trade.price * cost_rate

        # The sized-down leg is a new decision, so it is a new instruction:
        # the engine decides, the portfolio accounts for what it is handed.
        portfolio.apply(TradeInstruction(asset_id=trade.asset_id,
                                         side="BUY",
                                         quantity=affordable,
                                         price=trade.price,
                                         cost=reduced_cost), date)
        logger.warning(
            f"[{date}] Partially filled {trade.asset_id}: bought "
            f"{affordable:.4f} of {trade.quantity:.4f} requested "
            f"(needed {required:.2f}, had {available:.2f}).")

        return UnfilledOrder(date=date,
                             asset_id=trade.asset_id,
                             requested_quantity=trade.quantity,
                             filled_quantity=affordable,
                             price=trade.price,
                             shortfall_value=(trade.quantity - affordable) * trade.price)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    #todo: vectorise run for efficiency, currently iterative and may be slow
    def run(self) -> BacktestResult:
        """Execute the backtest and return a :class:`BacktestResult`.

        Returns:
            BacktestResult
        """
        logger.info(
            f"Starting backtest from {self.start_date.date()} to "
            f"{self.end_date.date()} with capital {self.initial_capital:.2f}"
        )

        trading_days = pd.bdate_range(start=self.start_date, end=self.end_date, freq="B")
        if trading_days.empty:
            logger.warning("No trading days in the specified date range.")
            portfolio = Portfolio(portfolio_id="backtest_portfolio",
                                  initial_cash=self.initial_capital,
                                  inception=self.start_date,
                                  source=self.data_provider)
            portfolio.freeze()
            return self._build_result(portfolio, [])

        # Day zero is the EVE of the first trading day, not the start date:
        # the start date is usually itself a trading day, and history keeps
        # the last write per date -- an inception row dated the first trading
        # day would be overwritten by that day's close mark, and the record
        # of what the run started with would be gone (decision 11).
        eve = trading_days[0] - pd.tseries.offsets.BDay(1)
        portfolio = Portfolio(portfolio_id="backtest_portfolio",
                              initial_cash=self.initial_capital,
                              inception=eve,
                              source=self.data_provider)

        unfilled: list[UnfilledOrder] = []

        delistings = self._delisting_dates()

        for idx, date in enumerate(trading_days):
            # 1. Update prices for existing holdings
            self._update_portfolio_prices(portfolio, date)

            # 1b. Settle anything that stopped being listed. This has to
            # happen before the rebalance, because a delisted holding cannot
            # be sold by the ordinary path -- that path needs a price, and
            # there is not one.
            self._dispose_delisted(portfolio, date, delistings)

            # 2. Check for rebalance
            target_w = self._get_target_weights_for_date(date)
            if target_w is not None:
                unfilled.extend(self._rebalance(portfolio, target_w, date))
                # Re-price after rebalance
                self._update_portfolio_prices(portfolio, date)

            # 3. End-of-day state is already in the books: the dated mark
            # in step 1 (and the re-mark after a rebalance) wrote the day's
            # position, cash and NAV rows. Nothing to flatten here.
            nav = portfolio.get_total_value()

            # Progress logging
            n = len(trading_days)
            if n > 10 and idx % (n // 10) == 0:
                logger.info(
                    f"Backtest progress: {(idx + 1) / n * 100:.0f}% "
                    f"({date.date()}, NAV={nav:.2f})"
                )

        logger.info(f"Backtest finished. Final NAV: {portfolio.get_total_value():.2f}")

        # The run is over, so its books are closed: the portfolio is now the
        # record of this backtest, and a later write would restate it.
        portfolio.freeze()

        return self._build_result(portfolio, unfilled)

    def _build_result(self,
                      portfolio: Portfolio,
                      unfilled: list[UnfilledOrder]) -> BacktestResult:
        """Assemble the result: the portfolio kept whole, plus the books.

        Nothing is flattened -- the portfolio recorded its own history as the
        run marked and traded, and the comparators become books so every one
        answers through the same spelling (decision 5).
        """
        # Bound to the run's own data (decision 16): asset-level views on
        # this result always read what the simulation read, however the
        # process-level source moves later.
        return BacktestResult(
            portfolio=portfolio,
            index=(Book.from_index(self.index_result)
                   if self.index_result is not None else None),
            target_index=(Book.from_index(self.target_index)
                          if self.target_index is not None else None),
            benchmark=self._benchmark_book(),
            unfilled=unfilled,
        ).with_data(self.data_provider)

    def _benchmark_book(self) -> "Book | None":
        """The benchmark of record, whichever form it was given in."""
        if self.benchmark is None:
            return None

        if isinstance(self.benchmark, pd.Series):
            return Book.from_levels(self.benchmark)

        return Book.from_index(self.benchmark)
