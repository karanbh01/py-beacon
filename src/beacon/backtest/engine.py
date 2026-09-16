# src/beacon/backtest/engine.py
"""
BacktestEngine — simulates portfolio execution against a target weight schedule.
"""
import logging

import pandas as pd

from ..data.fetcher import DataFetcher
from ..index.result import IndexResult
from ..index.schedule import SESSION_UNIT, sessions
from ..portfolio.base import CASH_TOLERANCE as PORTFOLIO_CASH_TOLERANCE

# TradeInstruction lives with the ledger that accepts it (BN-151). Imported
# here as well so `from beacon.backtest.engine import TradeInstruction` keeps
# working -- and because the engine is its main producer.
from ..portfolio.base import Holding, Portfolio, TradeInstruction
from .pricing import PricingMixin
from .result import (
    BacktestResult,
    Book,
    IndexBooks,
    PriceGap,
    RebalancePricing,
    UnfilledOrder,
)
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


class BacktestEngine(PricingMixin):
    """Simulates portfolio execution against a target weight schedule.

    The engine consumes target weights from an ``IndexResult`` — the sole
    schedule source since BN-165, when the raw weight-dict mode was removed —
    and simulates trading over a date range using prices from a
    ``DataFetcher``.

    Args:
        start_date: The start date of the backtest (YYYY-MM-DD).
        end_date: The end date of the backtest (YYYY-MM-DD).
        initial_capital: The starting capital for the backtest.
        data_provider: Data source for market prices.
        index_result: The IndexResult whose weight_snapshots provide
            the rebalance schedule and target weights.
        price_column: Column name to read from market data. Defaults to
            ``"CLOSE"``.
        transaction_cost_bps: Transaction cost in basis points applied to
            each trade's notional value. Defaults to 0 (no cost).
        modifiers: Optional hooks that can skip rebalances or adjust trades.
        benchmark: The benchmark of record, stored on the result so every
            reader quotes excess return against the same comparator.
        target_index: The calculated index the traded schedule was derived
            from, when it differs from the schedule itself — the
            derived-index shape (BN-167): *index_result* is an optimised
            calculation and this is its parent, and they land in
            `index.optimised` and `index.target` respectively. Omitted on a
            plain run, whose own calculation fills the target book.
        calendar: The exchange MIC the traded index schedules on, which since
            BN-180 every definition carries. It decides which days the run
            steps onto at all (BN-186) and how a missing bar on one of them is
            read (BN-183): on a day the calendar says was closed the
            market was shut and the previous session's price is what the
            position was worth, while on a day it says was open the data is
            missing something and the carried price is recorded as a gap.
            None falls back to the data's own sessions — a day the store has
            bars for is treated as open — which is all a caller assembling an
            engine by hand can offer.
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float,
                 data_provider: DataFetcher,
                 index_result: IndexResult,
                 price_column: str = "CLOSE",
                 currency: str = "USD",
                 transaction_cost_bps: float = 0.0,
                 modifiers: list[BacktestModifier] | None = None,
                 benchmark: IndexResult | pd.Series | None = None,
                 target_index: IndexResult | None = None,
                 calendar: str | None = None):
        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.initial_capital: float = initial_capital
        self.data_provider: DataFetcher = data_provider
        self.index_result: IndexResult = index_result

        # The comparators of record (decision 13). The engine trades on
        # neither; it stores them so the run states what it was measured
        # against, and every reader quotes the same numbers.
        self.benchmark: IndexResult | pd.Series | None = benchmark
        self.target_index: IndexResult | None = target_index
        self.price_column: str = price_column
        self.currency: str = currency.upper()
        self.calendar: str | None = calendar

        # Listing currency per identifier, resolved lazily and once. Prices
        # are quoted where the company lists; a portfolio has one currency.
        self._currencies: dict[str, str] = {}
        self._rates: dict[tuple[str, str], pd.Series] = {}

        # The last bar each name actually printed, so a miss is answered from
        # the session before it rather than by refetching a whole history.
        self._last_bars: dict[str, tuple[pd.Timestamp, float]] = {}

        # What the run has to report about its own pricing (BN-183). The set
        # is the dedupe: a rebalance day prices each name several times --
        # the mark, the sell test, the buy test, the re-mark -- and one
        # missing bar is one gap however many readers met it.
        self._price_gaps: list[PriceGap] = []
        self._gaps_seen: set[tuple[str, pd.Timestamp]] = set()
        self._rebalance_pricing: list[RebalancePricing] = []

        # Filled by `run`. A name past its last listed date has no price
        # because it no longer exists, which is neither a holiday nor a gap.
        self._delistings: dict[str, pd.Timestamp] = {}

        self.transaction_cost_bps: float = transaction_cost_bps
        self.modifiers: list[BacktestModifier] = modifiers or []

        # The internal schedule representation: rebalance date -> weights.
        self._weight_schedule: dict[pd.Timestamp, dict[str, float]] = (
            index_result.weight_snapshots)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

        Defensive about what the provider *offers*, not about whether it
        works. The provider is an interface rather than a class, so a fetcher
        assembled by hand or stood in for by a double need not implement this,
        and a backtest over a universe where nothing is ever delisted should
        not require it to — hence the `getattr`, and hence a non-mapping answer
        being read as "nothing leaves".

        A failure is a different thing, and used to be swallowed into the same
        empty mapping with a WARNING (BN-197). Empty does not mean "unknown"
        here, it means "nothing is ever delisted", and the engine acts on it:
        the price read stops declining to carry a dead name forward, and
        disposal never settles the holding. So the book ran to the end holding
        names that no longer existed, each marked at its last close, and
        published a NAV as though that were real — with the only record of it
        in a log nobody reads after the fact.

        `IndexCalculator.delisting_schedule` calls the same method bare and
        always has. Two surfaces over one call answering differently is the
        BN-174 shape, and the one that substitutes is the one that was wrong.
        """
        getter = getattr(self.data_provider, "delisting_dates", None)

        if getter is None:
            return {}

        dates = getter()

        return dates if isinstance(dates, dict) else {}

    def _dispose_delisted(self,
                          portfolio: Portfolio,
                          date: pd.Timestamp,
                          delistings: dict[str, pd.Timestamp]) -> None:
        """Settle any holding whose listing has ended, into cash.

        Without this the position is held forever. A name past its last listed
        date has no price, so `_update_portfolio_prices` leaves the holding
        marked at its last close and `_sell_instruction` returns None rather
        than a trade -- the NAV keeps carrying a company that no longer
        exists, and its weight is never released to anything that does.

        The signal is *this mapping*, read from reference data's `DATE_TO`,
        and it always was: nothing here infers a delisting from prices running
        out, which is why BN-183 could change the price read without touching
        disposal. `_fetch_price` consults the same mapping and declines to
        carry a price forward past it, so the two agree on when a name stopped
        existing rather than one of them guessing from an absence.

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

    def _scheduled_days(self) -> pd.DatetimeIndex:
        """Rebalance dates inside the run window that are not sessions.

        The engine steps onto its calendar's sessions (BN-186), and a schedule
        naming a day that is not one still has to be executed. An index
        calculated by Beacon can no longer produce such a date — its schedule
        and the loop read the same calendar — so this covers the schedule
        handed in from outside: a weight dict written by hand, or one built on
        a different venue's calendar from the one the book is priced against.

        Dropping the date would put back exactly the defect BN-183 fixed: a
        rebalance in the record on which nothing moved. Executing it on the
        day it names, at the price of the session in force, is what BN-183
        already built the price read to do.
        """
        within = [date for date in self._weight_schedule
                  if self.start_date <= date <= self.end_date]

        return pd.DatetimeIndex(sorted(within)).as_unit(SESSION_UNIT)

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

        # Recorded before the trades, and only for a rebalance that goes ahead:
        # a skipped one priced nothing, and a row saying otherwise would be a
        # session named for trades that never happened (BN-183).
        self._record_rebalance_pricing(date)

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

    def _record_rebalance_pricing(self,
                                  date: pd.Timestamp) -> None:
        """Note the session this rebalance's prices are read from (BN-183).

        One row per rebalance rather than one per leg: every name in it
        resolves to the same session, because the closure that moved the date
        closed the market for all of them. A run whose data cannot resolve a
        session at all — a hand-assembled provider — records the date itself,
        which is what it priced from.
        """
        session = self._resolved_session(date)

        self._rebalance_pricing.append(
            RebalancePricing(date=date,
                             priced_from=session if session is not None else date))

        if session is not None and session != date:
            logger.info("[%s] The market was shut; the rebalance prices from "
                        "the %s session.", date.date(), session.date())

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

        # What this run reports about its own pricing, cleared rather than
        # carried: a second `run()` on the same engine is a second run, and
        # inheriting the first one's gaps would report them twice.
        self._price_gaps.clear()
        self._gaps_seen.clear()
        self._rebalance_pricing.clear()

        # The traded index's own sessions (BN-186), from the calendar BN-183
        # already wired in for the price read -- one source of truth, so the
        # day the engine steps onto is the day it can price. Without a
        # calendar this is still Monday to Friday, which is all a caller
        # assembling an engine by hand has told it.
        trading_days = sessions(self.start_date, self.end_date,
                                self.calendar).union(self._scheduled_days())
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

        # Held on the engine as well as passed down: the price read consults
        # it to decline carrying a delisted name forward (BN-183), and
        # disposal reads it to settle the holding. One mapping, two readers.
        delistings = self._delisting_dates()
        self._delistings = delistings

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
            index=self._index_books(),
            benchmark=self._benchmark_book(),
            unfilled=unfilled,
            price_gaps=list(self._price_gaps),
            rebalance_pricing=list(self._rebalance_pricing),
        ).with_data(self.data_provider)

    def _index_books(self) -> IndexBooks:
        """The run's calculated indices, one book each (BN-164, BN-167).

        Given *index_result* alone, the run traded a plain calculation: it is
        the `index.target` book. Given *target_index* as well — the
        derived-index shape — the schedule the engine traded is an optimised
        calculation and *target_index* is the parent it was solved from, so
        they land in `index.optimised` and `index.target` respectively.
        """
        if self.target_index is not None:
            return IndexBooks(target=Book.from_index(self.target_index),
                              optimised=Book.from_index(self.index_result))

        return IndexBooks(target=Book.from_index(self.index_result))

    def _benchmark_book(self) -> "Book | None":
        """The benchmark of record, whichever form it was given in."""
        if self.benchmark is None:
            return None

        if isinstance(self.benchmark, pd.Series):
            return Book.from_levels(self.benchmark)

        return Book.from_index(self.benchmark)
