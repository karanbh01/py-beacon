# src/beacon/portfolio/base.py
"""
Portfolio accounting: `Portfolio`, the `Holding`s it keeps, the
`Transaction`s it records, and the `TradeInstruction`s it accepts.
"""
import logging
from dataclasses import dataclass

import pandas as pd

from ..data.fetcher import DataFetcher
from ..exceptions import FrozenPortfolioError
from ..sources import resolve
from .asset_view import PortfolioAssetView
from .history import PortfolioHistory

logger = logging.getLogger(__name__)

# Relative slack on the cash check in execute_buy. Sized to absorb float error
# accumulated over a sequence of trades, while staying far below any amount a
# caller would consider money.
CASH_TOLERANCE = 1e-9

@dataclass(frozen=True)
class Transaction:
    """
    Represents a single transaction (buy or sell) of an asset.
    """
    asset_id: str
    quantity: float
    price: float
    transaction_type: str  # 'BUY' or 'SELL'
    transaction_date: pd.Timestamp
    transaction_cost: float = 0.0

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty.")
        if self.quantity <= 0:
            raise ValueError("Transaction quantity must be positive.")
        if self.price < 0:
            raise ValueError("Transaction price cannot be negative.")
        if self.transaction_type.upper() not in ['BUY', 'SELL']:
            raise ValueError("Transaction type must be 'BUY' or 'SELL'.")
        if not isinstance(self.transaction_date, pd.Timestamp):
            try:
                object.__setattr__(self, 'transaction_date', pd.Timestamp(self.transaction_date))
            except Exception as e:
                raise TypeError(
                    f"transaction_date must be a pandas Timestamp. Error: {e}") from e


# TradeInstruction lives here rather than in the backtest layer because the
# portfolio is the layer that accepts one, and a ledger's input type belongs
# with the ledger (BN-151; previously in ``backtest/engine.py``, where it
# forced the codebase's one circular-import workaround).
@dataclass(frozen=True)
class TradeInstruction:
    """A single trade for a portfolio to record.

    Produced by whatever *decides* trades (the backtest engine sizes, prices
    and costs an order) and consumed by :meth:`Portfolio.apply`, which does
    the accounting.

    Attributes:
        asset_id: Asset identifier.
        side: ``"SELL"`` or ``"BUY"``.
        quantity: Number of units to trade.
        price: Execution price per unit.
        cost: Transaction cost in currency terms.
    """
    asset_id: str
    side: str      # "SELL" or "BUY"
    quantity: float
    price: float
    cost: float


@dataclass
class Holding:
    """
    Represents a holding of a specific asset in the portfolio.
    Mutable as quantity and market value change.
    """
    asset_id: str
    quantity: float
    average_cost_price: float
    current_price: float | None = None
    market_value: float | None = None

    def __post_init__(self) -> None:
        if not self.asset_id:
            raise ValueError("asset_id cannot be empty.")
        if self.quantity < 0:
            raise ValueError("Holding quantity cannot be negative.")
        if self.average_cost_price < 0:
            raise ValueError("Holding average_cost_price cannot be negative.")

    def update_market_data(self,
                           current_price: float,
                           update_date: pd.Timestamp) -> None:
        """Updates the holding with the latest market price and recalculates market value."""
        if current_price < 0:
            logger.warning(
                f"Attempted to update holding for {self.asset_id} with negative "
                f"price: {current_price}. Price not updated.")
            return
        self.current_price = current_price
        self.market_value = self.quantity * self.current_price


# The store-of-record role is BN-152.
class Portfolio:
    """
    Manages a collection of asset holdings, cash balance, and transaction history.

    Holdings are keyed by string asset identifiers. The accounting needs no
    Asset objects or data source: callers pass simple strings and prices.

    It is also the **store of record**: what it started with, and what the
    books said on every date something changed them, are kept here rather
    than flattened onto whatever ran it. See :attr:`positions`, :attr:`cash`
    and :attr:`nav`.

    Args:
        portfolio_id: Identifier for these books.
        initial_cash: Opening cash balance. Retained as
            :attr:`initial_capital`, because `cash_balance` goes on mutating
            and without it a portfolio could not say what it started with.
        inception: Optional day zero. When given, the books open on that date
            with NAV and cash equal to the initial capital, before anything
            trades. A backtest passes the business day before its first
            trading day; a hand-built portfolio can leave it out, and its
            history then starts at its first event.
        source: Data source that :meth:`asset` reads market data from. None
            uses the process's ambient source (:func:`beacon.sources.resolve`)
            at the time of the call.

    Raises:
        ValueError: If *portfolio_id* is empty or *initial_cash* is negative.
    """
    def __init__(self,
                 portfolio_id: str,
                 initial_cash: float = 0.0,
                 inception: pd.Timestamp | None = None,
                 source: "DataFetcher | None" = None):
        if not portfolio_id:
            raise ValueError("portfolio_id cannot be empty.")
        if initial_cash < 0:
            raise ValueError("Initial cash cannot be negative.")

        self.portfolio_id: str = portfolio_id
        self.holdings: dict[str, Holding] = {}
        self.cash_balance: float = initial_cash
        self.transactions: list[Transaction] = []

        self.initial_capital: float = initial_cash

        # Where this portfolio's market questions are answered. Bound wins
        # over the process-level fallback (decision 16): the engine binds its
        # data provider here, so a backtest result's views always read the
        # data the run used.
        self.source: DataFetcher | None = source
        self.inception: pd.Timestamp | None = (
            pd.Timestamp(inception) if inception is not None else None)
        self.frozen: bool = False

        self._history = PortfolioHistory()

        if self.inception is not None:
            self._history.record(self.inception, self.holdings, self.cash_balance)

        logger.info(
            f"Portfolio '{self.portfolio_id}' initialized with cash: {self.cash_balance:.2f}")

    # ------------------------------------------------------------------
    # The books over time
    # ------------------------------------------------------------------

    @property
    def positions(self) -> pd.DataFrame:
        """What was held, long-form: DATE, ASSET_ID, QUANTITY, PRICE,
        MARKET_VALUE, WEIGHT.

        A row per held asset per date on which the books changed (a trade or
        a mark). `WEIGHT` was computed and stored at write time, so it records
        what the portfolio believed then rather than what recomputing it now
        would say.
        """
        return self._history.positions

    @property
    def weights(self) -> pd.DataFrame:
        """The stored weights, wide: dates by asset.

        A pivot of the positions panel's `WEIGHT` column: the same recorded
        numbers, shaped for cross-book arithmetic. `portfolio.weights`
        subtracts cleanly against an index book's weights because both are
        date-by-asset frames.
        """
        positions = self.positions

        if positions.empty:
            return pd.DataFrame()

        return positions.pivot_table(index="DATE", columns="ASSET_ID",
                                     values="WEIGHT", observed=True)

    @property
    def cash(self) -> pd.Series:
        """The cash balance on every recorded date."""
        return self._history.cash

    @property
    def nav(self) -> pd.Series:
        """Total value on every recorded date.

        Starts at :attr:`initial_capital` on the inception date when one was
        given.
        """
        return self._history.nav

    def asset(self,
              asset_id: str) -> "PortfolioAssetView":
        """One asset's position and market data, read live.

        Args:
            asset_id: An asset this portfolio holds or has ever held.

        Returns:
            PortfolioAssetView: Position numbers from these books, market
            facts from the resolved data source.

        Raises:
            KeyError: If the books have never seen *asset_id*, matching
                `IndexResult.asset` for a non-constituent.
            DataSourceError: If no source is bound and the process has none.
        """
        known = set(self.holdings)
        positions = self.positions

        if not positions.empty:
            known |= set(positions["ASSET_ID"])

        if asset_id not in known:
            raise KeyError(
                f"Asset '{asset_id}' does not appear in this portfolio's "
                f"books.")

        fetcher = self.source if self.source is not None else resolve()

        return PortfolioAssetView(asset_id, fetcher, self)

    def freeze(self) -> None:
        """Close the books: from here on, any write raises.

        Called by the backtest engine when a run ends. The portfolio is then
        the record of that run, and a later trade against it would restate a
        result somebody may already have read. Idempotent.
        """
        if not self.frozen:
            logger.info(f"Portfolio '{self.portfolio_id}' frozen.")

        self.frozen = True

    def _refuse_if_frozen(self,
                          operation: str) -> None:
        """Guard every write path.

        Args:
            operation: The method that was called, for the message.

        Raises:
            FrozenPortfolioError: If :meth:`freeze` has been called.
        """
        if self.frozen:
            raise FrozenPortfolioError(self.portfolio_id, operation)

    def _record(self,
                date: pd.Timestamp) -> None:
        """Snapshot the books as of *date* into the history."""
        self._history.record(date, self.holdings, self.cash_balance)


    def execute_buy(self,
                    asset_id: str,
                    quantity: float,
                    price: float,
                    cost: float = 0.0,
                    date: pd.Timestamp | None = None) -> None:
        """Buy an asset: deduct cash, create/update holding, record transaction.

        The holding's average cost is updated as a quantity-weighted average
        and it is marked at *price*. If cash (allowing a tiny tolerance for
        float error) does not cover ``quantity * price + cost``, the buy is
        not made: an error is logged and nothing changes.

        Args:
            asset_id: String identifier for the asset.
            quantity: Number of units to buy (must be positive).
            price: Execution price per unit.
            cost: Optional transaction cost (brokerage, taxes, etc.).
            date: Optional execution date. Defaults to now, and the history
                row is written under that same timestamp.

        Raises:
            ValueError: If *quantity* is not positive or *price* is negative.
            FrozenPortfolioError: If the books have been frozen.
        """
        self._refuse_if_frozen("execute_buy")

        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if price < 0:
            raise ValueError("price cannot be negative.")

        trade_value = quantity * price
        required = trade_value + cost

        # Compared with tolerance, not exactly. A caller spending down a
        # balance — the backtest engine selling and then reinvesting the
        # proceeds is the normal case — arrives here with a required amount
        # that differs from the balance only by accumulated float error, and an
        # exact comparison rejects a purchase the portfolio can plainly afford.
        if self.cash_balance < required * (1 - CASH_TOLERANCE):
            logger.error(
                f"Insufficient cash for BUY of {asset_id}. "
                f"Required: {required:.2f}, Available: {self.cash_balance:.2f}"
            )
            return

        tx_date = date if date is not None else pd.Timestamp.now()
        # Clamped at zero so a purchase accepted inside the tolerance cannot
        # leave a residual negative balance.
        self.cash_balance = max(self.cash_balance - required, 0.0)

        if asset_id in self.holdings:
            h = self.holdings[asset_id]
            old_total = h.average_cost_price * h.quantity
            h.quantity += quantity
            if h.quantity > 1e-9:
                h.average_cost_price = (old_total + trade_value) / h.quantity
            else:
                h.average_cost_price = price
        else:
            self.holdings[asset_id] = Holding(
                asset_id=asset_id, quantity=quantity, average_cost_price=price
            )

        logger.debug(f"BUY: {quantity} of {asset_id} @ {price:.2f}. Cash: {self.cash_balance:.2f}")

        self.transactions.append(
            Transaction(asset_id, quantity, price, 'BUY', tx_date, cost)
        )

        # Update market data using execution price
        self.holdings[asset_id].update_market_data(price, tx_date)

        self._record(tx_date)

    def execute_sell(self,
                     asset_id: str,
                     quantity: float,
                     price: float,
                     cost: float = 0.0,
                     date: pd.Timestamp | None = None) -> None:
        """Sell an asset: add cash proceeds, reduce/remove holding, record transaction.

        Cash rises by ``quantity * price - cost``. A holding reduced to
        (almost) zero is removed; what remains of a partly sold holding is
        re-marked at *price*. If the portfolio holds less than *quantity*,
        the sell is not made: an error is logged and nothing changes.

        Args:
            asset_id: String identifier for the asset.
            quantity: Number of units to sell (must be positive).
            price: Execution price per unit.
            cost: Optional transaction cost (brokerage, taxes, etc.).
            date: Optional execution date. Defaults to now, and the history
                row is written under that same timestamp.

        Raises:
            ValueError: If *quantity* is not positive or *price* is negative.
            FrozenPortfolioError: If the books have been frozen.
        """
        self._refuse_if_frozen("execute_sell")

        if quantity <= 0:
            raise ValueError("quantity must be positive.")
        if price < 0:
            raise ValueError("price cannot be negative.")

        if asset_id not in self.holdings or self.holdings[asset_id].quantity < quantity:
            current_qty = self.holdings[asset_id].quantity if asset_id in self.holdings else 0
            logger.error(
                f"Insufficient holdings for SELL of {asset_id}. "
                f"Attempting to sell: {quantity}, Available: {current_qty}"
            )
            return

        tx_date = date if date is not None else pd.Timestamp.now()
        trade_value = quantity * price
        self.cash_balance += (trade_value - cost)

        self.holdings[asset_id].quantity -= quantity
        logger.debug(f"SELL: {quantity} of {asset_id} @ {price:.2f}. Cash: {self.cash_balance:.2f}")

        if self.holdings[asset_id].quantity < 1e-9:
            logger.debug(f"Fully sold asset: {asset_id}. Removing from holdings.")
            del self.holdings[asset_id]
        else:
            # Re-mark what is left, at the price it just traded at. Without
            # this the remaining holding keeps the market value of the
            # *larger* position it used to be, so `get_total_value` overstates
            # the books until the next mark — and the position row written
            # below would record a quantity and a market value that disagree.
            self.holdings[asset_id].update_market_data(price, tx_date)

        self.transactions.append(
            Transaction(asset_id, quantity, price, 'SELL', tx_date, cost)
        )

        self._record(tx_date)


    def apply(self,
              trade: TradeInstruction,
              date: pd.Timestamp | None = None) -> None:
        """Record one trade in the books.

        The entry point for anything that has already *decided* a trade, such
        as the engine after sizing and pricing it. Dispatches to
        :meth:`execute_buy` or :meth:`execute_sell`, so the accounting stays
        on the portfolio: weighted average cost, closing at ~zero quantity,
        and declining (with a logged error) entries that would push cash or
        holdings negative.

        Args:
            trade: The instruction, as the decider issued it.
            date: Execution date. Defaults to now, as the underlying
                accounting does.

        Raises:
            ValueError: If the side is neither ``"BUY"`` nor ``"SELL"``
                (case-insensitive). Refused rather than guessed, because
                silently ignoring an unknown side would drop a trade from the
                record.
            FrozenPortfolioError: If the books have been frozen. Checked here
                as well as in the accounting, so the message names the entry
                point the caller actually used.
        """
        self._refuse_if_frozen("apply")

        side = trade.side.upper()

        if side == "BUY":
            self.execute_buy(trade.asset_id, trade.quantity, trade.price,
                             cost=trade.cost, date=date)
        elif side == "SELL":
            self.execute_sell(trade.asset_id, trade.quantity, trade.price,
                              cost=trade.cost, date=date)
        else:
            raise ValueError(
                f"Unknown trade side '{trade.side}'. Expected BUY or SELL.")

    def update_prices(self,
                      prices: dict[str, float],
                      date: pd.Timestamp | None = None) -> None:
        """
        Update current prices for holdings from a dictionary.

        A mark changes what the books say the portfolio is worth, so it is
        recorded like a trade is.

        Args:
            prices: Mapping of asset_id to current price.
                    Holdings whose asset_id is not in the dict are left
                    unchanged with a warning.
            date: Optional date to mark as of. Defaults to now.

        Raises:
            FrozenPortfolioError: If the books have been frozen.
        """
        self._refuse_if_frozen("update_prices")

        as_of = date if date is not None else pd.Timestamp.now()

        for asset_id, holding in self.holdings.items():
            price = prices.get(asset_id)
            if price is not None:
                holding.update_market_data(price, as_of)
            else:
                logger.warning(
                    f"No price supplied for {asset_id}. "
                    "Market value may be stale."
                )

        self._record(as_of)

    def apply_ratio(self,
                    asset_id: str,
                    ratio: float) -> None:
        """Change a holding's share count for a split or stock dividend.

        The quantity is multiplied by *ratio* and the per-share average cost
        and price divided by it, so the holding's cost and value are unchanged.
        Nothing is traded and no transaction is recorded. A name not held is
        left alone.

        Args:
            asset_id: The holding.
            ratio: The share-count multiplier: 2.0 for a 2-for-1 split, 0.5
                for a 1-for-2 reverse split.

        Raises:
            ValueError: If *ratio* is not positive.
            FrozenPortfolioError: If the books have been frozen.
        """
        self._refuse_if_frozen("apply_ratio")

        if ratio <= 0:
            raise ValueError(f"A share-count ratio must be positive, got {ratio}.")

        holding = self.holdings.get(asset_id)

        if holding is None:
            return

        holding.quantity *= ratio
        holding.average_cost_price /= ratio

        if holding.current_price is not None:
            holding.current_price /= ratio

    def get_total_value(self) -> float:
        """
        Calculates the total current market value of the portfolio (holdings + cash).

        Relies on prices having been set via :meth:`update_prices`,
        :meth:`execute_buy`, or :meth:`execute_sell` beforehand.

        Returns:
            The total portfolio value as a float.
        """
        total_holdings_value = 0.0
        for asset_id, holding in self.holdings.items():
            if holding.market_value is not None:
                total_holdings_value += holding.market_value
            else:
                logger.warning(
                    f"Market value for {asset_id} is None. "
                    "It will not be included in total portfolio value calculation "
                    "based on market prices.")

        total_portfolio_value = total_holdings_value + self.cash_balance
        return total_portfolio_value

    def get_weights(self) -> dict[str, float]:
        """
        Calculates the current weight of each asset in the portfolio.
        Weights are based on last-updated market values.

        Returns:
            A dictionary mapping asset_id strings to weight floats.
        """
        total_value = self.get_total_value()
        weights: dict[str, float] = {}

        if total_value == 0:
            logger.warning(
                f"Total portfolio value is 0. Cannot calculate asset weights for "
                f"portfolio '{self.portfolio_id}'.")
            for asset_id in self.holdings:
                weights[asset_id] = 0.0
            return weights

        for asset_id, holding in self.holdings.items():
            if holding.market_value is not None:
                weights[asset_id] = holding.market_value / total_value
            else:
                weights[asset_id] = 0.0
                logger.warning(f"Weight for {asset_id} is 0 due to missing market value.")

        return weights

    def get_holdings_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing current holdings.

        Returns:
            A pandas DataFrame with columns: AssetID, Quantity,
            AvgCostPrice, CurrentPrice, MarketValue, Weight.
        """
        portfolio_total_value = self.get_total_value()

        summary_data = []
        for asset_id, holding in self.holdings.items():
            weight = ((holding.market_value / portfolio_total_value)
                      if portfolio_total_value != 0 and holding.market_value is not None
                      else 0.0)
            summary_data.append({
                'AssetID': asset_id,
                'Quantity': holding.quantity,
                'AvgCostPrice': holding.average_cost_price,
                'CurrentPrice': holding.current_price,
                'MarketValue': holding.market_value,
                'Weight': weight
            })

        # Add cash row
        summary_data.append({
            'AssetID': 'CASH',
            'Quantity': 1.0,
            'AvgCostPrice': self.cash_balance,
            'CurrentPrice': self.cash_balance,
            'MarketValue': self.cash_balance,
            'Weight': ((self.cash_balance / portfolio_total_value)
                       if portfolio_total_value != 0
                       else (1.0 if self.cash_balance > 0 else 0.0))
        })

        return pd.DataFrame(summary_data)


    def __repr__(self) -> str:
        return (f"Portfolio(portfolio_id='{self.portfolio_id}', "
                f"num_holdings={len(self.holdings)}, cash_balance={self.cash_balance:.2f})")
