# beacon/portfolio/base.py
"""
Module defining classes for managing investment portfolios, including
Transaction, Holding, and the main Portfolio class.
"""
import pandas as pd
from typing import List, Dict, Optional, Union, NamedTuple, TYPE_CHECKING
from dataclasses import dataclass, field
import logging

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..asset.base import Asset

logger = logging.getLogger(__name__)

@dataclass(frozen=True) # Making Transaction immutable
class Transaction:
    """
    Represents a single transaction (buy or sell) of an asset.
    Using dataclass for simplicity and auto-generated methods.
    """
    asset: 'Asset'
    quantity: float # Number of shares/units. Positive for BUY, can be positive for SELL too (absolute quantity)
    price: float    # Price per share/unit at which transaction occurred
    transaction_type: str # 'BUY' or 'SELL'
    transaction_date: pd.Timestamp
    transaction_cost: float = 0.0 # Optional: brokerage fees, taxes, etc.

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError("Transaction quantity must be positive.")
        if self.price < 0: # Price can be 0 for some corporate actions if modeled as transactions
            raise ValueError("Transaction price cannot be negative.")
        if self.transaction_type.upper() not in ['BUY', 'SELL']:
            raise ValueError("Transaction type must be 'BUY' or 'SELL'.")
        if not isinstance(self.transaction_date, pd.Timestamp):
             # Try to convert if it's a string, otherwise raise error
            try:
                object.__setattr__(self, 'transaction_date', pd.Timestamp(self.transaction_date))
            except Exception as e:
                raise TypeError(f"transaction_date must be a pandas Timestamp. Error: {e}")


@dataclass
class Holding:
    """
    Represents a holding of a specific asset in the portfolio.
    Mutable as quantity and market value change.
    """
    asset: 'Asset'
    quantity: float
    average_cost_price: float # Average acquisition price per unit
    current_price: Optional[float] = None # Last known market price
    market_value: Optional[float] = None # quantity * current_price

    def __post_init__(self):
        if self.quantity < 0: # Allowing 0 quantity if an asset was fully sold
            raise ValueError("Holding quantity cannot be negative.")
        if self.average_cost_price < 0:
            raise ValueError("Holding average_cost_price cannot be negative.")

    def update_market_data(self, current_price: float, update_date: pd.Timestamp):
        """Updates the holding with the latest market price and recalculates market value."""
        if current_price < 0:
            logger.warning(f"Attempted to update holding for {self.asset.asset_id} with negative price: {current_price}. Price not updated.")
            return
        self.current_price = current_price
        self.market_value = self.quantity * self.current_price
        logger.debug(f"Holding for {self.asset.asset_id} updated: Qty={self.quantity}, Price={self.current_price}, MV={self.market_value}")


class Portfolio:
    """
    Manages a collection of asset holdings, cash balance, and transaction history.

    The Portfolio is decoupled from data fetching concerns. External callers
    (e.g. the backtest engine) are responsible for fetching prices and passing
    them via :meth:`update_prices`.
    """
    def __init__(self,
                 portfolio_id: str,
                 initial_cash: float = 0.0,
                ):
        """
        Initializes a Portfolio.

        Args:
            portfolio_id: A unique identifier for the portfolio.
            initial_cash: The starting cash balance of the portfolio.
        """
        if not portfolio_id:
            raise ValueError("portfolio_id cannot be empty.")
        if initial_cash < 0:
            raise ValueError("Initial cash cannot be negative.")

        self.portfolio_id: str = portfolio_id
        self.holdings: Dict['Asset', Holding] = {} # Maps Asset object to Holding object
        self.cash_balance: float = initial_cash
        self.transactions: List[Transaction] = []

        logger.info(f"Portfolio '{self.portfolio_id}' initialized with cash: {self.cash_balance:.2f}")


    def add_transaction(self, transaction: Transaction) -> None:
        """
        Adds a transaction to the portfolio and updates holdings and cash balance.

        The transaction's price is used as the current price for the affected
        holding immediately after the trade.

        Args:
            transaction: The Transaction object to add.
        """
        if not isinstance(transaction, Transaction):
            raise TypeError("transaction must be a Transaction object.")

        asset = transaction.asset
        qty = transaction.quantity # Always positive
        price = transaction.price
        tx_type = transaction.transaction_type.upper()
        tx_cost = transaction.transaction_cost

        trade_value = qty * price

        if tx_type == 'BUY':
            if self.cash_balance < (trade_value + tx_cost):
                logger.error(f"Insufficient cash for BUY transaction of {asset.asset_id}. "
                             f"Required: {(trade_value + tx_cost):.2f}, Available: {self.cash_balance:.2f}")
                return # Skip transaction

            self.cash_balance -= (trade_value + tx_cost)

            if asset in self.holdings:
                current_holding = self.holdings[asset]
                old_total_value = current_holding.average_cost_price * current_holding.quantity
                new_total_value = price * qty
                current_holding.quantity += qty
                if current_holding.quantity > 1e-9:
                    current_holding.average_cost_price = (old_total_value + new_total_value) / current_holding.quantity
                else:
                    current_holding.average_cost_price = price

            else: # New holding
                self.holdings[asset] = Holding(
                    asset=asset,
                    quantity=qty,
                    average_cost_price=price
                )
            logger.info(f"BUY: {qty} of {asset.asset_id} @ {price:.2f}. Cash: {self.cash_balance:.2f}")

        elif tx_type == 'SELL':
            if asset not in self.holdings or self.holdings[asset].quantity < qty:
                current_qty = self.holdings[asset].quantity if asset in self.holdings else 0
                logger.error(f"Insufficient holdings for SELL transaction of {asset.asset_id}. "
                             f"Attempting to sell: {qty}, Available: {current_qty}")
                return # Skip transaction

            self.cash_balance += (trade_value - tx_cost) # Proceeds minus cost

            self.holdings[asset].quantity -= qty
            logger.info(f"SELL: {qty} of {asset.asset_id} @ {price:.2f}. Cash: {self.cash_balance:.2f}")

            if self.holdings[asset].quantity < 1e-9:
                logger.info(f"Fully sold asset: {asset.asset_id}. Removing from holdings.")
                del self.holdings[asset]
        else:
            logger.error(f"Unknown transaction type: {tx_type}")
            return

        self.transactions.append(transaction)

        # Update market data for the affected holding using the transaction price
        if asset in self.holdings:
            self.holdings[asset].update_market_data(price, transaction.transaction_date)


    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update current prices for holdings from a dictionary.

        The backtest engine or external caller fetches prices and passes them
        here. This keeps the Portfolio free of DataFetcher coupling.

        Args:
            prices: Mapping of ``asset_id`` (or ticker) to current price.
                    Holdings whose asset_id is not in the dict are left
                    unchanged with a warning.
        """
        for asset, holding in self.holdings.items():
            price = prices.get(asset.asset_id) or prices.get(getattr(asset, 'ticker', ''))
            if price is not None:
                holding.update_market_data(price, pd.Timestamp.now())
            else:
                logger.warning(
                    f"No price supplied for {asset.asset_id}. "
                    "Market value may be stale."
                )

    def get_total_value(self) -> float:
        """
        Calculates the total current market value of the portfolio (holdings + cash).

        Relies on prices having been set via :meth:`update_prices` or
        :meth:`add_transaction` beforehand.

        Returns:
            The total portfolio value as a float.
        """
        total_holdings_value = 0.0
        for asset, holding in self.holdings.items():
            if holding.market_value is not None:
                total_holdings_value += holding.market_value
            else:
                logger.warning(f"Market value for {asset.asset_id} is None. "
                               "It will not be included in total portfolio value calculation based on market prices.")

        total_portfolio_value = total_holdings_value + self.cash_balance
        logger.debug(f"Total portfolio value for '{self.portfolio_id}': "
                     f"{total_portfolio_value:.2f} (Holdings: {total_holdings_value:.2f}, Cash: {self.cash_balance:.2f})")
        return total_portfolio_value

    def get_weights(self) -> Dict['Asset', float]:
        """
        Calculates the current weight of each asset in the portfolio.
        Weights are based on last-updated market values.

        Returns:
            A dictionary mapping each Asset object to its weight (float).
        """
        total_value = self.get_total_value()
        weights: Dict['Asset', float] = {}

        if total_value == 0:
            logger.warning(f"Total portfolio value is 0. Cannot calculate asset weights for portfolio '{self.portfolio_id}'.")
            for asset in self.holdings.keys():
                weights[asset] = 0.0
            return weights

        for asset, holding in self.holdings.items():
            if holding.market_value is not None:
                weights[asset] = holding.market_value / total_value
            else:
                weights[asset] = 0.0
                logger.warning(f"Weight for {asset.asset_id} is 0 due to missing market value.")

        return weights

    def get_holdings_summary(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing current holdings.

        Returns:
            A pandas DataFrame with columns like ['AssetID', 'Ticker', 'Name', 'Quantity',
            'AvgCostPrice', 'CurrentPrice', 'MarketValue', 'Weight'].
        """
        portfolio_total_value = self.get_total_value()

        summary_data = []
        for asset, holding in self.holdings.items():
            from ..asset.equity import Equity
            ticker = asset.ticker if isinstance(asset, Equity) else asset.asset_id
            name = asset.name

            weight = (holding.market_value / portfolio_total_value) if portfolio_total_value != 0 and holding.market_value is not None else 0.0
            summary_data.append({
                'AssetID': asset.asset_id,
                'Ticker': ticker,
                'Name': name,
                'Quantity': holding.quantity,
                'AvgCostPrice': holding.average_cost_price,
                'CurrentPrice': holding.current_price,
                'MarketValue': holding.market_value,
                'Weight': weight
            })

        # Add cash row
        summary_data.append({
            'AssetID': 'CASH', 'Ticker': 'CASH', 'Name': 'Cash Balance', 'Quantity': 1.0,
            'AvgCostPrice': self.cash_balance, 'CurrentPrice': self.cash_balance,
            'MarketValue': self.cash_balance,
            'Weight': (self.cash_balance / portfolio_total_value) if portfolio_total_value != 0 else (1.0 if self.cash_balance > 0 else 0.0)
        })

        return pd.DataFrame(summary_data)


    def __repr__(self) -> str:
        return (f"Portfolio(portfolio_id='{self.portfolio_id}', "
                f"num_holdings={len(self.holdings)}, cash_balance={self.cash_balance:.2f})")
