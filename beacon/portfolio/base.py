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
    from ..data.fetcher import DataFetcher

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
                # Hack to allow pd.Timestamp to be mutable in __setattr__ for dataclass.
                # This is generally not good practice for frozen dataclasses.
                # Better to ensure transaction_date is pd.Timestamp upon creation.
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
    # last_update_date: Optional[pd.Timestamp] = None # When current_price/market_value were last updated

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
        # self.last_update_date = update_date
        logger.debug(f"Holding for {self.asset.asset_id} updated: Qty={self.quantity}, Price={self.current_price}, MV={self.market_value}")


class Portfolio:
    """
    Manages a collection of asset holdings, cash balance, and transaction history.
    """
    def __init__(self,
                 portfolio_id: str,
                 initial_cash: float = 0.0,
                 # data_provider: Optional['DataFetcher'] = None # Make it mandatory if methods rely heavily on it
                ):
        """
        Initializes a Portfolio.

        Args:
            portfolio_id: A unique identifier for the portfolio.
            initial_cash: The starting cash balance of the portfolio.
            # data_provider: A DataFetcher instance for fetching current prices.
        """
        if not portfolio_id:
            raise ValueError("portfolio_id cannot be empty.")
        if initial_cash < 0:
            raise ValueError("Initial cash cannot be negative.")

        self.portfolio_id: str = portfolio_id
        self.holdings: Dict['Asset', Holding] = {} # Maps Asset object to Holding object
        self.cash_balance: float = initial_cash
        self.transactions: List[Transaction] = []
        # self.data_provider: Optional['DataFetcher'] = data_provider # Store if needed by multiple methods

        logger.info(f"Portfolio '{self.portfolio_id}' initialized with cash: {self.cash_balance:.2f}")


    def add_transaction(self, transaction: Transaction, data_provider: 'DataFetcher', current_date_for_valuation: pd.Timestamp) -> None:
        """
        Adds a transaction to the portfolio and updates holdings and cash balance.

        Args:
            transaction: The Transaction object to add.
            data_provider: DataFetcher to get price if needed for updating holding's current price.
            current_date_for_valuation: The date to use for fetching prices for new holdings.
        """
        if not isinstance(transaction, Transaction):
            raise TypeError("transaction must be a Transaction object.")
        if not data_provider:
            raise ValueError("data_provider must be supplied to add_transaction for price updates.")

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
                # Depending on policy, either raise error or just log and skip
                # raise ValueError("Insufficient cash for transaction")
                return # Skip transaction

            self.cash_balance -= (trade_value + tx_cost)

            if asset in self.holdings:
                current_holding = self.holdings[asset]
                # Update average cost price: (old_total_value + new_total_value) / (old_qty + new_qty)
                old_total_value = current_holding.average_cost_price * current_holding.quantity
                new_total_value = price * qty
                current_holding.quantity += qty
                if current_holding.quantity > 1e-9: # Avoid division by zero if quantity becomes tiny
                    current_holding.average_cost_price = (old_total_value + new_total_value) / current_holding.quantity
                else: # Effectively sold and rebought, or very small quantity
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
                # raise ValueError("Insufficient holdings for sell transaction")
                return # Skip transaction

            self.cash_balance += (trade_value - tx_cost) # Proceeds minus cost
            
            self.holdings[asset].quantity -= qty
            logger.info(f"SELL: {qty} of {asset.asset_id} @ {price:.2f}. Cash: {self.cash_balance:.2f}")

            if self.holdings[asset].quantity < 1e-9: # Arbitrary small number to handle float precision
                logger.info(f"Fully sold asset: {asset.asset_id}. Removing from holdings.")
                del self.holdings[asset]
        else:
            logger.error(f"Unknown transaction type: {tx_type}")
            return

        self.transactions.append(transaction)

        # Update market data for the affected holding immediately after transaction
        if asset in self.holdings: # If still holding the asset
            try:
                # Use transaction price as current price immediately after trade, or fetch EOD price
                # For simplicity, if a backtester calls this mid-day with execution price,
                # that's the most recent price known for valuation until EOD update.
                self.holdings[asset].update_market_data(price, transaction.transaction_date)

                # Or, if strictly EOD valuation model:
                # asset_price_df = data_provider.fetch_prices(asset.ticker,
                #                                             current_date_for_valuation.strftime('%Y-%m-%d'),
                #                                             current_date_for_valuation.strftime('%Y-%m-%d'))
                # if not asset_price_df.empty and 'Adj Close' in asset_price_df.columns and pd.notna(asset_price_df['Adj Close'].iloc[0]):
                #     eod_price = asset_price_df['Adj Close'].iloc[0]
                #     self.holdings[asset].update_market_data(eod_price, current_date_for_valuation)
                # else:
                #     logger.warning(f"Could not fetch EOD price for {asset.ticker} on {current_date_for_valuation} post-transaction.")
            except Exception as e:
                 logger.error(f"Error updating market data for {asset.asset_id} post-transaction: {e}")


    def update_market_values(self, current_date: pd.Timestamp, data_provider: 'DataFetcher') -> None:
        """
        Updates the current_price and market_value for all holdings in the portfolio
        using data from the provided data_provider for the given date.

        Args:
            current_date: The date for which to fetch prices and update values.
            data_provider: The DataFetcher instance to use.
        """
        if not data_provider:
            logger.error("Data provider not available to update market values.")
            # raise ValueError("Data provider is required to update market values.")
            return

        logger.debug(f"[{current_date.strftime('%Y-%m-%d')}] Updating market values for portfolio '{self.portfolio_id}'.")
        for asset, holding in self.holdings.items():
            try:
                # Fetch price for the asset on current_date
                # Assuming asset object has a 'ticker' attribute or similar identifier
                price_df = data_provider.fetch_prices(asset.ticker,
                                                      current_date.strftime('%Y-%m-%d'),
                                                      current_date.strftime('%Y-%m-%d'))
                if not price_df.empty and 'Adj Close' in price_df.columns and pd.notna(price_df['Adj Close'].iloc[0]):
                    latest_price = price_df['Adj Close'].iloc[0]
                    holding.update_market_data(latest_price, current_date)
                else:
                    # What to do if price is not available? Use last known price? Log warning?
                    logger.warning(f"Could not retrieve price for {asset.asset_id} ({asset.ticker}) "
                                   f"on {current_date.strftime('%Y-%m-%d')}. Market value may be stale.")
                    # Optionally, mark holding.current_price as None or keep old value
                    # holding.current_price = None # Or keep stale price
                    # holding.market_value = holding.quantity * holding.current_price if holding.current_price else None


            except Exception as e:
                logger.error(f"Error updating market value for {asset.asset_id} ({asset.ticker}): {e}")
                # Decide on fallback for holding.current_price and holding.market_value


    def get_total_value(self, data_provider: 'DataFetcher', current_date: pd.Timestamp) -> float:
        """
        Calculates the total current market value of the portfolio (holdings + cash).
        Ensures market values are updated first if a data_provider is available.

        Args:
            data_provider: DataFetcher to get current prices.
            current_date: The date for which to calculate total value.

        Returns:
            The total portfolio value as a float.
        """
        # Ensure market values are current for the given date
        self.update_market_values(current_date, data_provider)

        total_holdings_value = 0.0
        for asset, holding in self.holdings.items():
            if holding.market_value is not None:
                total_holdings_value += holding.market_value
            else:
                # Fallback if market_value is None (e.g., price not found)
                # Could try to calculate using holding.current_price if available but MV not set
                # Or log and exclude, or use average_cost_price (book value) as a proxy (less ideal for MV)
                logger.warning(f"Market value for {asset.asset_id} is None. "
                               "It will not be included in total portfolio value calculation based on market prices.")

        total_portfolio_value = total_holdings_value + self.cash_balance
        logger.debug(f"Total portfolio value for '{self.portfolio_id}' on {current_date.strftime('%Y-%m-%d')}: "
                     f"{total_portfolio_value:.2f} (Holdings: {total_holdings_value:.2f}, Cash: {self.cash_balance:.2f})")
        return total_portfolio_value

    def get_weights(self, data_provider: 'DataFetcher', current_date: pd.Timestamp) -> Dict['Asset', float]:
        """
        Calculates the current weight of each asset in the portfolio.
        Weights are based on current market values.

        Args:
            data_provider: DataFetcher to ensure prices are current.
            current_date: The date for which to calculate weights.

        Returns:
            A dictionary mapping each Asset object to its weight (float).
        """
        total_value = self.get_total_value(data_provider, current_date)
        weights: Dict['Asset', float] = {}

        if total_value == 0: # Avoid division by zero if portfolio value is zero
            logger.warning(f"Total portfolio value is 0. Cannot calculate asset weights for portfolio '{self.portfolio_id}'.")
            # Return zero weights for existing holdings, or empty dict
            for asset in self.holdings.keys():
                weights[asset] = 0.0
            return weights

        for asset, holding in self.holdings.items():
            if holding.market_value is not None:
                weights[asset] = holding.market_value / total_value
            else:
                weights[asset] = 0.0 # If no market value, weight is 0
                logger.warning(f"Weight for {asset.asset_id} is 0 due to missing market value.")
        
        return weights

    def get_holdings_summary(self, data_provider: 'DataFetcher', current_date: pd.Timestamp) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing current holdings.

        Args:
            data_provider: DataFetcher to ensure prices are current.
            current_date: The date for valuation.

        Returns:
            A pandas DataFrame with columns like ['AssetID', 'Ticker', 'Name', 'Quantity',
            'AvgCostPrice', 'CurrentPrice', 'MarketValue', 'Weight'].
        """
        self.update_market_values(current_date, data_provider) # Ensure data is fresh
        portfolio_total_value = self.get_total_value(data_provider, current_date)

        summary_data = []
        for asset, holding in self.holdings.items():
            from ..asset.equity import Equity # Check type for Ticker/Name
            ticker = asset.ticker if isinstance(asset, Equity) else asset.asset_id
            name = asset.name if isinstance(asset, Equity) else asset.asset_type

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
        
        if not summary_data: # Add cash position if no holdings or to complete summary
            summary_data.append({
                'AssetID': 'CASH', 'Ticker': 'CASH', 'Name': 'Cash Balance', 'Quantity': 1.0,
                'AvgCostPrice': self.cash_balance, 'CurrentPrice': self.cash_balance,
                'MarketValue': self.cash_balance,
                'Weight': (self.cash_balance / portfolio_total_value) if portfolio_total_value != 0 else (1.0 if self.cash_balance > 0 else 0.0)
            })
        else: # Add cash as a separate row if there are holdings
             summary_data.append({
                'AssetID': 'CASH', 'Ticker': 'CASH', 'Name': 'Cash Balance', 'Quantity': 1.0,
                'AvgCostPrice': self.cash_balance, 'CurrentPrice': self.cash_balance,
                'MarketValue': self.cash_balance,
                'Weight': (self.cash_balance / portfolio_total_value) if portfolio_total_value != 0 else 0.0
            })


        return pd.DataFrame(summary_data)


    def __repr__(self) -> str:
        return (f"Portfolio(portfolio_id='{self.portfolio_id}', "
                f"num_holdings={len(self.holdings)}, cash_balance={self.cash_balance:.2f})")