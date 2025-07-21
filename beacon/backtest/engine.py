# beacon/backtest/engine.py
"""
Module defining the backtesting engine.
"""
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..asset.asset_base import Asset
    from ..index.constructor import IndexDefinition
    from .rules import BacktestRule
    from ..portfolio.portfolio_class import Portfolio, Transaction
    from ..data.data_fetcher import DataFetcher
    from ..index.calculation_agent import IndexCalculationAgent


logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Executes a backtest of an index or trading strategy.
    """
    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float,
                 data_provider: 'DataFetcher',
                 index_definition: Optional['IndexDefinition'] = None, # For index-based strategies
                 index_agent: Optional['IndexCalculationAgent'] = None, # Provides logic
                 rules: Optional[List['BacktestRule']] = None,
                 portfolio: Optional['Portfolio'] = None): # Can start with an existing portfolio
        """
        Initializes the BacktestEngine.

        Args:
            start_date: The start date of the backtest (YYYY-MM-DD).
            end_date: The end date of the backtest (YYYY-MM-DD).
            initial_capital: The initial capital for the backtest.
            data_provider: An instance of a DataFetcher to get market data.
            index_definition: The definition of the index to be backtested (optional).
            index_agent: The calculation agent for the index, used by rebalancing rules (optional).
            rules: A list of BacktestRule objects to apply during the backtest.
            portfolio: An initial Portfolio object. If None, one will be created.
        """
        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        self.initial_capital: float = initial_capital
        self.data_provider: 'DataFetcher' = data_provider
        self.index_definition: Optional['IndexDefinition'] = index_definition
        self.index_agent: Optional['IndexCalculationAgent'] = index_agent # Crucial for rebalancing logic
        self.rules: List['BacktestRule'] = rules if rules else []

        if portfolio:
            self.portfolio: 'Portfolio' = portfolio
        else:
            from ..portfolio.portfolio_class import Portfolio # Local import
            self.portfolio = Portfolio(portfolio_id="backtest_portfolio", initial_cash=initial_capital)

        self.results: Optional[pd.DataFrame] = None
        self._current_date: Optional[pd.Timestamp] = None
        self._last_rebalance_date: Optional[pd.Timestamp] = None

    def _handle_corporate_actions(self, date: pd.Timestamp) -> None:
        """
        Placeholder for handling corporate actions.
        This would adjust portfolio holdings or cash based on corporate actions
        fetched for assets in the portfolio on the given date.

        Args:
            date: The current date to check for corporate actions.
            # portfolio: The portfolio to adjust. (self.portfolio can be used)
        """
        logger.debug(f"[{date}] Checking for corporate actions (not yet implemented).")
        # Example logic:
        # for asset, holding in self.portfolio.holdings.items():
        #     actions = asset.get_corporate_actions(date, date, self.data_provider)
        #     for action in actions:
        #         if action['type'] == 'SPLIT':
        #             # Adjust holding.quantity
        #             pass
        #         elif action['type'] == 'DIVIDEND':
        #             # Add to self.portfolio.cash_balance
        #             pass
        pass


    def _rebalance(self, date: pd.Timestamp) -> None:
        """
        Applies rebalancing rules and adjusts the portfolio.

        Args:
            date: The current date, potentially a rebalancing date.
        """
        logger.info(f"[{date}] Attempting rebalance.")
        rebalanced_this_step = False
        for rule in self.rules:
            if isinstance(rule, RebalanceRule): # Check if it's a RebalanceRule
                # The RebalanceRule's _is_rebalance_date needs more robust implementation
                # For now, assume the engine determines if it's time based on rule's frequency
                # This is a conceptual simplification:
                if rule._is_rebalance_date(date, self._last_rebalance_date): # Simplified check
                    logger.info(f"[{date}] Executing rebalance rule: {rule}")

                    # The universe for reselection should come from a broader source,
                    # possibly defined in IndexDefinition or fetched via DataFetcher.
                    # For now, we assume the rule's IndexCalculationAgent can get what it needs.
                    # This part needs refinement based on how universe is defined and accessed.
                    # Placeholder: fetch a broad universe of assets
                    # Assume index_agent.select_constituents can access a pre-defined universe or fetch one.
                    # This is a simplification. The 'current_universe' for the rule might be
                    # all tradable assets, or assets meeting some pre-filter.
                    
                    # The rule.apply method should now use the index_agent associated with it.
                    # The IndexCalculationAgent within the RebalanceRule already has data_provider.
                    # It also needs the full available universe to select from.
                    
                    # Get all potential assets from data_provider (simplified)
                    # This is highly conceptual and needs a proper mechanism to define the universe.
                    # For example, from IndexDefinition's universe specification.
                    potential_universe: List['Asset'] = [] # This should be populated realistically.
                    if self.index_definition and hasattr(self.index_definition, 'get_eligible_universe'):
                         # potential_universe = self.index_definition.get_eligible_universe(date, self.data_provider)
                         pass # Placeholder for universe definition

                    new_constituents, new_target_weights = rule.apply(
                        current_date=date,
                        current_universe=potential_universe, # Pass the appropriate universe
                        current_weights=self.portfolio.get_weights(self.data_provider, date), # Current actual weights
                        portfolio=self.portfolio
                    )

                    # Generate trades to align portfolio with new_target_weights
                    # This is a complex step involving calculating trade sizes, considering cash, etc.
                    logger.info(f"[{date}] New target constituents: {[asset.asset_id for asset in new_constituents]}")
                    logger.info(f"[{date}] New target weights: { {asset.asset_id: w for asset, w in new_target_weights.items()} }")

                    # Simplistic rebalancing: sell all, then buy new weights
                    # (Highly inefficient and not realistic due to transaction costs, liquidity)
                    from ..portfolio.portfolio_class import Transaction # Local import
                    
                    # Sell existing holdings not in new constituents or to adjust weights
                    current_portfolio_value = self.portfolio.get_total_value(self.data_provider, date)
                    
                    # 1. Sell assets no longer in the index or to reduce overweight positions
                    assets_to_sell = []
                    for asset, holding in list(self.portfolio.holdings.items()): # Iterate over a copy for modification
                        asset_price_data = self.data_provider.fetch_prices(asset.ticker, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))
                        if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                            logger.warning(f"[{date}] No price data for {asset.ticker} to sell. Skipping sell.")
                            continue
                        sell_price = asset_price_data['Close'].iloc[0]

                        if asset not in new_target_weights or new_target_weights.get(asset, 0) < holding.quantity * sell_price / current_portfolio_value :
                            logger.debug(f"[{date}] Selling {holding.quantity} of {asset.ticker} at {sell_price}")
                            sell_transaction = Transaction(
                                asset=asset,
                                quantity=holding.quantity, # Sell all of it for simplicity here
                                price=sell_price,
                                transaction_type='SELL',
                                transaction_date=date
                            )
                            self.portfolio.add_transaction(sell_transaction, self.data_provider, date) # Date for valuation if needed

                    # 2. Buy new assets or increase underweight positions
                    for asset, target_weight in new_target_weights.items():
                        if target_weight <= 0: continue

                        target_value = current_portfolio_value * target_weight
                        asset_price_data = self.data_provider.fetch_prices(asset.ticker, date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))

                        if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                            logger.warning(f"[{date}] No price data for {asset.ticker} to buy. Skipping buy.")
                            continue
                        buy_price = asset_price_data['Close'].iloc[0]
                        
                        if buy_price <= 0:
                            logger.warning(f"[{date}] Invalid price ({buy_price}) for {asset.ticker}. Skipping buy.")
                            continue

                        current_holding_value = 0
                        if asset in self.portfolio.holdings:
                            current_holding_value = self.portfolio.holdings[asset].quantity * buy_price # Approx.

                        value_to_buy = target_value - current_holding_value
                        if value_to_buy > 0 : # Only if we need to buy more
                            quantity_to_buy = value_to_buy / buy_price
                            if self.portfolio.cash_balance >= value_to_buy: # Check affordability
                                logger.debug(f"[{date}] Buying {quantity_to_buy:.2f} of {asset.ticker} at {buy_price}")
                                buy_transaction = Transaction(
                                    asset=asset,
                                    quantity=quantity_to_buy,
                                    price=buy_price,
                                    transaction_type='BUY',
                                    transaction_date=date
                                )
                                self.portfolio.add_transaction(buy_transaction, self.data_provider, date)
                            else:
                                logger.warning(f"[{date}] Insufficient cash to buy {asset.ticker}. "
                                               f"Required: {value_to_buy:.2f}, Available: {self.portfolio.cash_balance:.2f}")
                    
                    self._last_rebalance_date = date
                    rebalanced_this_step = True
                    break # Assuming one rebalance rule for now, or handle priorities if multiple
        if not rebalanced_this_step:
             logger.debug(f"[{date}] No rebalance triggered.")


    def run_backtest(self) -> pd.DataFrame:
        """
        Executes the backtest from start_date to end_date.

        Returns:
            A pandas DataFrame containing the time series of portfolio/index values,
            constituent weights, and potentially other relevant metrics.
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date} "
                    f"with initial capital {self.initial_capital:.2f}")

        # Initialize results storage
        # Columns: Date, PortfolioValue, Cash, Asset1_Weight, Asset1_Price, ..., AssetN_Weight, AssetN_Price
        # This will grow dynamically, which is not ideal for performance but simpler for now.
        history = []

        # Determine the date range for the backtest (e.g., daily frequency)
        # This should ideally use a market calendar. For now, all weekdays.
        trading_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B') # Business days

        if trading_days.empty:
            logger.warning("No trading days found in the specified date range.")
            return pd.DataFrame(history)

        self._current_date = trading_days[0]
        self._last_rebalance_date = None # Or set to a date before start_date if needed for first rebalance check

        # Initialize portfolio value for the first day (before any operations)
        # On day 1, portfolio is just cash before any rebalancing.
        initial_record = {'Date': self._current_date, 'PortfolioValue': self.initial_capital, 'Cash': self.initial_capital}
        history.append(initial_record)


        for date_idx, current_date_dt in enumerate(trading_days):
            self._current_date = current_date_dt
            logger.debug(f"Processing date: {self._current_date.strftime('%Y-%m-%d')}")

            # 0. Update market values of existing holdings before any actions for the day
            self.portfolio.update_market_values(self._current_date, self.data_provider)

            # 1. Handle corporate actions (adjusts holdings based on splits, dividends etc.)
            self._handle_corporate_actions(self._current_date) # Affects portfolio state before rebalancing

            # 2. Apply rules (e.g., rebalancing)
            # The rebalance method itself will check if it's a rebalance day.
            self._rebalance(self._current_date) # Affects portfolio holdings and cash

            # 3. Update market values again after rebalancing for end-of-day valuation
            self.portfolio.update_market_values(self._current_date, self.data_provider)

            # 4. Record portfolio state at end of day
            current_total_value = self.portfolio.get_total_value(self.data_provider, self._current_date) # Ensure this uses EOD prices
            daily_record: Dict[str, Any] = {
                'Date': self._current_date,
                'PortfolioValue': current_total_value,
                'Cash': self.portfolio.cash_balance
            }
            # Record weights (this can make the DataFrame very wide)
            current_weights = self.portfolio.get_weights(self.data_provider, self._current_date)
            for asset, weight in current_weights.items():
                daily_record[f"{asset.asset_id}_weight"] = weight
                # Optionally, record prices used for valuation
                # price_data = self.data_provider.fetch_prices(asset.ticker, self._current_date.strftime('%Y-%m-%d'), self._current_date.strftime('%Y-%m-%d'))
                # if not price_data.empty:
                #    daily_record[f"{asset.asset_id}_price"] = price_data['Close'].iloc[0]


            history.append(daily_record)

            # Log progress periodically
            if date_idx % (len(trading_days) // 10 if len(trading_days) > 10 else 1) == 0 : # Log roughly 10 times
                 logger.info(f"Backtest progress: {((date_idx + 1) / len(trading_days) * 100):.1f}% complete. "
                             f"Date: {self._current_date.strftime('%Y-%m-%d')}, Value: {current_total_value:.2f}")


        self.results = pd.DataFrame(history)
        if not self.results.empty:
            self.results.set_index('Date', inplace=True)

        logger.info("Backtest finished.")
        if self.results is not None and not self.results.empty:
            logger.info(f"Final portfolio value: {self.results['PortfolioValue'].iloc[-1]:.2f}")
        else:
            logger.info("Backtest resulted in no data.")
            
        return self.results if self.results is not None else pd.DataFrame()