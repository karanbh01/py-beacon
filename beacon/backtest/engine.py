# beacon/backtest/engine.py
"""
Module defining the backtesting engine.
"""
import pandas as pd
import logging
from typing import List, Dict, Optional, Any, TYPE_CHECKING

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..asset.base import Asset
    from ..index.constructor import IndexDefinition
    from .rules import BacktestRule
    from ..portfolio.base import Portfolio, Transaction
    from ..data.fetcher import DataFetcher
    from ..index.calculation import IndexCalculator


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
                 index_agent: Optional['IndexCalculator'] = None, # Provides logic
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
        self.index_agent: Optional['IndexCalculator'] = index_agent
        self.rules: List['BacktestRule'] = rules if rules else []

        if portfolio:
            self.portfolio: 'Portfolio' = portfolio
        else:
            from ..portfolio.base import Portfolio # Local import
            self.portfolio = Portfolio(portfolio_id="backtest_portfolio", initial_cash=initial_capital)

        self.results: Optional[pd.DataFrame] = None
        self._current_date: Optional[pd.Timestamp] = None
        self._last_rebalance_date: Optional[pd.Timestamp] = None

    def _fetch_portfolio_prices(self, date: pd.Timestamp) -> Dict[str, float]:
        """Fetch closing prices for all portfolio holdings and return as a dict."""
        prices: Dict[str, float] = {}
        date_str = date.strftime('%Y-%m-%d')
        for asset in self.portfolio.holdings:
            ticker = getattr(asset, 'ticker', asset.asset_id)
            try:
                price_df = self.data_provider.fetch_prices(ticker, date_str, date_str)
                if not price_df.empty and 'Adj Close' in price_df.columns and pd.notna(price_df['Adj Close'].iloc[0]):
                    prices[asset.asset_id] = price_df['Adj Close'].iloc[0]
            except Exception as e:
                logger.error(f"Error fetching price for {ticker} on {date_str}: {e}")
        return prices

    def _update_portfolio_prices(self, date: pd.Timestamp) -> None:
        """Fetch prices and push them into the portfolio."""
        prices = self._fetch_portfolio_prices(date)
        self.portfolio.update_prices(prices)

    def _handle_corporate_actions(self, date: pd.Timestamp) -> None:
        """
        Placeholder for handling corporate actions.
        This would adjust portfolio holdings or cash based on corporate actions
        fetched for assets in the portfolio on the given date.

        Args:
            date: The current date to check for corporate actions.
        """
        logger.debug(f"[{date}] Checking for corporate actions (not yet implemented).")
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
            if isinstance(rule, RebalanceRule):
                if rule._is_rebalance_date(date, self._last_rebalance_date):
                    logger.info(f"[{date}] Executing rebalance rule: {rule}")

                    potential_universe: List['Asset'] = []
                    if self.index_definition and hasattr(self.index_definition, 'get_eligible_universe'):
                         pass # Placeholder for universe definition

                    new_constituents, new_target_weights = rule.apply(
                        current_date=date,
                        current_universe=potential_universe,
                        current_weights=self.portfolio.get_weights(),
                        portfolio=self.portfolio
                    )

                    logger.info(f"[{date}] New target constituents: {[asset.asset_id for asset in new_constituents]}")
                    logger.info(f"[{date}] New target weights: { {asset.asset_id: w for asset, w in new_target_weights.items()} }")

                    from ..portfolio.base import Transaction

                    current_portfolio_value = self.portfolio.get_total_value()

                    # 1. Sell assets no longer in the index or to reduce overweight positions
                    for asset, holding in list(self.portfolio.holdings.items()):
                        date_str = date.strftime('%Y-%m-%d')
                        asset_price_data = self.data_provider.fetch_prices(asset.ticker, date_str, date_str)
                        if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                            logger.warning(f"[{date}] No price data for {asset.ticker} to sell. Skipping sell.")
                            continue
                        sell_price = asset_price_data['Close'].iloc[0]

                        if asset not in new_target_weights or new_target_weights.get(asset, 0) < holding.quantity * sell_price / current_portfolio_value :
                            logger.debug(f"[{date}] Selling {holding.quantity} of {asset.ticker} at {sell_price}")
                            sell_transaction = Transaction(
                                asset=asset,
                                quantity=holding.quantity,
                                price=sell_price,
                                transaction_type='SELL',
                                transaction_date=date
                            )
                            self.portfolio.add_transaction(sell_transaction)

                    # 2. Buy new assets or increase underweight positions
                    for asset, target_weight in new_target_weights.items():
                        if target_weight <= 0: continue

                        target_value = current_portfolio_value * target_weight
                        date_str = date.strftime('%Y-%m-%d')
                        asset_price_data = self.data_provider.fetch_prices(asset.ticker, date_str, date_str)

                        if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                            logger.warning(f"[{date}] No price data for {asset.ticker} to buy. Skipping buy.")
                            continue
                        buy_price = asset_price_data['Close'].iloc[0]

                        if buy_price <= 0:
                            logger.warning(f"[{date}] Invalid price ({buy_price}) for {asset.ticker}. Skipping buy.")
                            continue

                        current_holding_value = 0
                        if asset in self.portfolio.holdings:
                            current_holding_value = self.portfolio.holdings[asset].quantity * buy_price

                        value_to_buy = target_value - current_holding_value
                        if value_to_buy > 0:
                            quantity_to_buy = value_to_buy / buy_price
                            if self.portfolio.cash_balance >= value_to_buy:
                                logger.debug(f"[{date}] Buying {quantity_to_buy:.2f} of {asset.ticker} at {buy_price}")
                                buy_transaction = Transaction(
                                    asset=asset,
                                    quantity=quantity_to_buy,
                                    price=buy_price,
                                    transaction_type='BUY',
                                    transaction_date=date
                                )
                                self.portfolio.add_transaction(buy_transaction)
                            else:
                                logger.warning(f"[{date}] Insufficient cash to buy {asset.ticker}. "
                                               f"Required: {value_to_buy:.2f}, Available: {self.portfolio.cash_balance:.2f}")

                    self._last_rebalance_date = date
                    rebalanced_this_step = True
                    break
        if not rebalanced_this_step:
             logger.debug(f"[{date}] No rebalance triggered.")

    #todo: vectorise run_backtest for efficiency, currently very iterative and may be slow
    def run_backtest(self) -> pd.DataFrame:
        """
        Executes the backtest from start_date to end_date.

        Returns:
            A pandas DataFrame containing the time series of portfolio/index values,
            constituent weights, and potentially other relevant metrics.
        """
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date} "
                    f"with initial capital {self.initial_capital:.2f}")

        history = []
        trading_days = pd.date_range(start=self.start_date, end=self.end_date, freq='B')

        if trading_days.empty:
            logger.warning("No trading days found in the specified date range.")
            return pd.DataFrame(history)

        self._current_date = trading_days[0]
        self._last_rebalance_date = None

        initial_record = {'Date': self._current_date, 'PortfolioValue': self.initial_capital, 'Cash': self.initial_capital}
        history.append(initial_record)


        for date_idx, current_date_dt in enumerate(trading_days):
            self._current_date = current_date_dt
            logger.debug(f"Processing date: {self._current_date.strftime('%Y-%m-%d')}")

            # 0. Update market values of existing holdings before any actions for the day
            self._update_portfolio_prices(self._current_date)

            # 1. Handle corporate actions
            self._handle_corporate_actions(self._current_date)

            # 2. Apply rules (e.g., rebalancing)
            self._rebalance(self._current_date)

            # 3. Update market values again after rebalancing for end-of-day valuation
            self._update_portfolio_prices(self._current_date)

            # 4. Record portfolio state at end of day
            current_total_value = self.portfolio.get_total_value()
            daily_record: Dict[str, Any] = {
                'Date': self._current_date,
                'PortfolioValue': current_total_value,
                'Cash': self.portfolio.cash_balance
            }
            current_weights = self.portfolio.get_weights()
            for asset, weight in current_weights.items():
                daily_record[f"{asset.asset_id}_weight"] = weight


            history.append(daily_record)

            if date_idx % (len(trading_days) // 10 if len(trading_days) > 10 else 1) == 0 :
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
