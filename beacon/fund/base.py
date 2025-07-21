# beacon/fund/base.py
"""
Module defining the IndexFund class.
"""
import pandas as pd
from typing import Dict, Any, TYPE_CHECKING
import logging

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..index.constructor import IndexDefinition
    from ..portfolio.base import Portfolio
    from ..data.fetcher import DataFetcher
    from ..index.calculation import IndexCalculationAgent


logger = logging.getLogger(__name__)

class IndexFund:
    """
    Represents an index fund that aims to track a target index.
    """
    def __init__(self,
                 fund_id: str,
                 target_index_definition: 'IndexDefinition', # The static definition
                 index_agent: 'IndexCalculationAgent', # The agent to calculate weights for target_index
                 portfolio: 'Portfolio',
                 data_provider: 'DataFetcher',
                 management_fee_bps: int = 0):
        """
        Initializes an IndexFund.

        Args:
            fund_id: A unique identifier for the fund.
            target_index_definition: The definition of the index the fund aims to track.
            index_agent: The calculation agent associated with the target_index_definition.
                         Used to get target weights.
            portfolio: The Portfolio object representing the fund's holdings.
            data_provider: DataFetcher instance for market data.
            management_fee_bps: The annual management fee in basis points (e.g., 10 bps = 0.1%).
        """
        if not fund_id:
            raise ValueError("fund_id cannot be empty.")
        if not target_index_definition:
            raise ValueError("target_index_definition must be provided.")
        if not index_agent:
            raise ValueError("index_agent must be provided.")
        if not portfolio:
            raise ValueError("portfolio must be provided.")
        if not data_provider:
            raise ValueError("data_provider must be provided.")
        if management_fee_bps < 0:
            raise ValueError("management_fee_bps cannot be negative.")

        self.fund_id: str = fund_id
        self.target_index_definition: 'IndexDefinition' = target_index_definition
        self.index_agent: 'IndexCalculationAgent' = index_agent
        self.portfolio: 'Portfolio' = portfolio
        self.data_provider: 'DataFetcher' = data_provider
        self.management_fee_bps: int = management_fee_bps # e.g., 20 for 0.20%

        # Store target weights, to be updated upon rebalance_to_index
        self._target_weights: Dict['Asset', float] = {}


    def rebalance_to_index(self, current_date: pd.Timestamp) -> None:
        """
        Adjusts the fund's internal portfolio to match the target_index weights.
        This method would determine the target weights from the index_agent
        and then generate transactions in its portfolio to align.
        For simplicity, this assumes perfect replication and immediate execution.

        Args:
            current_date: The date on which rebalancing occurs.
        """
        logger.info(f"[{current_date.strftime('%Y-%m-%d')}] Fund '{self.fund_id}' rebalancing to target index '{self.target_index_definition.index_name}'.")

        # 1. Get the current constituents and weights from the index agent
        # The universe for selection should be appropriately sourced by the index_agent
        # This is a simplification, actual universe definition is complex.
        eligible_universe = [] # This needs to be defined, e.g., from index_definition
        if hasattr(self.target_index_definition, 'get_eligible_universe'):
            # eligible_universe = self.target_index_definition.get_eligible_universe(current_date, self.data_provider)
            pass # Placeholder

        target_constituents = self.index_agent.select_constituents(
            universe=eligible_universe, # Needs a proper source for the universe
            current_date=current_date
        )
        self._target_weights = self.index_agent.calculate_constituent_weights(
            constituents=target_constituents,
            current_date=current_date
        )
        
        logger.debug(f"Target weights for '{self.fund_id}': {{asset.asset_id: w for asset, w in self._target_weights.items()}}")

        # 2. Adjust the fund's portfolio to match these target_weights
        # This is a complex operation:
        # - Get current portfolio value.
        # - For each asset in target_weights, calculate target value.
        # - Compare with current holding value.
        # - Generate buy/sell transactions.
        # - This should ideally use the Portfolio's trading logic.
        # For now, we'll conceptualize it. The BacktestEngine has a more detailed rebalance.

        from ..portfolio.portfolio_class import Transaction # Local import to avoid circularity at module level

        current_portfolio_value = self.portfolio.get_total_value(self.data_provider, current_date)
        if current_portfolio_value == 0 and self.portfolio.cash_balance > 0 : # Initial setup
            current_portfolio_value = self.portfolio.cash_balance

        # Simplified rebalancing logic (similar to BacktestEngine's _rebalance)
        # Sell assets not in target or overweights
        for asset, holding in list(self.portfolio.holdings.items()): # Iterate copy
            asset_price_data = self.data_provider.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
            if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                logger.warning(f"[{current_date}] No price for {asset.ticker} to sell during fund rebalance.")
                continue
            current_price = asset_price_data['Close'].iloc[0]
            
            target_weight_for_asset = self._target_weights.get(asset, 0)
            current_value_of_asset = holding.quantity * current_price
            
            # If asset no longer in target or needs reduction
            if target_weight_for_asset == 0 or (current_value_of_asset > current_portfolio_value * target_weight_for_asset):
                quantity_to_sell = holding.quantity # Sell all if not in target
                if target_weight_for_asset > 0: # Reduce to target
                    value_to_keep = current_portfolio_value * target_weight_for_asset
                    quantity_to_keep = value_to_keep / current_price
                    quantity_to_sell = holding.quantity - quantity_to_keep

                if quantity_to_sell > 1e-6: # Avoid tiny sells
                    sell_transaction = Transaction(asset, quantity_to_sell, current_price, 'SELL', current_date)
                    self.portfolio.add_transaction(sell_transaction, self.data_provider, current_date)
                    logger.debug(f"Fund rebalance: Sold {quantity_to_sell:.2f} of {asset.ticker}")


        # Buy assets in target or underweights
        for asset, target_weight in self._target_weights.items():
            if target_weight <= 0: continue

            asset_price_data = self.data_provider.fetch_prices(asset.ticker, current_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
            if asset_price_data.empty or pd.isna(asset_price_data['Close'].iloc[0]):
                logger.warning(f"[{current_date}] No price for {asset.ticker} to buy during fund rebalance.")
                continue
            current_price = asset_price_data['Close'].iloc[0]
            if current_price <= 0: continue

            target_value_of_asset = current_portfolio_value * target_weight
            current_holding_value = 0
            if asset in self.portfolio.holdings:
                current_holding_value = self.portfolio.holdings[asset].quantity * current_price
            
            value_to_buy = target_value_of_asset - current_holding_value
            if value_to_buy > 1e-6: # Avoid tiny buys, use a monetary threshold in practice
                quantity_to_buy = value_to_buy / current_price
                if self.portfolio.cash_balance >= value_to_buy:
                    buy_transaction = Transaction(asset, quantity_to_buy, current_price, 'BUY', current_date)
                    self.portfolio.add_transaction(buy_transaction, self.data_provider, current_date)
                    logger.debug(f"Fund rebalance: Bought {quantity_to_buy:.2f} of {asset.ticker}")
                else:
                    logger.warning(f"Fund rebalance: Insufficient cash to buy {asset.ticker} for fund '{self.fund_id}'.")
        
        logger.info(f"Fund '{self.fund_id}' rebalancing completed for {current_date.strftime('%Y-%m-%d')}.")


    def calculate_nav(self, current_date: pd.Timestamp) -> float:
        """
        Calculates the Net Asset Value (NAV) of the fund.
        NAV = (Total Value of Assets - Liabilities) / Number of Fund Shares.
        For simplicity, assumes no other liabilities and fund shares aspect is abstracted.
        This NAV will represent the total value of the managed portfolio.
        If per-share NAV is needed, a 'number_of_fund_shares' attribute would be required.

        Args:
            current_date: The date for which to calculate NAV. The DataProvider
                          will be used to get prices for this date.

        Returns:
            The total Net Asset Value of the fund's portfolio.
        """
        # Ensure portfolio market values are up-to-date for 'current_date'
        self.portfolio.update_market_values(current_date, self.data_provider)
        nav = self.portfolio.get_total_value(self.data_provider, current_date) # This uses the portfolio's data_provider
        
        # Deduct accrued management fee (simplified daily accrual)
        # This is a very basic fee model. Real funds accrue daily, pay periodically.
        # For a backtest, daily deduction from NAV is a common way to model fees.
        if self.management_fee_bps > 0:
            daily_fee_rate = (self.management_fee_bps / 10000.0) / 252.0 # Assuming 252 trading days
            fee_amount = nav * daily_fee_rate
            # This nav is before today's fee.
            # In a simulation, the fee would reduce the return or cash.
            # For a simple NAV calculation, this might represent value before fee deduction for the day.
            # If NAV is post-fee, then nav -= fee_amount
            # Let's assume for now this is NAV before daily fee deduction.
            # The portfolio's value would implicitly be reduced if fees were treated as cash withdrawals.
            
        logger.debug(f"Calculated NAV for fund '{self.fund_id}' on {current_date.strftime('%Y-%m-%d')}: {nav:.2f}")
        return nav

    def __repr__(self) -> str:
        return (f"IndexFund(fund_id='{self.fund_id}', "
                f"target_index='{self.target_index_definition.index_name}', "
                f"management_fee_bps={self.management_fee_bps})")