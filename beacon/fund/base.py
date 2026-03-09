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
    from ..index.calculation import IndexCalculator


logger = logging.getLogger(__name__)

class IndexFund:
    """
    Represents an index fund that aims to track a target index.
    """
    def __init__(self,
                 fund_id: str,
                 target_index_definition: 'IndexDefinition', # The static definition
                 index_agent: 'IndexCalculator', # The agent to calculate weights for target_index
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
        self.index_agent: 'IndexCalculator' = index_agent
        self.portfolio: 'Portfolio' = portfolio
        self.data_provider: 'DataFetcher' = data_provider
        self.management_fee_bps: int = management_fee_bps # e.g., 20 for 0.20%

        # Store target weights, to be updated upon rebalance_to_index
        self._target_weights: Dict['Asset', float] = {}


    def _fetch_price(self, ticker: str, current_date: pd.Timestamp) -> float | None:
        """Helper to fetch a single closing price via data_provider."""
        date_str = current_date.strftime('%Y-%m-%d')
        price_data = self.data_provider.fetch_prices(ticker, date_str, date_str)
        if price_data.empty or pd.isna(price_data['Close'].iloc[0]):
            return None
        return price_data['Close'].iloc[0]

    def _update_portfolio_prices(self, current_date: pd.Timestamp) -> None:
        """Fetch prices for all holdings and push them into the portfolio."""
        prices: Dict[str, float] = {}
        for asset_id in self.portfolio.holdings:
            price = self._fetch_price(asset_id, current_date)
            if price is not None:
                prices[asset_id] = price
        self.portfolio.update_prices(prices)

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
        eligible_universe = []
        if hasattr(self.target_index_definition, 'get_eligible_universe'):
            pass # Placeholder

        target_constituents = self.index_agent.select_constituents(
            universe=eligible_universe,
            current_date=current_date
        )
        self._target_weights = self.index_agent.calculate_constituent_weights(
            constituents=target_constituents,
            current_date=current_date
        )

        logger.debug(f"Target weights for '{self.fund_id}': {{asset.asset_id: w for asset, w in self._target_weights.items()}}")

        # Build target weights keyed by asset_id string
        target_weights_by_id: Dict[str, float] = {
            asset.asset_id: w for asset, w in self._target_weights.items()
        }

        # 2. Adjust the fund's portfolio to match these target_weights
        self._update_portfolio_prices(current_date)
        current_portfolio_value = self.portfolio.get_total_value()
        if current_portfolio_value == 0 and self.portfolio.cash_balance > 0:
            current_portfolio_value = self.portfolio.cash_balance

        # Sell assets not in target or overweights
        for asset_id, holding in list(self.portfolio.holdings.items()):
            current_price = self._fetch_price(asset_id, current_date)
            if current_price is None:
                logger.warning(f"[{current_date}] No price for {asset_id} to sell during fund rebalance.")
                continue

            target_weight = target_weights_by_id.get(asset_id, 0)
            current_value_of_asset = holding.quantity * current_price

            if target_weight == 0 or (current_value_of_asset > current_portfolio_value * target_weight):
                quantity_to_sell = holding.quantity
                if target_weight > 0:
                    value_to_keep = current_portfolio_value * target_weight
                    quantity_to_keep = value_to_keep / current_price
                    quantity_to_sell = holding.quantity - quantity_to_keep

                if quantity_to_sell > 1e-6:
                    self.portfolio.execute_sell(asset_id, quantity_to_sell, current_price, date=current_date)
                    logger.debug(f"Fund rebalance: Sold {quantity_to_sell:.2f} of {asset_id}")

        # Buy assets in target or underweights
        for asset_id, target_weight in target_weights_by_id.items():
            if target_weight <= 0: continue

            current_price = self._fetch_price(asset_id, current_date)
            if current_price is None or current_price <= 0:
                logger.warning(f"[{current_date}] No price for {asset_id} to buy during fund rebalance.")
                continue

            target_value_of_asset = current_portfolio_value * target_weight
            current_holding_value = 0
            if asset_id in self.portfolio.holdings:
                current_holding_value = self.portfolio.holdings[asset_id].quantity * current_price

            value_to_buy = target_value_of_asset - current_holding_value
            if value_to_buy > 1e-6:
                quantity_to_buy = value_to_buy / current_price
                if self.portfolio.cash_balance >= value_to_buy:
                    self.portfolio.execute_buy(asset_id, quantity_to_buy, current_price, date=current_date)
                    logger.debug(f"Fund rebalance: Bought {quantity_to_buy:.2f} of {asset_id}")
                else:
                    logger.warning(f"Fund rebalance: Insufficient cash to buy {asset_id} for fund '{self.fund_id}'.")

        logger.info(f"Fund '{self.fund_id}' rebalancing completed for {current_date.strftime('%Y-%m-%d')}.")


    def calculate_nav(self, current_date: pd.Timestamp) -> float:
        """
        Calculates the Net Asset Value (NAV) of the fund.

        Args:
            current_date: The date for which to calculate NAV.

        Returns:
            The total Net Asset Value of the fund's portfolio.
        """
        self._update_portfolio_prices(current_date)
        nav = self.portfolio.get_total_value()

        if self.management_fee_bps > 0:
            daily_fee_rate = (self.management_fee_bps / 10000.0) / 252.0
            fee_amount = nav * daily_fee_rate
            # NAV before daily fee deduction (placeholder for more sophisticated fee model)

        logger.debug(f"Calculated NAV for fund '{self.fund_id}' on {current_date.strftime('%Y-%m-%d')}: {nav:.2f}")
        return nav

    def __repr__(self) -> str:
        return (f"IndexFund(fund_id='{self.fund_id}', "
                f"target_index='{self.target_index_definition.index_name}', "
                f"management_fee_bps={self.management_fee_bps})")
