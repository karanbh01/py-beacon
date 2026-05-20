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
    from ..backtest.result import BacktestResult


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
        self._last_backtest_result: 'BacktestResult | None' = None


    def _fetch_price(self, ticker: str, current_date: pd.Timestamp) -> float | None:
        """Helper to fetch a single closing price via data_provider."""
        date_str = current_date.strftime('%Y-%m-%d')
        price_data = self.data_provider.fetch_market_data(ticker, date_str, date_str)
        if price_data.empty:
            return None

        price_column = "CLOSE" if "CLOSE" in price_data.columns else "Close"
        if price_column not in price_data.columns or pd.isna(price_data[price_column].iloc[0]):
            return None
        return price_data[price_column].iloc[0]

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

        # 1. Get target constituents and weights from the index calculator.
        eligible_universe = self.index_agent._get_universe(current_date)
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

        # 2. Delegate portfolio execution to the backtest engine.
        from ..backtest.engine import BacktestEngine

        self._update_portfolio_prices(current_date)
        current_portfolio_value = self.portfolio.get_total_value()
        if current_portfolio_value == 0 and self.portfolio.cash_balance > 0:
            current_portfolio_value = self.portfolio.cash_balance

        engine = BacktestEngine(
            start_date=current_date.strftime('%Y-%m-%d'),
            end_date=current_date.strftime('%Y-%m-%d'),
            initial_capital=current_portfolio_value,
            data_provider=self.data_provider,
            target_weights={current_date: target_weights_by_id},
            portfolio=self.portfolio,
        )
        self._last_backtest_result = engine.run()
        self.portfolio = engine.portfolio

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
            nav -= fee_amount

        logger.debug(f"Calculated NAV for fund '{self.fund_id}' on {current_date.strftime('%Y-%m-%d')}: {nav:.2f}")
        return nav

    def __repr__(self) -> str:
        return (f"IndexFund(fund_id='{self.fund_id}', "
                f"target_index='{self.target_index_definition.index_name}', "
                f"management_fee_bps={self.management_fee_bps})")
