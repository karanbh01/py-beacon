# beacon/backtest/rules.py
"""
Module defining rules for backtesting, such as rebalancing rules.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Tuple, TYPE_CHECKING

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from ..asset.asset_base import Asset
    from ..index.calculation_agent import IndexCalculationAgent
    from ..portfolio.portfolio_class import Portfolio # For future _handle_corporate_actions type hint


class BacktestRule(ABC):
    """
    Abstract base class or interface for a backtest rule.
    A rule defines actions to be taken at certain points in the backtest,
    typically related to rebalancing or reconstitution.
    """
    @abstractmethod
    def apply(self,
              current_date: pd.Timestamp,
              current_universe: List['Asset'],
              current_weights: Dict['Asset', float],
              # Potentially add portfolio or other state if rules need more context
              portfolio: 'Portfolio' # Example
             ) -> Tuple[List['Asset'], Dict['Asset', float]]:
        """
        Applies the rule given the current state.

        Args:
            current_date: The current date in the backtest.
            current_universe: The list of assets currently in the universe/portfolio.
            current_weights: The current weights of assets in the portfolio.
            portfolio: The current portfolio state.

        Returns:
            A tuple containing the new list of assets and their new target weights
            after the rule has been applied.
        """
        pass

class RebalanceRule(BacktestRule):
    """
    A rule that triggers rebalancing based on a defined frequency and methodology.
    """
    def __init__(self,
                 rebalance_frequency: str, # e.g., 'MONTHLY', 'QUARTERLY', 'ANNUALLY', 'END_OF_MONTH'
                 index_methodology: 'IndexCalculationAgent'):
        """
        Initializes the RebalanceRule.

        Args:
            rebalance_frequency: A string indicating the rebalancing frequency.
                                 (e.g., 'MONTHLY', 'QUARTERLY', 'ANNUALLY').
                                 Specific dates might require more complex logic or a calendar.
            index_methodology: An instance of IndexCalculationAgent that defines
                               how the index/portfolio should be rebalanced.
        """
        if not rebalance_frequency:
            raise ValueError("rebalance_frequency cannot be empty.")
        if not index_methodology: # Check if it's a valid IndexCalculationAgent instance
            raise ValueError("index_methodology must be provided.")

        self.rebalance_frequency: str = rebalance_frequency
        self.index_methodology: 'IndexCalculationAgent' = index_methodology
        # Store the next rebalance date to avoid re-calculating it every step
        self._next_rebalance_date: Optional[pd.Timestamp] = None


    def _is_rebalance_date(self, current_date: pd.Timestamp, previous_rebalance_date: Optional[pd.Timestamp]) -> bool:
        """
        Determines if the current_date is a rebalance date based on the frequency.
        This is a simplified example. Real-world scenarios might need business day conventions,
        specific days of the month (e.g., last trading day), etc.
        """
        if previous_rebalance_date is None: # First rebalance
            return True

        # This simple logic assumes rebalancing at fixed intervals.
        # More sophisticated date logic (e.g., using pandas DateOffsets) would be better.
        if self.rebalance_frequency.upper() == 'MONTHLY':
            return current_date.month != previous_rebalance_date.month
        elif self.rebalance_frequency.upper() == 'QUARTERLY':
            return (current_date.year > previous_rebalance_date.year) or \
                   (current_date.quarter != previous_rebalance_date.quarter)
        elif self.rebalance_frequency.upper() == 'ANNUALLY':
            return current_date.year != previous_rebalance_date.year
        # Add more frequencies as needed, or use pandas.tseries.offsets for robust date calculations
        # e.g., if current_date >= previous_rebalance_date + pd.offsets.MonthEnd(1) for end of month
        return False # Default if frequency not recognized or not yet time

    def apply(self,
              current_date: pd.Timestamp,
              current_universe: List['Asset'], # This might be the full eligible universe from data source
              current_weights: Dict['Asset', float],
              portfolio: 'Portfolio' # Portfolio state might be more relevant here
             ) -> Tuple[List['Asset'], Dict['Asset', float]]:
        """
        Implements rebalancing logic based on the linked IndexCalculationAgent.
        If it's a rebalance date, it recalculates constituents and weights.
        Otherwise, it returns the current assets and weights.

        Args:
            current_date: The current date in the backtest.
            current_universe: The broader universe of assets available for selection.
            current_weights: The current weights in the portfolio being managed.
            portfolio: The current portfolio.

        Returns:
            A tuple containing the new list of constituent assets and their target weights.
            If no rebalance occurs, it might return the existing constituents/weights or signal no change.
        """
        # This method would typically be called by the BacktestEngine, which would know the *portfolio's*
        # last rebalance date. For now, this is a simplified placeholder.
        # A more robust implementation would involve the engine managing rebalance dates.

        # Placeholder: For simplicity, assume it decides to rebalance.
        # In a real engine, the engine would check if it's a rebalance day.
        # This `apply` method would then *perform* the rebalancing.

        # 1. Select new constituents from the provided universe based on eligibility rules.
        #    The `current_universe` argument here likely means the overall pool of available assets.
        new_constituents = self.index_methodology.select_constituents(
            universe=current_universe, # The universe from which to select
            current_date=current_date
        )

        # 2. Calculate new target weights for these constituents.
        new_target_weights = self.index_methodology.calculate_constituent_weights(
            constituents=new_constituents,
            current_date=current_date
        )

        # The rule itself doesn't change the portfolio, it just provides the new target.
        # The BacktestEngine would use this output to generate trades.
        return new_constituents, new_target_weights