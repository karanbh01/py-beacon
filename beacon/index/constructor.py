# beacon/index/constructor.py
"""
Module for defining the structure and rules of a financial index.
"""
import pandas as pd
from typing import List, Optional, TYPE_CHECKING
import logging

# Avoid circular imports for type hinting
if TYPE_CHECKING:
    from .methodology import EligibilityRuleBase, WeightingSchemeBase
    # from ..asset.asset_base import Asset # If needed for universe definition
    # from ..data.data_fetcher import DataFetcher # If needed for universe definition


logger = logging.getLogger(__name__)

class IndexDefinition:
    """
    Defines the static characteristics and rules for constructing a financial index.
    """
    def __init__(self,
                 index_id: str,
                 index_name: str,
                 base_date: str, # YYYY-MM-DD
                 base_value: float,
                 currency: str,
                 eligibility_rules: List['EligibilityRuleBase'],
                 weighting_scheme: 'WeightingSchemeBase',
                 rebalancing_frequency: str, # e.g., 'QUARTERLY', 'MONTHLY', 'ANNUALLY'
                 description: Optional[str] = None
                 # initial_universe: Optional[List['Asset']] = None, # Consider how universe is managed
                ):
        """
        Initializes an IndexDefinition.

        Args:
            index_id: A unique identifier for the index.
            index_name: The common name of the index.
            base_date: The date from which the index calculation begins (YYYY-MM-DD).
            base_value: The initial value of the index on its base_date.
            currency: The currency of the index.
            eligibility_rules: A list of EligibilityRuleBase objects that define
                               criteria for constituent selection.
            weighting_scheme: A WeightingSchemeBase object that defines how
                              constituents are weighted.
            rebalancing_frequency: A string indicating how often the index is rebalanced
                                   (e.g., 'QUARTERLY', 'MONTHLY', 'SEMI-ANNUALLY', 'ANNUALLY').
                                   More complex schedules (e.g. "Third Friday of March, June...")
                                   would require a more sophisticated scheduler.
            description: Optional textual description of the index.
            # initial_universe: Optional list of assets forming the starting pool for selection.
        """
        if not index_id: raise ValueError("index_id cannot be empty.")
        if not index_name: raise ValueError("index_name cannot be empty.")
        if not base_date: raise ValueError("base_date cannot be empty.")
        if base_value <= 0: raise ValueError("base_value must be positive.")
        if not currency: raise ValueError("currency cannot be empty.")
        if not eligibility_rules: logger.warning(f"Index '{index_name}' defined with no eligibility rules.")
        if not weighting_scheme: raise ValueError("weighting_scheme must be provided.")
        if not rebalancing_frequency: raise ValueError("rebalancing_frequency cannot be empty.")


        self.index_id: str = index_id
        self.index_name: str = index_name
        self.base_date: pd.Timestamp = pd.Timestamp(base_date)
        self.base_value: float = base_value
        self.currency: str = currency.upper()
        self.eligibility_rules: List['EligibilityRuleBase'] = eligibility_rules
        self.weighting_scheme: 'WeightingSchemeBase' = weighting_scheme
        self.rebalancing_frequency: str = rebalancing_frequency.upper()
        self.description: Optional[str] = description
        # self.initial_universe: List['Asset'] = initial_universe if initial_universe else []

        logger.info(f"IndexDefinition for '{self.index_name}' ({self.index_id}) created successfully.")

    # Placeholder for how the broad universe of assets might be defined or fetched.
    # This is a complex area: Is it a fixed list? Dynamically fetched based on criteria (e.g., all stocks on an exchange)?
    # def get_eligible_universe(self, current_date: pd.Timestamp, data_provider: 'DataFetcher') -> List['Asset']:
    # """
    # Returns the broad list of assets from which constituents can be selected
    # before applying eligibility rules. This might involve fetching all assets
    # listed on a specific exchange or a pre-defined list.
    # """
    # # This needs to be implemented based on how the universe is sourced.
    # # For example, it could be configured with a list of tickers, or rules to query a DataFetcher.
    # logger.warning("get_eligible_universe is a placeholder and needs concrete implementation.")
    # return [] # Return an empty list as a placeholder

    def __repr__(self) -> str:
        return (f"IndexDefinition(index_id='{self.index_id}', index_name='{self.index_name}', "
                f"base_date='{self.base_date.strftime('%Y-%m-%d')}', base_value={self.base_value}, "
                f"currency='{self.currency}', rebalancing_frequency='{self.rebalancing_frequency}')")