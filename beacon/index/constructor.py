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
                 description: Optional[str] = None,
                 universe_identifiers: Optional[List[str]] = None
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
            universe_identifiers: Optional list of string identifiers (e.g., tickers, ISINs)
                                  defining the asset universe from which constituents are selected.
        """
        if not index_id: raise ValueError("index_id cannot be empty.")
        if not index_name: raise ValueError("index_name cannot be empty.")
        if not base_date: raise ValueError("base_date cannot be empty.")
        if base_value <= 0: raise ValueError("base_value must be positive.")
        if not currency: raise ValueError("currency cannot be empty.")
        if not eligibility_rules: logger.warning(f"Index '{index_name}' defined with no eligibility rules.")
        if not weighting_scheme: raise ValueError("weighting_scheme must be provided.")
        if not rebalancing_frequency: raise ValueError("rebalancing_frequency cannot be empty.")
        if universe_identifiers is not None and not universe_identifiers:
            raise ValueError("universe_identifiers, when provided, must be a non-empty list.")

        self.index_id: str = index_id
        self.index_name: str = index_name
        self.base_date: pd.Timestamp = pd.Timestamp(base_date)
        self.base_value: float = base_value
        self.currency: str = currency.upper()
        self.eligibility_rules: List['EligibilityRuleBase'] = eligibility_rules
        self.weighting_scheme: 'WeightingSchemeBase' = weighting_scheme
        self.rebalancing_frequency: str = rebalancing_frequency.upper()
        self.description: Optional[str] = description
        self.universe_identifiers: Optional[List[str]] = universe_identifiers

        logger.info(f"IndexDefinition for '{self.index_name}' ({self.index_id}) created successfully.")

    def get_rebalance_dates(self, start_date: str, end_date: str) -> List[pd.Timestamp]:
        """
        Return all rebalance dates within [start_date, end_date] based on
        the index's rebalancing frequency. Dates are adjusted to business days.

        Args:
            start_date: Start of the range (YYYY-MM-DD), inclusive.
            end_date: End of the range (YYYY-MM-DD), inclusive.

        Returns:
            A chronologically sorted list of business-day-adjusted rebalance dates.

        Raises:
            ValueError: If the rebalancing frequency is unsupported.
        """
        #todo: - This method currently supports only simple monthly/quarterly/semi-annual/annual frequencies.
        #      - More complex schedules (e.g. "Third Friday of March, June...") would require a more sophisticated scheduler, potentially using a library like `dateutil` or `pandas` offsets.
        freq = self.rebalancing_frequency
        freq_map = {
            "MONTHLY": 1,
            "QUARTERLY": 3,
            "SEMI-ANNUAL": 6,
            "ANNUAL": 12,
        }

        if freq not in freq_map:
            raise ValueError(
                f"Unsupported rebalancing frequency: '{freq}'. "
                f"Supported values: {list(freq_map.keys())}"
            )

        interval_months = freq_map[freq]
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        # Generate first-business-day-of-month dates covering the range
        # BMonthBegin gives the first business day of each month
        all_bmonth_starts = pd.date_range(
            start=start - pd.offsets.MonthBegin(1),
            end=end + pd.offsets.MonthEnd(1),
            freq="BMS",  # Business Month Start
        )

        # Filter to only months matching the interval from the first candidate
        candidates = []
        for d in all_bmonth_starts:
            if start <= d <= end:
                candidates.append(d)

        if not candidates:
            return []

        # Select dates at the specified interval starting from the first candidate
        rebalance_dates = [candidates[0]]
        for d in candidates[1:]:
            months_diff = (d.year - rebalance_dates[-1].year) * 12 + (d.month - rebalance_dates[-1].month)
            if months_diff >= interval_months:
                rebalance_dates.append(d)

        return rebalance_dates

    def __repr__(self) -> str:
        universe_size = len(self.universe_identifiers) if self.universe_identifiers else 0
        return (f"IndexDefinition(index_id='{self.index_id}', index_name='{self.index_name}', "
                f"base_date='{self.base_date.strftime('%Y-%m-%d')}', base_value={self.base_value}, "
                f"currency='{self.currency}', rebalancing_frequency='{self.rebalancing_frequency}', "
                f"universe_size={universe_size})")