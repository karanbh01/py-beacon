# src/beacon/index/constructor.py
"""
Module for defining the structure and rules of a financial index.
"""
import logging

import pandas as pd

from .calculation.total_return import PRICE, RETURN_TYPES
from .methodology import EligibilityRuleBase, WeightingSchemeBase
from .schedule import (
    DAY_RULES,
    DEFAULT_DAY_RULE,
    next_rebalance,
    rebalance_dates,
)

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
                 eligibility_rules: list[EligibilityRuleBase],
                 weighting_scheme: WeightingSchemeBase,
                 rebalancing_frequency: str, # e.g., 'QUARTERLY', 'MONTHLY', 'SEMI-ANNUAL', 'ANNUAL'
                 description: str | None = None,
                 universe_identifiers: list[str] | None = None,
                 max_constituent_weight: float | None = None,
                 rebalance_day_rule: str = DEFAULT_DAY_RULE,
                 calendar: str | None = None,
                 return_type: str = PRICE,
                 withholding_tax_rate: float = 0.0,
                 effective_lag_sessions: int = 0):
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
                                   (e.g., 'QUARTERLY', 'MONTHLY', 'SEMI-ANNUAL', 'ANNUAL').
                                   More complex schedules (e.g. "Third Friday of March, June...")
                                   would require a more sophisticated scheduler.
            description: Optional textual description of the index.
            universe_identifiers: Optional list of string identifiers (e.g., tickers, ISINs)
                                  defining the asset universe from which constituents are selected.
            max_constituent_weight: Optional cap on any single constituent's
                                  weight, as a fraction (0.1 is 10%). Applied
                                  after the weighting scheme and iterated until
                                  no constituent breaches it. None means
                                  uncapped.
            rebalance_day_rule: Which day of a scheduled month the rebalance
                                  falls on. Defaults to the first business day,
                                  which is what every index defined before
                                  BN-121 used.
            calendar: Exchange MIC backing trading-day arithmetic, e.g.
                                  ``"XNYS"``. None means Monday to Friday, again
                                  the previous behaviour. Naming one requires
                                  the `calendars` extra — an index that declares
                                  a calendar must not quietly compute against a
                                  different one.
            return_type: PRICE, TOTAL_RETURN or NET_TOTAL_RETURN. PRICE is the
                                  default and the behaviour of every index
                                  defined before BN-125; the other two reinvest
                                  cash distributions across the index.
            withholding_tax_rate: Fraction of each distribution withheld, for a
                                  net index. Ignored unless the return type is
                                  NET_TOTAL_RETURN, so a definition carrying a
                                  rate it does not use cannot quietly apply it.
            effective_lag_sessions: Sessions between a composition being
                                  announced and its weights taking effect. Zero
                                  is same-day, which is what every index did
                                  before BN-126.
        """
        if not index_id:
            raise ValueError("index_id cannot be empty.")
        if not index_name:
            raise ValueError("index_name cannot be empty.")
        if not base_date:
            raise ValueError("base_date cannot be empty.")
        if rebalance_day_rule not in DAY_RULES:
            raise ValueError(
                f"Unsupported day rule: '{rebalance_day_rule}'. "
                f"Supported: {', '.join(DAY_RULES)}.")
        if return_type not in RETURN_TYPES:
            raise ValueError(
                f"Unsupported return type: '{return_type}'. "
                f"Supported: {', '.join(RETURN_TYPES)}.")
        if not 0.0 <= withholding_tax_rate < 1.0:
            raise ValueError(
                "withholding_tax_rate must be in [0, 1); got "
                f"{withholding_tax_rate}.")
        if effective_lag_sessions < 0:
            raise ValueError(
                "effective_lag_sessions cannot be negative; got "
                f"{effective_lag_sessions}.")
        if base_value <= 0:
            raise ValueError("base_value must be positive.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if not weighting_scheme:
            raise ValueError("weighting_scheme must be provided.")
        if not rebalancing_frequency:
            raise ValueError("rebalancing_frequency cannot be empty.")

        if not eligibility_rules:
            logger.warning(f"Index '{index_name}' defined with no eligibility rules.")

        if universe_identifiers is not None and not universe_identifiers:
            raise ValueError("universe_identifiers, when provided, must be a non-empty list.")

        if max_constituent_weight is not None and not 0.0 < max_constituent_weight <= 1.0:
            raise ValueError(
                "max_constituent_weight, when provided, must be in (0, 1]; got "
                f"{max_constituent_weight}.")

        self.index_id: str = index_id
        self.index_name: str = index_name
        self.base_date: pd.Timestamp = pd.Timestamp(base_date)
        self.base_value: float = base_value
        self.currency: str = currency.upper()
        self.eligibility_rules: list[EligibilityRuleBase] = eligibility_rules
        self.weighting_scheme: WeightingSchemeBase = weighting_scheme
        self.rebalancing_frequency: str = rebalancing_frequency.upper()
        self.description: str | None = description
        self.universe_identifiers: list[str] | None = universe_identifiers
        self.max_constituent_weight: float | None = max_constituent_weight
        self.rebalance_day_rule: str = rebalance_day_rule
        self.calendar: str | None = calendar
        self.return_type: str = return_type
        self.withholding_tax_rate: float = withholding_tax_rate
        self.effective_lag_sessions: int = effective_lag_sessions

        logger.info(
            f"IndexDefinition for '{self.index_name}' ({self.index_id}) created successfully.")

    def get_rebalance_dates(self,
                            start_date: str,
                            end_date: str) -> list[pd.Timestamp]:
        """
        Return all rebalance dates within [start_date, end_date] based on
        the index's rebalancing frequency, day rule and calendar.

        Delegates to `beacon.index.schedule`, which replaced the first-business-
        day-of-month assumption this method used to hard-code. An index that
        names neither a day rule nor a calendar gets exactly the dates it always
        did — pinned by a test, because changing them would silently redate
        every stored backtest.

        Args:
            start_date: Start of the range (YYYY-MM-DD), inclusive.
            end_date: End of the range (YYYY-MM-DD), inclusive.

        Returns:
            A chronologically sorted list of business-day-adjusted rebalance dates.

        Raises:
            ValueError: If the rebalancing frequency is unsupported.
        """
        return rebalance_dates(self.rebalancing_frequency,
                               start_date,
                               end_date,
                               self.rebalance_day_rule,
                               self.calendar)

    def next_rebalance(self,
                       as_of: str) -> pd.Timestamp | None:
        """The first rebalance strictly after a date.

        Anchored on the base date, like every other date this class produces,
        so the answer names a day the index would genuinely rebalance on.

        Args:
            as_of: The date being asked from, YYYY-MM-DD.

        Returns:
            The date, or None if none falls within the lookahead window.
        """
        return next_rebalance(self.rebalancing_frequency,
                              self.base_date,
                              as_of,
                              self.rebalance_day_rule,
                              self.calendar)

    def __repr__(self) -> str:
        universe_size = len(self.universe_identifiers) if self.universe_identifiers else 0
        return (f"IndexDefinition(index_id='{self.index_id}', index_name='{self.index_name}', "
                f"base_date='{self.base_date.strftime('%Y-%m-%d')}', base_value={self.base_value}, "
                f"currency='{self.currency}', "
                f"rebalancing_frequency='{self.rebalancing_frequency}', "
                f"universe_size={universe_size})")
