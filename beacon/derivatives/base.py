# beacon/derivatives/base.py
"""
DerivativeBase — abstract base class for all Delta-1 derivative instruments.
"""
from abc import ABC, abstractmethod
from typing import Dict
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Days per year for the ACT/365 time-to-expiry convention.
_DAYS_PER_YEAR = 365.0
_SECONDS_PER_YEAR = _DAYS_PER_YEAR * 24 * 3600


class DerivativeBase(ABC):
    """Abstract base for Delta-1 derivative instruments.

    Holds the common contract terms (identifiers, currency, expiry, notional)
    and the ACT/365 time-to-expiry helper. Concrete subclasses implement
    :meth:`fair_value` and :meth:`mark_to_market`.
    """

    #: Underlying instrument types recognised by Delta-1 derivatives.
    VALID_UNDERLYING_TYPES = frozenset({"INDEX", "ETF", "EQUITY"})

    def __init__(
        self,
        derivative_id: str,
        underlying_id: str,
        underlying_type: str,
        currency: str,
        expiry_date: str,
        notional: float,
    ):
        """Initialise the common contract terms.

        Args:
            derivative_id: Unique identifier for this derivative.
            underlying_id: Identifier of the referenced underlying.
            underlying_type: One of ``INDEX``, ``ETF`` or ``EQUITY``
                (case-insensitive).
            currency: Contract currency (e.g. ``USD``).
            expiry_date: Expiry date (YYYY-MM-DD).
            notional: Contract notional; must be positive.

        Raises:
            ValueError: If any required field is empty/invalid, the underlying
                type is unrecognised, or *notional* is not positive.
        """
        if not derivative_id:
            raise ValueError("derivative_id cannot be empty.")
        if not underlying_id:
            raise ValueError("underlying_id cannot be empty.")
        if not underlying_type:
            raise ValueError("underlying_type cannot be empty.")
        if not currency:
            raise ValueError("currency cannot be empty.")
        if not expiry_date:
            raise ValueError("expiry_date cannot be empty.")

        underlying_type_norm = underlying_type.upper()
        if underlying_type_norm not in self.VALID_UNDERLYING_TYPES:
            raise ValueError(
                f"underlying_type must be one of "
                f"{sorted(self.VALID_UNDERLYING_TYPES)}, got '{underlying_type}'."
            )

        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}.")

        self.derivative_id: str = derivative_id
        self.underlying_id: str = underlying_id
        self.underlying_type: str = underlying_type_norm
        self.currency: str = currency.upper()
        self.expiry_date: pd.Timestamp = pd.Timestamp(expiry_date)
        self.notional: float = notional

        logger.info(
            f"{type(self).__name__} '{self.derivative_id}' created on "
            f"{self.underlying_type} '{self.underlying_id}', "
            f"expiry {self.expiry_date.strftime('%Y-%m-%d')}."
        )

    def time_to_expiry(self, valuation_date: pd.Timestamp) -> float:
        """Time to expiry in years using the ACT/365 convention.

        Args:
            valuation_date: The date from which to measure.

        Returns:
            Years to expiry, clamped to ``0.0`` once the contract has expired.
        """
        seconds = (self.expiry_date - pd.Timestamp(valuation_date)).total_seconds()
        return max(0.0, seconds / _SECONDS_PER_YEAR)

    @abstractmethod
    def fair_value(
        self,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> float:
        """Return the model fair value of the derivative.

        Args:
            spot_price: Current spot/level of the underlying.
            valuation_date: The valuation date.
            market_data: Additional inputs (e.g. ``risk_free_rate``,
                ``dividend_yield``) keyed by name.

        Returns:
            The fair value in contract currency.
        """
        raise NotImplementedError

    @abstractmethod
    def mark_to_market(
        self,
        market_price: float,
        spot_price: float,
        valuation_date: pd.Timestamp,
        market_data: Dict[str, float],
    ) -> Dict[str, float]:
        """Mark the position to market against an observed *market_price*.

        Args:
            market_price: Observed traded price of the derivative.
            spot_price: Current spot/level of the underlying.
            valuation_date: The valuation date.
            market_data: Additional inputs keyed by name.

        Returns:
            A dictionary of mark-to-market results (e.g. fair value, PnL,
            basis) in contract currency.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(derivative_id='{self.derivative_id}', "
            f"underlying_id='{self.underlying_id}', "
            f"underlying_type='{self.underlying_type}', "
            f"currency='{self.currency}', "
            f"expiry_date='{self.expiry_date.strftime('%Y-%m-%d')}', "
            f"notional={self.notional})"
        )
