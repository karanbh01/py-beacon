# src/beacon/derivatives/swaps.py
"""
Swap contracts referencing beacon instruments: TotalReturnSwap.
"""
from typing import Dict
import logging

import pandas as pd

from .base import DerivativeBase

logger = logging.getLogger(__name__)

# Financing legs accrue on an ACT/360 money-market basis.
_FINANCING_DAY_COUNT = 360.0


class TotalReturnSwap(DerivativeBase):
    """A total return swap (TRS) on an index or equity basket.

    The total-return receiver earns the price return of the underlying and pays
    a financing leg. For an ``UNFUNDED`` swap the financing leg is
    ``reference_rate + spread``; for a ``FUNDED`` swap the principal is posted
    up front and only the ``spread`` accrues.

    ``market_data`` inputs (read by key on the valuation methods):

    - ``initial_price`` — reference price ``S_0`` at inception/last reset
      (defaults to *spot_price*, i.e. zero return)
    - ``reference_rate`` — the floating rate for the current period (default 0)
    - ``last_reset_date`` — start of the current accrual period
      (defaults to the swap start date)
    """

    #: Payment frequencies recognised for the financing leg.
    VALID_PAYMENT_FREQUENCIES = frozenset(
        {"MONTHLY", "QUARTERLY", "SEMI-ANNUAL", "ANNUAL"}
    )
    #: Recognised reset/funding types.
    VALID_RESET_TYPES = frozenset({"FUNDED", "UNFUNDED"})

    def __init__(self,
                 derivative_id: str,
                 underlying_id: str,
                 currency: str,
                 start_date: str,
                 end_date: str,
                 notional: float,
                 spread_bps: float,
                 reference_rate: str,
                 payment_frequency: str,
                 reset_type: str = "UNFUNDED"):
        """Initialise a total return swap.

        Args:
            derivative_id: Unique identifier for the swap.
            underlying_id: Identifier of the referenced index/basket.
            currency: Contract currency.
            start_date: Swap start date (YYYY-MM-DD).
            end_date: Swap maturity date (YYYY-MM-DD); used as the base expiry.
            notional: Swap notional; must be positive.
            spread_bps: Financing spread over the reference rate, in basis points.
            reference_rate: Name/identifier of the floating reference rate
                (e.g. ``SOFR``).
            payment_frequency: One of ``MONTHLY``, ``QUARTERLY``,
                ``SEMI-ANNUAL``, ``ANNUAL`` (case-insensitive).
            reset_type: ``UNFUNDED`` (default) or ``FUNDED``.

        Raises:
            ValueError: On empty dates, ``end_date`` not after ``start_date``,
                unrecognised *payment_frequency*/*reset_type*, or the base-class
                validations (including non-positive *notional*).
        """
        if not start_date:
            raise ValueError("start_date cannot be empty.")
        if not end_date:
            raise ValueError("end_date cannot be empty.")

        freq = (payment_frequency or "").upper()
        if freq not in self.VALID_PAYMENT_FREQUENCIES:
            raise ValueError(
                f"payment_frequency must be one of "
                f"{sorted(self.VALID_PAYMENT_FREQUENCIES)}, got '{payment_frequency}'."
            )

        reset = (reset_type or "").upper()
        if reset not in self.VALID_RESET_TYPES:
            raise ValueError(
                f"reset_type must be one of {sorted(self.VALID_RESET_TYPES)}, "
                f"got '{reset_type}'."
            )

        # end_date is the contract expiry from the base class's perspective.
        super().__init__(
            derivative_id=derivative_id,
            underlying_id=underlying_id,
            underlying_type="INDEX",
            currency=currency,
            expiry_date=end_date,
            notional=notional,
        )

        self.start_date: pd.Timestamp = pd.Timestamp(start_date)
        self.end_date: pd.Timestamp = pd.Timestamp(end_date)
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date.")

        self.spread_bps: float = spread_bps
        self.spread: float = spread_bps / 10_000.0
        self.reference_rate: str = reference_rate
        self.payment_frequency: str = freq
        self.reset_type: str = reset

    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    def financing_cost(self,
                       valuation_date: pd.Timestamp,
                       last_reset_date: pd.Timestamp,
                       reference_rate: float) -> float:
        """Financing accrued since *last_reset_date* on an ACT/360 basis.

        For an ``UNFUNDED`` swap the accrual rate is ``reference_rate + spread``;
        for a ``FUNDED`` swap only the ``spread`` accrues.

        Args:
            valuation_date: The accrual end date.
            last_reset_date: Start of the current accrual period.
            reference_rate: Floating reference rate for the period (decimal).

        Returns:
            The accrued financing cost in contract currency.

        Raises:
            ValueError: If *valuation_date* precedes *last_reset_date*.
        """
        days = (pd.Timestamp(valuation_date) - pd.Timestamp(last_reset_date)).days
        if days < 0:
            raise ValueError("valuation_date must be on or after last_reset_date.")

        rate = self.spread
        if self.reset_type == "UNFUNDED":
            rate += reference_rate

        day_count_fraction = days / _FINANCING_DAY_COUNT
        return self.notional * rate * day_count_fraction

    def fair_value(self,
                   spot_price: float,
                   valuation_date: pd.Timestamp,
                   market_data: Dict[str, float]) -> float:
        """Total-return-receiver P&L: total return leg minus accrued financing.

        ``receiver_pnl = notional * (S_t / S_0 - 1) - accrued_financing``
        """
        market_data = market_data or {}
        s0 = market_data.get("initial_price", spot_price)
        if s0 <= 0:
            raise ValueError(f"initial_price must be positive, got {s0}.")

        last_reset = market_data.get("last_reset_date", self.start_date)
        reference_rate = market_data.get("reference_rate", 0.0)

        total_return_leg = self.notional * (spot_price / s0 - 1.0)
        financing = self.financing_cost(valuation_date, last_reset, reference_rate)
        return total_return_leg - financing

    def mark_to_market(self,
                       market_price: float,
                       spot_price: float,
                       valuation_date: pd.Timestamp,
                       market_data: Dict[str, float]) -> Dict[str, float]:
        """Decompose the swap P&L into its legs.

        *market_price* is unused (a TRS has no separately quoted price); it is
        accepted to satisfy the :class:`DerivativeBase` interface.

        Returns a dict with ``total_return_leg``, ``financing_leg``,
        ``net_mtm`` and ``accrued_days``.
        """
        market_data = market_data or {}
        s0 = market_data.get("initial_price", spot_price)
        if s0 <= 0:
            raise ValueError(f"initial_price must be positive, got {s0}.")

        last_reset = pd.Timestamp(market_data.get("last_reset_date", self.start_date))
        reference_rate = market_data.get("reference_rate", 0.0)

        total_return_leg = self.notional * (spot_price / s0 - 1.0)
        financing_leg = self.financing_cost(valuation_date, last_reset, reference_rate)
        accrued_days = (pd.Timestamp(valuation_date) - last_reset).days

        return {
            "total_return_leg": total_return_leg,
            "financing_leg": financing_leg,
            "net_mtm": total_return_leg - financing_leg,
            "accrued_days": accrued_days,
        }
