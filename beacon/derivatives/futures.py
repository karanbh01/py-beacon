# beacon/derivatives/futures.py
"""
Futures contracts referencing beacon instruments: IndexFuture (on an index)
and ETFFuture (on an ETF).

IndexFuture is implemented here. ETFFuture is implemented in issue BN-44.
"""
from typing import Dict
import logging

import pandas as pd

from .base import DerivativeBase
from .pricing import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    implied_repo_rate,
)

logger = logging.getLogger(__name__)


class IndexFuture(DerivativeBase):
    """A cash-settled futures contract on an equity index.

    Prices are quoted in index points; currency amounts are obtained by
    multiplying by :attr:`contract_multiplier`. Fair value uses the
    cost-of-carry model from :mod:`beacon.derivatives.pricing`.

    Market-data inputs (passed via the ``market_data`` dict on valuation
    methods) are read by key:

    - ``risk_free_rate`` — continuous risk-free rate ``r`` (default 0)
    - ``dividend_yield`` — continuous dividend yield ``q`` (default 0)
    - ``borrow_cost`` — continuous borrow/financing spread ``c`` (default 0)
    """

    def __init__(self,
                 derivative_id: str,
                 underlying_id: str,
                 currency: str,
                 expiry_date: str,
                 contract_multiplier: float,
                 tick_size: float,
                 tick_value: float,
                 underlying_type: str = "INDEX"):
        """Initialise an index future.

        Args:
            derivative_id: Unique identifier for the contract.
            underlying_id: Identifier of the referenced index.
            currency: Contract currency (e.g. ``USD``).
            expiry_date: Expiry date (YYYY-MM-DD).
            contract_multiplier: Currency value of one index point.
            tick_size: Minimum price increment, in index points.
            tick_value: Currency value of one tick.
            underlying_type: Underlying instrument type; defaults to ``INDEX``.
                Subclasses (e.g. :class:`ETFFuture`) override it.

        Raises:
            ValueError: If any of *contract_multiplier*, *tick_size* or
                *tick_value* is non-positive (plus the base-class validations).
        """
        if contract_multiplier <= 0:
            raise ValueError(
                f"contract_multiplier must be positive, got {contract_multiplier}."
            )
        if tick_size <= 0:
            raise ValueError(f"tick_size must be positive, got {tick_size}.")
        if tick_value <= 0:
            raise ValueError(f"tick_value must be positive, got {tick_value}.")

        # The per-point multiplier stands in as the contract notional for the base.
        super().__init__(
            derivative_id=derivative_id,
            underlying_id=underlying_id,
            underlying_type=underlying_type,
            currency=currency,
            expiry_date=expiry_date,
            notional=contract_multiplier,
        )

        self.contract_multiplier: float = contract_multiplier
        self.tick_size: float = tick_size
        self.tick_value: float = tick_value

    # ------------------------------------------------------------------
    # Pricing / analytics
    # ------------------------------------------------------------------

    def fair_value(self,
                   spot_price: float,
                   valuation_date: pd.Timestamp,
                   market_data: Dict[str, float]) -> float:
        """Cost-of-carry fair value ``F = S * exp((r - q + c) * T)`` in points.

        Returns *spot_price* when the contract is at or past expiry (``T == 0``).
        """
        market_data = market_data or {}
        t = self.time_to_expiry(valuation_date)
        return cost_of_carry_fair_value(
            spot=spot_price,
            risk_free_rate=market_data.get("risk_free_rate", 0.0),
            dividend_yield=market_data.get("dividend_yield", 0.0),
            time_to_expiry_years=t,
            borrow_cost=market_data.get("borrow_cost", 0.0),
        )

    def basis(self,
              futures_price: float,
              spot_price: float) -> float:
        """Simple basis: ``futures_price - spot_price`` (index points)."""
        return futures_price - spot_price

    def annualised_basis(self,
                         futures_price: float,
                         spot_price: float,
                         valuation_date: pd.Timestamp) -> float:
        """Annualised implied financing rate ``ln(F / S) / T``.

        Implemented via :func:`implied_repo_rate` with zero dividend yield.

        Raises:
            ValueError: At or past expiry (``T == 0``), where the rate is
                undefined.
        """
        t = self.time_to_expiry(valuation_date)
        return implied_repo_rate(
            futures_price=futures_price,
            spot=spot_price,
            dividend_yield=0.0,
            time_to_expiry_years=t,
        )

    def daily_settlement_pnl(self,
                             settle_today: float,
                             settle_yesterday: float,
                             contracts: float = 1.0) -> float:
        """Variation-margin P&L for the day, in contract currency.

        ``(settle_today - settle_yesterday) * contract_multiplier * contracts``.
        Positive *contracts* is a long position, negative is short.
        """
        return (settle_today - settle_yesterday) * self.contract_multiplier * contracts

    def roll_cost(self,
                  front_price: float,
                  back_price: float) -> float:
        """Cost of rolling from the front to the back contract: ``back - front``.

        Positive in contango (back above front), negative in backwardation.
        """
        return back_price - front_price

    def mark_to_market(self,
                       market_price: float,
                       spot_price: float,
                       valuation_date: pd.Timestamp,
                       market_data: Dict[str, float]) -> Dict[str, float]:
        """Mark the contract against an observed *market_price*.

        Returns a dict with ``fair_value`` (points), ``basis`` (market vs spot),
        ``theoretical_edge`` (fair value minus market price), and
        ``time_to_expiry`` (years).
        """
        fv = self.fair_value(spot_price, valuation_date, market_data)
        return {
            "fair_value": fv,
            "basis": self.basis(market_price, spot_price),
            "theoretical_edge": fv - market_price,
            "time_to_expiry": self.time_to_expiry(valuation_date),
        }


class ETFFuture(IndexFuture):
    """A futures contract on an ETF.

    Behaves like :class:`IndexFuture` but prices with discrete known dividends,
    which better reflects an ETF's periodic cash distributions than a continuous
    yield. When discrete dividends are supplied via the ``market_data`` key
    ``"discrete_dividends"`` (a list of ``(time_to_ex_years, amount)`` tuples for
    ex-dates within the tenor), fair value uses
    ``F = (S - PV(divs)) * exp(r * T)``. Otherwise it falls back to the
    continuous cost-of-carry model inherited from :class:`IndexFuture`.
    """

    def __init__(self,
                 derivative_id: str,
                 underlying_id: str,
                 currency: str,
                 expiry_date: str,
                 contract_multiplier: float,
                 tick_size: float,
                 tick_value: float):
        """Initialise an ETF future. See :class:`IndexFuture` for the args."""
        super().__init__(
            derivative_id=derivative_id,
            underlying_id=underlying_id,
            currency=currency,
            expiry_date=expiry_date,
            contract_multiplier=contract_multiplier,
            tick_size=tick_size,
            tick_value=tick_value,
            underlying_type="ETF",
        )

    def fair_value(self,
                   spot_price: float,
                   valuation_date: pd.Timestamp,
                   market_data: Dict[str, float]) -> float:
        """Discrete-dividend fair value, falling back to continuous carry.

        If ``market_data["discrete_dividends"]`` is present and non-empty, prices
        with the discrete-dividend model; otherwise defers to the continuous
        cost-of-carry model of :class:`IndexFuture`.
        """
        market_data = market_data or {}
        dividends = market_data.get("discrete_dividends")
        if not dividends:
            return super().fair_value(spot_price, valuation_date, market_data)

        t = self.time_to_expiry(valuation_date)
        return discrete_dividend_fair_value(
            spot=spot_price,
            risk_free_rate=market_data.get("risk_free_rate", 0.0),
            time_to_expiry_years=t,
            dividends=dividends,
        )
