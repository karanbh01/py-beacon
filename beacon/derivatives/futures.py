"""Future contract models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .pricing import continuous_carry_forward_price, discrete_dividend_forward_price


@dataclass
class IndexFuture:
    """Index future priced with a continuous dividend-yield model."""

    contract_id: str
    underlying_ticker: str
    maturity_date: Any
    market_data: Dict[str, Any] = field(default_factory=dict)

    def fair_value(self) -> float:
        """Return fair value using continuous cost-of-carry pricing."""
        spot_price = float(self.market_data["spot_price"])
        risk_free_rate = float(self.market_data.get("risk_free_rate", 0.0))
        time_to_maturity = float(self.market_data["time_to_maturity"])
        dividend_yield = float(self.market_data.get("dividend_yield", 0.0))

        return continuous_carry_forward_price(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            time_to_maturity=time_to_maturity,
            dividend_yield=dividend_yield,
        )


@dataclass
class ETFFuture(IndexFuture):
    """ETF future priced with known discrete ETF dividends when available."""

    def fair_value(self) -> float:
        """Return fair value using discrete dividends, falling back to yield.

        ``market_data["discrete_dividends"]`` should be a list of
        ``(time_to_payment, amount)`` pairs where times are in years from the
        valuation date and amounts are cash dividends per ETF share.
        """
        dividends: List[Tuple[float, float]] = self.market_data.get("discrete_dividends") or []
        if not dividends:
            return super().fair_value()

        spot_price = float(self.market_data["spot_price"])
        risk_free_rate = float(self.market_data.get("risk_free_rate", 0.0))
        time_to_maturity = float(self.market_data["time_to_maturity"])

        return discrete_dividend_forward_price(
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
            time_to_maturity=time_to_maturity,
            dividends=dividends,
        )
