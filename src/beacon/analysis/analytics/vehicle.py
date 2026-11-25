# src/beacon/analysis/analytics/vehicle.py
"""
A run's vehicle analytics: the fund as an ETF on the exchange, and the index
futures it held.

    vehicle = result.analytics.portfolio.vehicle
    vehicle.etf_premium()
    vehicle.market_return()                      # None for a fund not listed
"""
# BN-442 (Karan, 2026-11-24): each method is its catalogue statistic, worked
# out as the run's analysis works it out.
from typing import Any

import pandas as pd

from ..panels import MARKET
from ..performance import formulas
from .category import Category, Day


class VehicleCategory(Category):
    """A run's vehicle analytics: the fund as an ETF on the exchange, and
    the index futures it held. Each method is a statistic, named by its id;
    its keywords pick the form it is given in
    (`beacon.analysis.analytics.forms`)."""

    name = "vehicle"
    statistics = ("etf_premium", "futures_trading", "futures_basis", "futures_basis_risk")
    extras = ("market_return", "market_summary")

    # -- etf ------------------------------------------------------------------

    def etf_premium(self,
                    *,
                    start: Day = None,
                    end: Day = None) -> Any:
        """Mean, volatility and extremes of the premium to NAV, and the mean
        spread."""
        return self._get("etf_premium", start=start, end=end)

    def market_return(self) -> float | None:
        """An exchange investor's return on an ETF: bought at the ask on the
        first day and sold at the bid on the last; None for a fund not
        listed. Cash the fund paid out is not included."""
        market = self._market()

        if market is None:
            return None

        return formulas.market_return(float(market["ask"].iloc[0]),
                                      float(market["bid"].iloc[-1]))

    def market_summary(self) -> dict[str, float | None] | None:
        """An ETF's trading on the exchange: the investor's return at market
        prices, the average premium and spread, and the share of days at a
        premium; None for a fund not listed."""
        market = self._market()

        if market is None:
            return None

        return {"market_return": self.market_return(),
                "average_premium": float(market["premium"].mean()),
                "average_spread": float(market["spread"].mean()),
                "days_at_premium": float((market["premium"] > 0).mean())}

    def _market(self) -> pd.DataFrame | None:
        """The ETF's quotes each day; None for a fund not listed."""
        tables = self._analytics.run_data.tables
        market = tables.get(MARKET) if MARKET in tables else None

        return None if market is None or market.empty else market

    # -- futures --------------------------------------------------------------

    def futures_trading(self,
                        *,
                        start: Day = None,
                        end: Day = None) -> Any:
        """Futures trades by reason, contracts and notional, cost per
        contract and in bp, turnover."""
        return self._get("futures_trading", start=start, end=end)

    def futures_basis(self,
                      *,
                      start: Day = None,
                      end: Day = None) -> Any:
        """The futures' price over their index, split into carry and
        mispricing."""
        return self._get("futures_basis", start=start, end=end)

    def futures_basis_risk(self,
                           *,
                           start: Day = None,
                           end: Day = None) -> Any:
        """Tracking error from the pricing basis and from the underlying
        mismatch."""
        return self._get("futures_basis_risk", start=start, end=end)
