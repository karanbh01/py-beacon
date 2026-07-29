"""
Example: pricing an IndexFuture off an IndexResult.

This script demonstrates how the index-calculation module and the derivatives
module connect end to end:

    IndexCalculator.run()  ->  IndexResult  ->  spot level
                                                  |
                                                  v
                          IndexFuture.fair_value() / basis / implied repo

It builds a small equal-weighted index from synthetic prices (no external data
or network access), takes the latest index level as the spot, and prices a
3-month index future using the cost-of-carry model. Run it directly:

    python examples/futures_pricing_example.py
"""
import logging

import pandas as pd

# Keep the example output focused on the summary table.
logging.getLogger("beacon").setLevel(logging.ERROR)

from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.calculation import IndexCalculator
from beacon.derivatives.futures import IndexFuture


# ---------------------------------------------------------------------------
# 1. Synthetic market data
# ---------------------------------------------------------------------------
# The IndexCalculator expects a data provider exposing fetch_reference_data,
# fetch_prices and fetch_shares_outstanding. We supply a tiny in-memory provider
# so the example is fully self-contained.

ASSETS = ["ASSET_A", "ASSET_B"]
BASE_DATE = "2024-01-02"
END_DATE = "2024-03-29"   # ~3 months of business days


def _build_price_map():
    """Deterministic daily prices: ASSET_A +10%, ASSET_B +20% over the window."""
    trading_days = pd.bdate_range(start=BASE_DATE, end=END_DATE)
    n = len(trading_days)
    prices = {}
    for i, day in enumerate(trading_days):
        frac = i / (n - 1)
        date_str = day.strftime("%Y-%m-%d")
        prices[("ASSET_A", date_str)] = 100.0 * (1.10 ** frac)
        prices[("ASSET_B", date_str)] = 50.0 * (1.20 ** frac)
    return prices


class SyntheticDataProvider:
    """Minimal data provider satisfying the IndexCalculator interface."""

    def __init__(self,
                 price_map):
        self._prices = price_map

    def fetch_reference_data(self,
                             identifier,
                             date_str=None):
        if identifier in ASSETS:
            return pd.DataFrame(
                {"NAME": [identifier], "CURRENCY": ["USD"], "EXCHANGE": ["NYSE"]},
                index=pd.Index([identifier], name="IDENTIFIER"),
            )
        return pd.DataFrame()

    def fetch_market_data(self,
                          identifier,
                          start=None,
                          end=None,
                          columns=None):
        price = self._prices.get((identifier, start))
        if price is None:
            return pd.DataFrame()
        return pd.DataFrame(
            {"CLOSE": [price]},
            index=pd.Index([pd.Timestamp(start)], name="DATE"),
        )

    def fetch_shares_outstanding(self,
                                 ticker,
                                 date_str):
        # Equal shares for both names — keeps the synthetic index simple.
        return 1_000


# ---------------------------------------------------------------------------
# 2. Build the index and run the calculator -> IndexResult
# ---------------------------------------------------------------------------

def build_index_result():
    """Construct a simple equal-weighted index and return its IndexResult."""
    definition = IndexDefinition(
        index_id="EXMPL_EW",
        index_name="Example Equal Weight",
        base_date=BASE_DATE,
        base_value=1000.0,
        currency="USD",
        eligibility_rules=[],          # pass-all universe
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=ASSETS,
    )
    calculator = IndexCalculator(definition, SyntheticDataProvider(_build_price_map()))
    return calculator.run(end_date=END_DATE)


# ---------------------------------------------------------------------------
# 3-5. Price the future, analyse basis / implied repo, print a summary
# ---------------------------------------------------------------------------

def main():
    result = build_index_result()

    # Use the latest index level as the spot price for the future.
    valuation_date = result.index_levels.index[-1]
    spot = float(result.index_levels.iloc[-1])

    # Define a 3-month future on the index. contract_multiplier is the currency
    # value of one index point (e.g. $10 per point here).
    expiry = (valuation_date + pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    future = IndexFuture(
        derivative_id="EXMPL_FUT",
        underlying_id=result.index_id,
        currency="USD",
        expiry_date=expiry,
        contract_multiplier=10.0,
        tick_size=0.25,
        tick_value=2.5,
    )

    # Cost-of-carry inputs: 5% risk-free, 2% dividend yield.
    market_data = {"risk_free_rate": 0.05, "dividend_yield": 0.02}

    # Theoretical fair value from cost of carry: F = S * exp((r - q + c) * T).
    fair_value = future.fair_value(spot, valuation_date, market_data)

    # Suppose the future trades slightly rich to fair value in the market.
    market_futures_price = fair_value + 3.0

    # Basis analysis against the observed market price.
    basis = future.basis(market_futures_price, spot)
    implied_repo = future.annualised_basis(market_futures_price, spot, valuation_date)
    ttm = future.time_to_expiry(valuation_date)

    mtm = future.mark_to_market(market_futures_price, spot, valuation_date, market_data)

    # ----- Summary table -----------------------------------------------------
    print("=" * 56)
    print("  IndexResult -> IndexFuture pricing summary")
    print("=" * 56)
    rows = [
        ("Index", result.index_id),
        ("Valuation date", valuation_date.strftime("%Y-%m-%d")),
        ("Spot (latest level)", f"{spot:,.4f}"),
        ("Future expiry", expiry),
        ("Time to expiry (yrs)", f"{ttm:.4f}"),
        ("Risk-free rate", f"{market_data['risk_free_rate']:.2%}"),
        ("Dividend yield", f"{market_data['dividend_yield']:.2%}"),
        ("Fair value (cost-of-carry)", f"{fair_value:,.4f}"),
        ("Market futures price", f"{market_futures_price:,.4f}"),
        ("Basis (market - spot)", f"{basis:,.4f}"),
        ("Annualised basis / implied repo", f"{implied_repo:.4%}"),
        ("Theoretical edge (fair - market)", f"{mtm['theoretical_edge']:,.4f}"),
    ]
    for label, value in rows:
        print(f"  {label:<34} {value:>18}")
    print("=" * 56)


if __name__ == "__main__":
    main()
