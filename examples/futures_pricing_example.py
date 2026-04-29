"""Price an IndexFuture using spot levels from an IndexResult.

This example is intentionally synthetic so it can run without external data.
It demonstrates the integration path:

    IndexResult -> latest index level as spot -> IndexFuture fair value
"""

from __future__ import annotations

import pandas as pd

from beacon.derivatives import IndexFuture, implied_repo_rate
from beacon.index.result import IndexResult


def build_synthetic_index_result() -> IndexResult:
    """Create a small pre-built IndexResult with synthetic index levels."""
    dates = pd.bdate_range("2025-01-02", periods=5)
    index_levels = pd.Series([100.0, 101.2, 100.8, 102.4, 103.0], index=dates)
    divisor_history = pd.Series(1.0, index=dates)
    weights = {dates[0]: {"AAA": 0.6, "BBB": 0.4}}
    constituents = {dates[0]: ["AAA", "BBB"]}

    return IndexResult(
        index_id="synthetic_two_asset_index",
        index_levels=index_levels,
        divisor_history=divisor_history,
        constituent_snapshots=constituents,
        weight_snapshots=weights,
    )


def main() -> None:
    """Run the end-to-end synthetic futures pricing example."""
    # 1. Create or load an IndexResult. A production workflow would usually
    #    produce this from IndexCalculator; this example keeps the data inline.
    index_result = build_synthetic_index_result()

    # 2. Use the latest index level as the spot price for the futures contract.
    valuation_date = index_result.index_levels.index[-1]
    spot_price = float(index_result.index_levels.iloc[-1])

    # 3. Price an IndexFuture with cost-of-carry assumptions.
    risk_free_rate = 0.045
    dividend_yield = 0.015
    borrow_cost = 0.002
    future = IndexFuture(
        derivative_id="SYN-FUT-2026",
        underlying_id=index_result.index_id,
        currency="USD",
        expiry_date="2026-01-02",
        contract_multiplier=50.0,
        tick_size=0.25,
        tick_value=12.5,
    )
    fair_value = future.fair_value(
        spot_price=spot_price,
        valuation_date=valuation_date,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        borrow_cost=borrow_cost,
    )

    # 4. Compare a synthetic market futures price to fair value.
    market_futures_price = fair_value + 0.75
    basis = future.basis(market_futures_price, spot_price)
    repo = implied_repo_rate(
        futures_price=market_futures_price,
        spot=spot_price,
        dividend_yield=dividend_yield,
        time_to_expiry_years=future.time_to_expiry(valuation_date),
    )

    # 5. Print a compact summary table for docs/tutorial usage.
    summary = pd.DataFrame(
        [
            ("valuation_date", valuation_date.date().isoformat()),
            ("index_id", index_result.index_id),
            ("spot_level", f"{spot_price:.4f}"),
            ("future_fair_value", f"{fair_value:.4f}"),
            ("market_futures_price", f"{market_futures_price:.4f}"),
            ("basis", f"{basis:.4f}"),
            ("implied_repo_rate", f"{repo:.4%}"),
        ],
        columns=["metric", "value"],
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
