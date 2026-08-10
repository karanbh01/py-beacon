# examples/03_index_futures.py
"""
Build an index, then price a future on it.

    python examples/03_index_futures.py

Covers fair value from cost of carry, basis against a quoted price, implied
repo, DV01 by bump-and-revalue, and the cost of rolling to the next contract.

## The idea in one line

A future is a claim on the index at expiry, so its fair value is the index
compounded at the financing rate and reduced by the dividends you forgo by not
holding the constituents:

    F = S x exp((r - q) x T)

Carry is `r - q`. When financing exceeds the dividend yield the future trades
*above* spot; when the yield exceeds financing it trades below. Nothing
mysterious happens at expiry: `T` goes to zero and the future converges on
spot.
"""
import pandas as pd
from _shared import heading, index_definition, market_data, parse_arguments

from beacon.derivatives import IndexFuture
from beacon.derivatives.pricing import futures_roll_return, implied_repo_rate
from beacon.index.calculation import IndexCalculator

RISK_FREE = 0.045
DIVIDEND_YIELD = 0.018

# One index point is 50 currency units, the convention for a major equity
# future. Tick size and value follow from it.
MULTIPLIER = 50.0
TICK_SIZE = 0.25
TICK_VALUE = MULTIPLIER * TICK_SIZE

# A basis point, for the DV01 bump.
ONE_BASIS_POINT = 0.0001


def main() -> int:
    arguments = parse_arguments(__doc__ or "")
    dataset, fetcher, config = market_data(arguments.full)

    definition = index_definition(dataset, config, index_id="FUTIDX", cap=0.10)
    index = IndexCalculator(definition, fetcher).run(
        start_date=config.start, end_date=config.end)

    spot = float(index.index_levels.iloc[-1])
    valuation_date = pd.Timestamp(index.index_levels.index[-1])
    expiry = valuation_date + pd.DateOffset(months=3)

    future = IndexFuture(derivative_id="FUTIDX-Z",
                         underlying_id=definition.index_id,
                         currency=config.currency,
                         expiry_date=expiry.strftime("%Y-%m-%d"),
                         contract_multiplier=MULTIPLIER,
                         tick_size=TICK_SIZE,
                         tick_value=TICK_VALUE)

    market = {"risk_free_rate": RISK_FREE, "dividend_yield": DIVIDEND_YIELD}

    heading("Contract")
    print(f"  underlying       {future.underlying_id} at {spot:,.2f}")
    print(f"  valuation        {valuation_date.date()}")
    print(f"  expiry           {expiry.date()}")
    print(f"  time to expiry   {future.time_to_expiry(valuation_date):.4f} years"
          f"   (ACT/365)")
    print(f"  multiplier       {MULTIPLIER:,.0f} per index point")

    # --- Fair value ------------------------------------------------------
    fair = future.fair_value(spot, valuation_date, market)
    carry = RISK_FREE - DIVIDEND_YIELD

    heading("Fair value")
    print(f"  risk free        {RISK_FREE:>8.2%}")
    print(f"  dividend yield   {DIVIDEND_YIELD:>8.2%}")
    print(f"  net carry        {carry:>8.2%}")
    print(f"  spot             {spot:>10,.2f}")
    print(f"  fair value       {fair:>10,.2f}")
    print(f"  premium          {fair - spot:>+10,.2f}"
          f"   ({fair / spot - 1:+.3%})")
    print()
    print("  Financing costs more than the dividends forgone, so the future")
    print("  sits above spot. Reverse the two and it would sit below.")

    # --- Basis against a quote -------------------------------------------
    quoted = fair * 1.0015

    heading("Basis")
    print(f"  quoted           {quoted:>10,.2f}")
    print(f"  fair             {fair:>10,.2f}")
    print(f"  basis to spot    {future.basis(quoted, spot):>+10,.2f}")
    print(f"  rich/cheap       {quoted - fair:>+10,.2f}"
          f"   versus fair value")
    print(f"  annualised basis "
          f"{future.annualised_basis(quoted, spot, valuation_date):>+9.3%}")

    implied = implied_repo_rate(
        futures_price=quoted,
        spot=spot,
        dividend_yield=DIVIDEND_YIELD,
        time_to_expiry_years=future.time_to_expiry(valuation_date))

    print(f"\n  implied repo     {implied:>9.3%}")
    print(f"  actual financing {RISK_FREE:>9.3%}")
    print()
    print("  Implied repo is the financing rate the quoted price implies. Above")
    print("  your actual funding cost, a cash-and-carry trade is profitable;")
    print("  below it, the reverse. It is the basis expressed as a rate, which")
    print("  is what makes two contracts of different maturities comparable.")

    # --- DV01 ------------------------------------------------------------
    bumped = future.fair_value(spot, valuation_date,
                               {**market, "risk_free_rate": RISK_FREE + ONE_BASIS_POINT})
    dv01 = (bumped - fair) * MULTIPLIER

    heading("Rate sensitivity")
    print(f"  fair value       {fair:>10,.4f}")
    print(f"  +1bp financing   {bumped:>10,.4f}")
    print(f"  DV01             {dv01:>+10,.2f}   per contract")
    print()
    print("  Computed by bumping and revaluing rather than differentiating.")
    print("  It is one extra call to the same function, so the number cannot")
    print("  drift from the pricer the way a hand-derived formula can.")

    # --- Roll ------------------------------------------------------------
    far_expiry = expiry + pd.DateOffset(months=3)
    far = IndexFuture(derivative_id="FUTIDX-H",
                      underlying_id=definition.index_id,
                      currency=config.currency,
                      expiry_date=far_expiry.strftime("%Y-%m-%d"),
                      contract_multiplier=MULTIPLIER,
                      tick_size=TICK_SIZE,
                      tick_value=TICK_VALUE)

    back_fair = far.fair_value(spot, valuation_date, market)

    heading("Rolling to the next contract")
    print(f"  front  ({expiry.date()})   {fair:>10,.2f}")
    print(f"  back   ({far_expiry.date()})   {back_fair:>10,.2f}")
    print(f"  roll cost        {future.roll_cost(fair, back_fair):>+10,.2f}"
          f"   per index point")
    print(f"  per contract     "
          f"{future.roll_cost(fair, back_fair) * MULTIPLIER:>+10,.2f}")
    annualised_roll = futures_roll_return(fair, back_fair, expiry, far_expiry)
    print(f"  annualised roll  {annualised_roll:>+10.3%}")
    print()
    print("  Positive carry makes the back contract dearer, so a long position")
    print("  pays to roll. That recurring cost is why a futures-based tracker")
    print("  drifts from the index it follows even when it never mis-trades.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
