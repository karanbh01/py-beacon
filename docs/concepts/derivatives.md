# Derivatives

`beacon.derivatives` prices Delta-1 instruments on indices, ETFs and equities:
index futures, ETF futures and total return swaps, plus the rate curves, term
structures and pure pricing functions behind them. It is part of the core
install and needs no extra.

## Conventions

These hold across the whole package, so they are stated once here.

| What | Convention |
| --- | --- |
| Rates and yields | Continuously compounded, as annual decimals (`0.05` is 5%) |
| Time to expiry | ACT/365: calendar time to expiry divided by 365 days |
| TRS financing | ACT/360: whole calendar days accrued divided by 360 |
| Roll return | Year fraction between two expiries uses 365.25-day years |
| Futures prices | Index points; multiply by the contract multiplier for currency |

Past expiry, time to expiry is clamped to zero, so an index future's fair
value becomes the spot price.

## Index futures

An `IndexFuture` is a cash-settled future on an index. Its fair value is
continuous cost of carry:

`F = S * exp((r - q + c) * T)`

where `r` is the risk-free rate, `q` the dividend yield, `c` a borrow or
financing spread and `T` the time to expiry.

Rates reach the valuation methods through a `market_data` dict, read by the
keys `risk_free_rate`, `dividend_yield` and `borrow_cost`. **A missing key
counts as zero**, so a misspelt key silently prices as if that input were
zero.

```python
import pandas as pd

from beacon.derivatives import IndexFuture

future = IndexFuture(derivative_id="ESZ6",
                     underlying_id="SPX",
                     currency="USD",
                     expiry_date="2026-12-18",
                     contract_multiplier=50.0,
                     tick_size=0.25,
                     tick_value=12.5)

today = pd.Timestamp("2026-09-18")
rates = {"risk_free_rate": 0.045, "dividend_yield": 0.013}

future.time_to_expiry(today)                  # 0.2493 years, ACT/365
future.fair_value(5000.0, today, rates)       # 5040.05 points

marked = future.mark_to_market(5042.0, 5000.0, today, rates)
print(marked)
```

`mark_to_market(market_price, spot_price, valuation_date, market_data)`
returns `fair_value`, `basis` (market minus spot), `theoretical_edge` (fair
value minus market, so negative when the contract trades rich) and
`time_to_expiry`.

The other helpers work in points or contract currency:

```python
future.basis(5042.0, 5000.0)                           # 42.0 points
future.annualised_basis(5042.0, 5000.0, today)         # ln(F / S) / T
future.daily_settlement_pnl(5042.0, 5030.0, contracts=-3)   # -1800.0
future.roll_cost(front_price=5042.0, back_price=5080.0)     # 38.0 points
```

- `annualised_basis` is the implied financing rate with no dividend yield. It
  raises `ValueError` at or past expiry, where the rate is undefined.
- `daily_settlement_pnl` is variation margin:
  `(today - yesterday) * multiplier * contracts`. Negative `contracts` is a
  short position.
- `roll_cost` is `back - front`: positive in contango, negative in
  backwardation.

## ETF futures

`ETFFuture` behaves like `IndexFuture` but can price with discrete cash
dividends, which match an ETF's periodic distributions better than a
continuous yield. Pass them as `discrete_dividends`, a list of
`(time_to_ex_years, amount)` pairs:

`F = (S - PV(dividends)) * exp(r * T)`

Each dividend is discounted at the risk-free rate from its ex-date. Only
dividends with an ex-date between now and expiry count. Without
`discrete_dividends`, the future falls back to continuous cost of carry.

```python
from beacon.derivatives import ETFFuture

etf_future = ETFFuture(derivative_id="SPYZ6",
                       underlying_id="SPY",
                       currency="USD",
                       expiry_date="2026-12-18",
                       contract_multiplier=100.0,
                       tick_size=0.01,
                       tick_value=1.0)

etf_future.fair_value(500.0,
                      today,
                      {"risk_free_rate": 0.045,
                       "discrete_dividends": [(0.08, 1.75)]})   # 503.88
```

## Total return swaps

A `TotalReturnSwap` pays the receiver the underlying's price return, in
exchange for a financing leg.

- **`UNFUNDED`** (the default): financing accrues at `reference_rate + spread`.
- **`FUNDED`**: the principal is paid up front, so only the spread accrues.

Financing accrues ACT/360 on the notional since the last reset. The
underlying type is always `INDEX`, and the payment frequency must be
`MONTHLY`, `QUARTERLY`, `SEMI-ANNUAL` or `ANNUAL`.

```python
from beacon.derivatives import TotalReturnSwap

swap = TotalReturnSwap(derivative_id="TRS-1",
                       underlying_id="CANON",
                       currency="USD",
                       start_date="2026-06-15",
                       end_date="2027-06-15",
                       notional=10_000_000.0,
                       spread_bps=35.0,
                       reference_rate="SOFR",
                       payment_frequency="QUARTERLY")

last_reset = pd.Timestamp("2026-09-15")
swap_data = {"initial_price": 1000.0,
             "reference_rate": 0.043,
             "last_reset_date": last_reset}

swap.financing_cost(today, last_reset, 0.043)   # 3875.0: 3 days at 4.65%
swap.fair_value(1012.0, today, swap_data)       # 116125.0 receiver P&L

legs = swap.mark_to_market(0.0, 1012.0, today, swap_data)
print(legs)
```

`reference_rate` on the constructor is only the rate's name. The rate itself
arrives in `market_data`, with `initial_price` (the level at inception or last
reset) and `last_reset_date`. When they are missing, `initial_price` defaults
to the spot price (zero return), the rate to zero and the reset date to the
start date.

`fair_value` is the receiver's P&L, `notional * (S / S0 - 1)` minus accrued
financing, not a quoted price. `mark_to_market` ignores its `market_price`
argument, because a swap has no quoted price, and returns
`total_return_leg`, `financing_leg`, `net_mtm` and `accrued_days`.

### DV01

```python
swap.dv01(today, last_reset, reference_rate=0.043)    # -8.33
swap.financing_duration(today, last_reset)            # 3 / 360
```

`dv01` is the change in the receiver's value when the reference rate rises
one basis point, found by bumping the rate and revaluing the accrued
financing. It covers the accrual since the last reset only, and equals
`-notional * 0.0001 * financing_duration`. The sign is kept: it is negative
for a receiver on an unfunded swap, because the receiver pays the financing.
A funded swap returns `0.0`, since its spread does not move with the rate.

## Rate curves

A `RateCurve` holds continuously compounded zero rates at pillar tenors, in
years. Between pillars it interpolates linearly in the zero rate. Beyond the
first and last pillars it stays flat at the nearest pillar's rate rather than
extending the slope.

```python
from beacon.derivatives import BASIS_POINT, RateCurve

curve = RateCurve.from_pillars({0.25: 0.043, 1.0: 0.041, 2.0: 0.039})

curve.zero_rate(0.5)              # 0.04233, interpolated
curve.zero_rate(5.0)              # 0.039, flat beyond the last pillar
curve.discount_factor(1.0)        # exp(-0.041 * 1.0)
curve.forward_rate(0.25, 1.0)     # the rate implied between the two tenors

parallel = curve.shifted(BASIS_POINT)                # every pillar +1bp
key_rate = curve.with_pillar_bump(1.0, BASIS_POINT)  # one pillar +1bp
print(key_rate.to_dict())
```

`RateCurve.flat(rate)` returns exactly `rate` at every tenor, so pricing off
a flat curve gives the same answer as passing the scalar rate.
`with_pillar_bump` only accepts an existing pillar and raises
`CalculationError` for any other tenor. Construction also raises
`CalculationError` for an empty curve, mismatched lengths, a negative tenor or
tenors that are not strictly increasing; `from_pillars` sorts them for you.

## Term structure

A `TermStructure` values a strip of futures on one underlying off one curve.
Each expiry is financed at the curve's zero rate for its time to expiry.

```python
from beacon.derivatives import FuturesQuote, TermStructure

strip = TermStructure(underlying="SPX",
                      spot=5000.0,
                      valuation_date=today,
                      quotes=[FuturesQuote(pd.Timestamp("2026-12-18"), 5042.0, "ESZ6"),
                              FuturesQuote(pd.Timestamp("2027-03-19"), 5095.0, "ESH7"),
                              FuturesQuote(pd.Timestamp("2027-06-18"), None, "ESM7")],
                      curve=curve,
                      dividend_yield=0.013)

table = strip.to_frame()
print(table[["label", "theoretical", "market", "basis", "implied_repo"]])
```

`to_frame()` has one row per expiry, nearest first, with `label`,
`time_to_expiry`, `financing_rate`, `theoretical`, `market`, `basis` and
`implied_repo`. `theoretical_prices()`, `market_prices()`, `basis()` and
`implied_repo()` return the same columns as Series indexed by expiry.

Basis and implied repo describe the same disagreement in two ways:

- **`basis`** is market minus theoretical price. Positive means the contract
  trades rich to the model.
- **`implied_repo`** is the financing rate that would make the model match
  the market, `(ln(F / S) + q * T) / T`. The borrow cost is not subtracted, so
  it shows up inside this rate.

A rich contract has a positive basis and an implied repo above the curve, so
the two always agree on direction. A quote with no market price (like `ESM7`
above) gets a theoretical value and `NaN` for both. So does an expiry on the
valuation date itself.

## Sensitivity grids

`sensitivity_grid` revalues continuous cost of carry across a grid of tenors
(rows) and financing rates (columns):

```python
from beacon.derivatives import sensitivity_grid

grid = sensitivity_grid(spot=5000.0,
                        tenors=[0.25, 0.5, 1.0],
                        rates=[0.03, 0.04, 0.05],
                        dividend_yield=0.013)
print(grid.round(2))
```

The index is named `time_to_expiry` and the columns `rate`.

## The pricing functions

The instruments above are built on five stateless functions in
`beacon.derivatives.pricing`. They take plain floats (and Timestamps for the
roll return), import nothing else from py-beacon, and can be checked against
a textbook on their own. Their positional orders differ, so keyword arguments
are the safer way to call them.

| Function | Formula |
| --- | --- |
| `cost_of_carry_fair_value` | `S * exp((r - q + c) * T)` |
| `discrete_dividend_fair_value` | `(S - PV(dividends)) * exp(r * T)` |
| `implied_repo_rate` | `(ln(F / S) + q * T) / T` |
| `futures_roll_return` | `(front / back - 1) / dt`, annualised |
| `trs_breakeven_spread` | `implied_repo_rate(F, S, q, T) - r` |

```python
from beacon.derivatives import (
    cost_of_carry_fair_value,
    discrete_dividend_fair_value,
    futures_roll_return,
    implied_repo_rate,
    trs_breakeven_spread,
)

cost_of_carry_fair_value(spot=5000.0,
                         risk_free_rate=0.045,
                         dividend_yield=0.013,
                         time_to_expiry_years=0.25)

discrete_dividend_fair_value(spot=500.0,
                             risk_free_rate=0.045,
                             time_to_expiry_years=0.25,
                             dividends=[(0.08, 1.75), (0.40, 1.80)])

implied_repo_rate(futures_price=5042.0,
                  spot=5000.0,
                  dividend_yield=0.013,
                  time_to_expiry_years=0.25)

futures_roll_return(front_price=5042.0,
                    back_price=5095.0,
                    front_expiry=pd.Timestamp("2026-12-18"),
                    back_expiry=pd.Timestamp("2027-03-19"))

spread = trs_breakeven_spread(futures_price=5042.0,
                              spot=5000.0,
                              risk_free_rate=0.045,
                              time_to_expiry_years=0.25,
                              dividend_yield=0.013)
print(f"Breakeven TRS spread: {spread * 10_000:.1f}bp")
```

- In `discrete_dividend_fair_value`, the second dividend above goes ex after
  expiry and is ignored.
- `futures_roll_return` is positive in backwardation (front above back) and
  negative in contango, the opposite sign to `IndexFuture.roll_cost`. It
  raises `ValueError` unless the back expiry is strictly after the front.
- `trs_breakeven_spread` is the spread over `r` at which a swap costs the
  same as holding the future. A fairly priced future gives zero.
- `cost_of_carry_fair_value` and `discrete_dividend_fair_value` raise
  `ValueError` for a negative spot or time to expiry; `implied_repo_rate` and
  `trs_breakeven_spread` need all three of price, spot and time to be
  positive.

## On the server

The API server exposes the same pricing under `/derivatives`: futures and TRS
pricing, and an index's futures term structure and roll. See the
[Server guide](../server.md).

The full API is in the [Derivatives reference](../reference/derivatives.md).
