# Funds and ETFs

`beacon.fund` models a fund that tracks an index. `IndexFund` runs a
[backtest](backtest.md) of its index and reports a NAV net of a management
fee. `ETF` is an `IndexFund` with a ticker, a creation unit size and a
simulated market price. Tracking analytics for either live in
`beacon.analysis`.

## What a fund is made of

A fund holds no trading logic. Its `run_backtest()` builds a `Backtest` and
runs it on the fund's index definition: the index is calculated (or reused
from the cache) and a portfolio is simulated trading to it, exactly as
described in [Backtest](backtest.md). Rebalancing, costs, partial fills,
delistings and price gaps all behave as they do there.

```python
import logging

import pandas as pd

from beacon.fund import ETF
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.portfolio.base import Portfolio
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

fetcher = dataset.data_fetcher()  # frozen sample data, held in memory

definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)

etf = ETF(fund_id="SAMPLE-FUND",
          etf_ticker="SMPL",
          target_index_definition=definition,
          index_agent=IndexCalculator(definition, fetcher),
          portfolio=Portfolio("seed", initial_cash=10_000_000.0),
          data_provider=fetcher,
          management_fee_bps=20)

result = etf.run_backtest(end_date="2024-12-31", transaction_cost_bps=2.0)
print(result.summary()["total_return"])
```

The constructor arguments, and what the fund does with each:

- **`portfolio`** is seed capital only. Its cash balance becomes the
  backtest's initial capital; the fund never trades in it or changes it. The
  simulated holdings are in `result.portfolio`.
- **`index_agent`** is an `IndexCalculator` for the index, of which only
  `price_column` is read. The backtest calculates the index itself from
  `target_index_definition`.
- **`data_provider`** feeds both the calculation and the simulation.
- **`management_fee_bps`** is the annual fee in basis points (20 is 0.20%). It
  must not be negative.

`run_backtest(start_date=None, end_date=..., transaction_cost_bps=0.0)` runs
from the index's base date unless told otherwise, and requires `end_date`.
The trading cost is separate from the management fee. The result is kept on
the fund as `backtest_result`, and the index calculation it tracked as
`index_result`.

The `Backtest` a fund builds uses the defaults for everything the fund does
not pass: the book is kept in **USD** whatever the index's currency, there
are no modifiers and no benchmark, and calculations are cached in the default
location (which only applies when the data comes from a store on disk). To
change any of these, run a `Backtest` yourself.

## NAV and the management fee

`calculate_nav(date)` returns the fund's NAV on a date, net of the management
fee:

1. The **gross NAV** is the backtest's `trading_nav` on the last day on or
   before `date`.
2. The **fee** accrues daily at the annual rate divided by 252, compounded
   over the number of NAV-series days elapsed since the first:

    `net = gross * (1 - fee_bps / 10_000 / 252) ** n`

    where `n` is 0 on the first simulated day, 1 on the next, and so on. `n`
    counts rows of the NAV series (trading sessions), not calendar days.

Before the simulation starts, `calculate_nav` returns the seed cash.

The fee is applied to the NAV when it is read. It is never taken out of the
simulated portfolio, so the backtest result stays gross of it.

```python
last_day = result.trading_nav.index[-1]
days = len(result.trading_nav) - 1

gross = result.trading_nav.iloc[-1]
net = etf.calculate_nav(last_day)

print(round(gross, 2), round(net, 2))
print(round(net / gross, 6), round((1 - 0.0020 / 252) ** days, 6))
```

If `date` is later than the last day of the existing run, `calculate_nav`
runs the backtest again from the index's base date through `date`, and that
run uses **no transaction cost**, whatever the earlier `run_backtest` call
used. It replaces the stored result. Call `run_backtest` with an `end_date`
that covers every date you will ask about to keep your cost assumption.
`rebalance_to_index(date)` does the same extension and nothing else.

## ETF market price

`simulate_market_price(date)` sets `etf.market_price` and returns it. It is
the fee-adjusted NAV from `calculate_nav(date)`: no premium, discount or
bid-ask spread is modelled, and `market_factors` is ignored. It is the whole
fund's value, not a price per share, since the fund does not track shares in
issue. `creation_unit_size` (default 50,000) is stored on the ETF but no
calculation uses it.

```python
price = etf.simulate_market_price(last_day)
print(price == net)
```

## Tracking difference and tracking error

**Tracking difference** is the fund's cumulative return minus the index's
over the period. It is not annualised. **Tracking error** is the annualised
standard deviation of the daily return differences, which measures how
consistently the fund follows the index rather than how far it falls behind.

`etf.get_tracking_performance(result)` returns both as a dict, taken from the
result's `get_tracking_difference()` and `get_tracking_error()`. These compare
the backtest's **gross** NAV with the index, so they include trading costs
and partial fills but not the management fee. They start from the initial
capital, so the opening trade's cost is in them (see
[Reading the result](backtest.md#reading-the-result)). A result that tracked
no index gives `{"error": ...}` instead.

```python
print(etf.get_tracking_performance(result))
```

To measure the fund as an investor holds it, net of the fee, build the net NAV
series and use the functions in `beacon.analysis`:

```python
from beacon.analysis import (
    calculate_premium_discount,
    calculate_tracking_difference,
    calculate_tracking_error,
)

net_nav = pd.Series({day: etf.calculate_nav(day)
                     for day in result.trading_nav.index})
index_levels = result.index.target.levels.reindex(net_nav.index)

fund_returns = net_nav.pct_change().dropna()
index_returns = index_levels.pct_change().dropna()

print("Tracking difference:",
      round(calculate_tracking_difference(fund_returns, index_returns), 6))
print("Tracking error:",
      f"{calculate_tracking_error(fund_returns, index_returns):.2e}")
print("Premium or discount:",
      calculate_premium_discount(etf.market_price, net))
```

Over these two years the fee takes about 0.4% off the NAV. The tracking
difference is larger, about 0.58 percentage points, because it compares
cumulative returns and the fee comes off a NAV that has grown by about 45%.
The tracking error barely changes from the gross figure: a steady fee shifts
every day's return by the same small amount.

The functions and their rules:

| Function | Returns | Notes |
|---|---|---|
| `calculate_tracking_difference(fund_returns, index_returns)` | `prod(1 + r_fund) - prod(1 + r_index)` | The two series must be the same length and are compounded separately, not aligned by date. NaN returns are skipped. |
| `calculate_tracking_error(fund_returns, index_returns, periods_per_year=252)` | Sample standard deviation of the differences times `sqrt(periods_per_year)` | The series are aligned by date and must be the same length. |
| `calculate_premium_discount(market_price, nav)` | `market_price / nav - 1` | Raises on a zero NAV. |

`ETFAnalytics` offers the same three as methods.

See the [API reference](../reference/fund.md) for every parameter.
