---
title: Funds and ETFs
description: "Fund, a fund product with share classes, one strategy and one vehicle, and tracking it against its index."
---

# Funds and ETFs

A `Fund` is a fund product: a name, a currency, its share classes and
documents, with one strategy (what it holds) and one
[vehicle](fund-vehicles.md) (how the money is held). Its `backtest()` runs a
[backtest](backtest.md) with the fund's vehicle on its strategy.

```python
import logging

from beacon.backtest import PeriodicFlows, ucits_etf
from beacon.fund import Fund, ShareClass
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
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

fund = Fund(name="Sample UCITS ETF", strategy=definition,
            vehicle=ucits_etf(management_fee_bps=12),
            share_classes=[ShareClass("Acc"),
                           ShareClass("Dist", distribution="distributing"),
                           ShareClass("I Acc", management_fee_bps=7)],
            documents={"prospectus": "https://example.com/prospectus.pdf"})

for share_class in fund.share_classes:
    result = fund.backtest(end="2024-12-31", initial_capital=50_000_000.0,
                           share_class=share_class.name, transaction_cost_bps=2.0,
                           flows=PeriodicFlows(fraction=0.01),
                           data_provider=fetcher)
    print(share_class.name, round(result.summary()["total_return"], 6))
```

Each share class has its own management fee (the vehicle's when unset) and
distribution policy: an accumulating class reinvests its income and a
distributing class pays it out. `fund.backtest(share_class=...)` runs the
fund once for that class. A real fund's classes share one pool of assets, so
their flows trade one book and share its capacity and costs; a run per class
leaves that out.

## Tracking a fund against its index

A fund's backtest is measured against the index it tracked.
`get_tracking_difference()` is the run's cumulative return minus the index's
over the period, not annualised. `get_tracking_error()` is the annualised
standard deviation of the daily return differences, which measures how
consistently the fund follows the index rather than how far it falls behind.

```python
print("Tracking difference:", round(result.get_tracking_difference(), 6))
print("Tracking error:", f"{result.get_tracking_error():.2e}")
```

The functions in `beacon.analysis` measure any pair of return series, such as
the fund's NAV per unit, which is net of its fees and costs, against the
index, and an exchange-traded fund's market price against its NAV:

```python
from beacon.analysis import (
    calculate_premium_discount,
    calculate_tracking_difference,
    calculate_tracking_error,
)

nav = result.nav_per_unit
index_levels = result.index.target.levels.reindex(nav.index)

fund_returns = nav.pct_change().dropna()
index_returns = index_levels.pct_change().dropna()

print("Tracking difference:",
      round(calculate_tracking_difference(fund_returns, index_returns), 6))
print("Tracking error:",
      f"{calculate_tracking_error(fund_returns, index_returns):.2e}")
print("Premium or discount:",
      round(calculate_premium_discount(result.market["market_price"].iloc[-1],
                                       nav.iloc[-1]), 6))
```

The functions and their rules:

| Function | Returns | Notes |
|---|---|---|
| `calculate_tracking_difference(fund_returns, index_returns)` | `prod(1 + r_fund) - prod(1 + r_index)` | The two series must be the same length and are compounded separately, not aligned by date. NaN returns are skipped. |
| `calculate_tracking_error(fund_returns, index_returns, periods_per_year=252)` | Sample standard deviation of the differences times `sqrt(periods_per_year)` | The series are aligned by date and must be the same length. |
| `calculate_premium_discount(market_price, nav)` | `market_price / nav - 1` | Raises on a zero NAV. |

`ETFAnalytics` offers the same three as methods.

See the [API reference](../reference/fund.md) for every parameter.
