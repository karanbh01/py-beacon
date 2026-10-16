---
title: Funds and ETFs
description: "Fund, a fund product with share classes, one strategy and one vehicle, and tracking it against its index."
---

# Funds and ETFs

A `Fund` is a fund product: a name, a currency, its share classes and
documents, with one strategy (what it holds) and one
[vehicle](fund-vehicles.md) (how the money is held). Its `backtest()` runs a
[backtest](backtest.md) with the fund's vehicle and share classes on its
strategy.

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

result = fund.backtest(end="2024-12-31",
                       initial_capital={"Acc": 30_000_000.0, "Dist": 10_000_000.0,
                                        "I Acc": 10_000_000.0},
                       flows={"Acc": PeriodicFlows(fraction=0.01)},
                       transaction_cost_bps=2.0, data_provider=fetcher)

for name, share_class in result.share_classes.items():
    print(name, round(share_class.total_return(), 6),
          round(share_class.net_assets.iloc[-1]))
```

Each share class has its own management fee (the vehicle's when unset) and
distribution policy: an accumulating class keeps its income in the fund and a
distributing class pays it out. The classes share one pool of assets, as a
real fund's do: `fund.backtest()` runs the fund once with every class, each
with its own capital and flows (by class name; one number for a fund of one
class), and their money trades one book, so capacity caps, market impact and
execution limits see the whole fund.

Each day the book's gains, losses and income are split between the classes
in proportion to their net assets at the previous close. Then what belongs to
one class reaches it alone: its fee, accrued on its own net assets; its flows,
dealt at its own NAV per unit (moved by the vehicle's swing or levy, which the
fund's net flow sets); and, for a distributing class, its share of the
dividends, paid out. `result.share_classes` holds each class's book: its net
assets, units, NAV per unit, fees, distributions and flows, with
`get_returns()` and `total_return()` per unit, distributions added back. The
result itself is the whole fund's.

The same works without a `Fund`: pass `share_classes=` to `Backtest`, with
`initial_capital` and `flows` by class name.

### Classes in another currency, hedged or not

A class can be denominated in another currency from the fund's book:
`ShareClass("EUR", currency="EUR")`. Its capital, flows, fees and distributions
are in its own currency, converted at each day's rate, and its NAV per unit
moves with the book's return and the exchange rate together. Its
`book_net_assets` gives its net assets in the book's currency, so the classes
still add up to the fund.

A hedged class takes the currency move out:
`ShareClass("EUR H", currency="EUR", hedged=True)`. It sells the book's currency
forward for its own, in the amount of its net assets times `hedge_ratio` (1 by
default):

- **Pricing.** Each forward is priced at covered interest parity from the
  day's spot rate and the two currencies' short rates, each on its own day
  count. A currency's short rate comes from the data (see
  [Short rates](data.md#short-rates)); without one, the run's cash rate is used
  and a warning says so.
- **Rolls.** A one-month forward is opened at the first session and rolled on
  the first business day of each month. Between rolls it is resized when its
  size leaves the hedge ratio by more than `hedge_band` (3% by default; `None`
  turns resizing off).
- **Settlement.** A roll or resize settles the old forward in cash, as a
  `"HEDGE"` cash flow. A loss is paid by selling across the book and a gain is
  invested across it, so the other classes are not left holding the hedged
  class's cash.

The forward's gains and losses are the hedged class's alone, and its open
value is part of its net assets and of the fund's NAV
(`result.hedge_value`). A hedged class earns the book's return plus the
interest rate differential: a euro class hedged out of dollars at 3% against
5% gives up about 2% a year. `share_classes[name].hedges` lists each trade
with its reason (`"open"`, `"roll"` or `"band"`), and `hedge_value` its daily
value. A class hedged into the book's own currency is refused.

A class is measured against the index in its own currency.
`index_in_class_currency(levels, "USD", "EUR", data)` in
`beacon.backtest.class_benchmark` converts the index at each day's rate, for
an unhedged class. With `hedge_ratio=` it hedges the index the way the class
hedges: at the first session of each month the index's value is sold forward
for the class's currency, at covered interest parity from the same short
rates, at the class's hedge ratio. `class_tracking(book, benchmark)` gives
the class's total return, tracking difference and tracking error against it,
defined as a backtest's are. Against the hedged index, a hedged class's
tracking difference is its fees, costs and hedge slippage, not the 2% a year
the euro class above gives up.

```python
from beacon.backtest.class_benchmark import class_tracking, index_in_class_currency

book = result.share_classes["EUR H"]
benchmark = index_in_class_currency(result.index.tracked.levels, "USD", "EUR",
                                    data, hedge_ratio=1.0)
class_tracking(book, benchmark)  # total_return, tracking_difference, tracking_error
```

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
