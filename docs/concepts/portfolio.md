---
title: Portfolio
description: "What a run held, traded and paid: weights, groups, turnover, costs, characteristics, flows and the vehicle's own figures."
---

# Portfolio

[Performance](performance.md) says how a run did. `PortfolioAnalytics` says
what it held, how much it traded and why, and what that cost: its weights and
groups, its turnover, its costs, the characteristics of what it held, its
income and flows, and the figures particular to an ETF, a fund with share
classes or one holding index futures.

A finished backtest carries it as `result.portfolio_analytics`. For a stored
run, build it from the run's data: `PortfolioAnalytics(RunData.from_record(folder))`.
It reads the run's tables (`result.run_data.tables`), which are built the
first time any is asked for.

## Weights and groups

```python
import logging

from beacon.backtest import Backtest
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)
backtest = Backtest(initial_capital=1_000_000.0, transaction_cost_bps=5.0,
                    data_provider=dataset.data_fetcher())
result = backtest.run(definition, start="2023-01-03", end="2024-12-31")

portfolio = result.portfolio_analytics
print(portfolio.weights().tail(1))          # dates by name
print(portfolio.holdings_count().iloc[-1])  # 5
print(portfolio.group_weights("SECTOR").tail(1))
```

`weights(book)` is a portfolio's daily weights, dates by name, for `fund`,
`index` or `benchmark`. It is None for a portfolio the run kept no weights
for, such as a benchmark given as a level series. `holdings_count(book)` is
how many names it held each day.

`group_weights(field, book)` adds the weights up by a grouping field of the
run's reference data: `SECTOR`, `CURRENCY` and `EXCHANGE` in the sample, and
whatever else the data gives. A name's group on a day is the one it had at
the latest rebalance on or before that day, and a name with no group recorded
is in `Unknown`. **The groups add up to the invested weight, not to 1**: the
fund's cash is in none of them. A field the run kept no reference data for
raises a `KeyError` naming the fields it has.

`active_group_weights(field, against)` is the fund's weight in each group
less the `index`'s or `benchmark`'s, on the fund's days. The sample tracks its
index, so its active weights are near zero.

```python
print(portfolio.active_group_weights("SECTOR").abs().max().max())   # about 2e-8
```

## Turnover and trading

A market-cap index's weights move with prices, so the sample does not trade
again after buying in. An equal-weighted index over the same names rebalances back
to equal weights each quarter, which gives turnover to look at.

```python
from beacon.index.methodology import EqualWeighted

equal = IndexDefinition(
    index_id="EQUAL", index_name="Sample Equal-Weighted Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=EqualWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)
trading = backtest.run(equal, start="2023-01-03", end="2024-12-31").portfolio_analytics

print(trading.turnover("rebalance")[["one_way", "two_way"]].round(4))
print(trading.turnover("year")[["one_way", "two_way"]].round(4))
print(trading.trading_by_reason())
```

`turnover(by)` gives the buys, sells, number of trades, mean trade, and
one-way and two-way turnover, by `day` (every trading day, with whether it
was a rebalance), `rebalance` (each rebalance's own trades) or `year`.

- **One-way turnover** is the lesser of what was bought and what was sold,
  over the fund's assets: the SEC's measure. Money arriving or leaving is
  bought or sold on one side only, so it does not inflate it.
- **Two-way turnover** is everything bought and sold over the same assets.

A day's and a rebalance's are over the NAV at the close before (the initial
capital on the first day); a year's over its mean daily NAV. So the opening
purchase is about 100% two-way turnover on the first day and in the first
year, and 0 one-way.

### Why each trade was made

Each trade has **one reason**, the event that triggered it, and **a set of
tags**: the reason, what it implies, and anything else that shaped it. The
run's trades table (`result.run_data.tables["trades"]`) has both, as `REASON`
and `TAGS` (joined by `"; "`).

| Reason | The trade was made |
| --- | --- |
| `rebalance` | At a rebalance, toward the new target |
| `subscription`, `redemption` | To invest money arriving, or raise money leaving |
| `cash raise`, `cash spend-down` | By the cash band, selling back up to the buffer or investing the excess |
| `dividend reinvestment` | To reinvest a dividend |
| `settlement` | To invest cash a derivative paid, or raise cash one is owed: a swap's value, a currency hedge's settlement, futures margin |
| `corporate action` | To carry out a corporate action |
| `completion` | To work an order an execution limit left unfilled |

A rebalance trade is also tagged with any stage that changed the name's
target: `diversification` (the index's cap or the vehicle's limits),
`liquidity`, `exclusion` (a screen), `position limit` (the smallest position
allowed); and `model-driven` for an active strategy. Any trade whose reason
moves cash is tagged `cash raise` when it sells and `cash spend-down` when it
buys, whatever moved the cash. `beacon.portfolio.reasons` names a few more
(`index change`, `drift`, `discretionary`, `risk-driven` and others) that no
part of the engine records yet.

**Figures by reason add up to the run's; figures by tag may count a trade
more than once.** `trading_by_reason()` gives each reason's trades, buys,
sells and share of all traded. `cash_trading()` gives every trade tagged a
cash raise or spend-down, by tag and reason.

## Costs

```python
print(trading.costs())          # over the run, with basis points a year
print(trading.cost_per_traded())
print(trading.costs_by_reason())
```

`costs(by)` gives the costs by kind, `fixed` (the fixed cost of each trade),
`impact` (its market impact), `futures_rolls` and `futures_trading`, then the
`creation_fees_received` and the `net`. **Creation fees are received by the
fund, so they are set against its costs**: the net is what was paid less
them.

- Over the `run`, each kind's `total` and `bps_a_year`: the total over the
  mean NAV times the years, the years being the NAV's days over
  `periods_per_year`.
- Each `year`, each kind's total and `net_bps`, the net over that year's mean
  NAV for its own days, not annualised.

`cost_per_traded()` is the fixed cost, the market impact and both in basis
points of the value traded. `costs_by_reason()` is each reason's cost, impact,
value traded and cost in basis points of it. Both cover the trades of shares
only, so they add up to the run's trading costs, without the futures' costs
or the creation fees.

## Characteristics

```python
print(portfolio.characteristics().iloc[-1])
```

`characteristics()` gives the weighted market cap, P/E, P/B and dividend
yield of the fund, its index and its benchmark (when it holds names) at each
rebalance, from each portfolio's weights at the end of that day: a row a
rebalance, columns by portfolio and figure. Each is weighted over the names
with a value, the weights scaled to them. **Market cap and dividend yield are
weighted means; P/E and P/B are weighted harmonic means**, taken through the
earnings and book yields, so a loss or a tiny book does not blow them up. A
figure the data does not give is None: the sample has no P/E or P/B.

The values are the ones in force on each rebalance date, kept with the run: a
market cap is the close times shares outstanding in the portfolio's currency,
P/E and P/B the `pe_ratio` and `pb_ratio` features, and the dividend yield the
ordinary dividends over the trailing year over the close.

## Income, AUM and flows

| Method | Gives |
| --- | --- |
| `income()` | Each day's distribution yield (paid out over the year to it, over that day's NAV) and trailing dividend yield (the gross dividends the fund became owed over the year, over its mean NAV over the year); blank for the run's first year |
| `dividend_growth()` | Each calendar year's gross dividends, and their growth on the year before |
| `aum()` | The fund's assets under management (its NAV) and units outstanding each day |
| `net_flows(frequency)` | Each month's (`M`) or year's (`Y`) net flows, money in less money out as dealt, and their share of the AUM at the close before it began |
| `flow_costs()` | For each day with flows: the flow, the trading it caused, the creation fees received and the net cost |

```python
print(portfolio.aum().tail(1))
print(portfolio.net_flows("Y"))   # no flows in the sample, so zeros
```

**A flow's trading cost** is every trading cost of its day when the day had
no rebalance. On a rebalance day, the flow caused only part of the trading,
so it is charged the share of the day's costs that its size is of the value
traded, at most all of them.

## Dealing, share classes and cash management

These describe flows, share classes and the cash band, which the sample has
none of, so its tables are empty.

`dealing_costs()` gives each day's subscription or redemption (an ETF's
creation or redemption): its kind and size, the trading it caused, what the
dealing investors paid into the fund toward it through the price or a levy,
the net cost to the fund and in basis points of the flow. For an ETF it adds
`ap_cost`, the flow times the quote's cost of creating or redeeming that day.
`dealing_summary(by)` adds them up by kind over the `run` or each `year`.

`class_costs(by)` gives each share class's costs in the portfolio's
currency: its `fees`; its `hedge_costs` less the `hedge_carry` its forwards
earned (each forward's locked-in carry for the share of its term it was
held); the `flow_trading` its own flows caused, less the `dealing_received`
from its dealing investors; and its `other_trading`, its share of the rest of
the fund's trading (rebalances, dividends, the cash band, futures) by its
share of the fund's assets at the close before. The classes' trading adds up
to the fund's. Over the `run` the net is also in basis points of the class's
mean assets a year, each `year` of that year's. A run without share classes
raises a `KeyError`.

`cash_band_events()` lists each time the cash left its band and was raised
or spent back to its target: the cash before and after, the target and band,
what was traded and its cost, the gap to the target before and after, the
share of the gap closed (1 is cash brought exactly to its target) and the
cost in basis points of the value traded. `cash_band_summary()` totals them
for raises, spend-downs and both, and `cash_band_calendar(frequency)` counts
them and their cost each month or year. See the cash band in
[backtest](backtest.md#flows-units-and-the-fee).

```python
print(portfolio.flow_costs().empty, portfolio.cash_band_events().empty)   # True True
print(portfolio.etf_premium())                                            # None
```

## ETFs and futures

`etf_premium()` is an [ETF](fund-vehicles.md#exchange-traded-funds)'s premium
to NAV on the exchange (its price over its NAV, less one): the mean, its
standard deviation, the widest premium and widest discount, and the mean
bid-ask spread. It is None for any other vehicle.

For a fund holding [index futures](strategies.md#futures-based-replication),
each method below raises a `KeyError` on a run that held none:

| Method | Gives |
| --- | --- |
| `futures_trading()` | A row for each reason (`open`, `roll`, `resize`), one for `between_rolls` (all but rolls) and one for `all`: trades, contracts and notional traded, cost, cost per contract and in basis points of the notional, and turnover (the notional a year over the mean NAV) |
| `futures_basis(daily)` | The basis (the futures' price over their index's level, less one), split into the carry (fair value over the level) and the mispricing (the price over fair value): each day, or their mean, standard deviation, lowest and highest in basis points |
| `futures_basis_risk()` | The tracking error the futures bring, split into the pricing basis (the futures' daily return less their own index's) and the underlying mismatch (their index's less the fund's index's), each with its mean a year and the days measured |

The underlying mismatch brings tracking error even at fair value with no
costs. A day the position rolled is left out of the basis risk, since the
price moved to another contract.

## Where it sits

Downstream of the [backtest](backtest.md), beside
[performance](performance.md): both read the same run, finished or stored,
and compute rather than re-simulate. The risk of what the run held is in
[risk analysis](risk-analysis.md) and the [risk model](risk-model.md); where
its return came from is in [attribution](attribution.md). Which of these
figures a report shows is in [reports](reports.md).

The full API is in the [analysis reference](../reference/analysis.md).
