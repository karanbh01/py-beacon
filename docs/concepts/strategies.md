---
title: Strategies
description: "What a backtest holds: an index tracked in full, through an optimised subset or a stratified sample, synthetically through a swap, or through index futures, or an active strategy built from a signal against a benchmark."
---

# Strategies

The strategy is what a [backtest](backtest.md) holds, whatever the
[vehicle](fund-vehicles.md) that holds it. An index definition passed to
`run()` is tracked in full. `IndexTracking` holds an index another way:

| Replication | What it holds |
| --- | --- |
| `FullReplication()` | Every constituent at its index weight (the default) |
| `OptimisedReplication(holdings=50)` | A subset weighted by the optimiser to minimise tracking error to the index |
| `SampledReplication(holdings=50)` | The largest names in each sector and size cell, each cell at its index weight |
| `SyntheticReplication(basket=...)` | A substitute basket, its return swapped for the index's through a total return swap |
| `FuturesReplication()` | Cash, with the index held through rolled index futures |

The index itself is still calculated, and the run is measured against it:
tracking error and difference compare the portfolio with the index, not with
the replicated weights. At each rebalance the replication turns the index's
weights into the weights the portfolio trades to, and `result.replication`
records what it did.

## Optimised replication

At each rebalance the covariance of the constituents is estimated from their
trailing daily returns in the book's currency (`lookback_days=252` by
default), shrunk toward constant correlation. The optimiser then finds the
long-only, fully invested weights closest to the index in tracking error,
holding no more than `holdings` names and meeting any other `constraints`
(such as `GroupBounds` on sectors). The holdings limit is met by solving,
keeping the largest positions and solving again over them, so the answer is
feasible but not proven optimal.

A name with fewer than `minimum_observations` returns (126 by default), such
as a recent listing, cannot be measured, so it is held at its index weight
outside the optimisation and listed in that rebalance's `unmeasured`.

Each rebalance records the **ex-ante tracking error**: the annualised
tracking error the solve expected, from that rebalance's covariance.

## Sampled replication

Stratified sampling keeps the index's mix without an optimiser. The
constituents are split into cells by sector (the `SECTOR` classification by
default) and by size (`size_buckets=3`: terciles of market cap). Each cell is
given holdings in proportion to its index weight, at least one each, holds its
largest names by index weight, and is scaled to its index weight, so every
sector and size group keeps its weight. When the limit is smaller than the
number of cells, the heaviest cells are kept and the others' weight is spread
across them.

```python
import logging

from beacon.backtest import Backtest
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.strategy import IndexTracking, OptimisedReplication, SampledReplication
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

fetcher = dataset.data_fetcher()
definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2024-01-02", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=list(dataset.UNIVERSE),
)
backtest = Backtest(initial_capital=10_000_000.0, transaction_cost_bps=5.0,
                    data_provider=fetcher)

for replication in (OptimisedReplication(holdings=4), SampledReplication(holdings=4)):
    result = backtest.run(IndexTracking(definition, replication),
                          start="2024-01-02", end="2025-12-31")
    print(replication.name, result.replication[0].holdings, "names,",
          "tracking error", round(result.get_tracking_error(), 4))
```

A `Fund` can hold its index through a replication too: pass the
`IndexTracking` as its `strategy`.

## Synthetic replication

A synthetic fund holds a basket of securities that need not be the index's,
and a total return swap with a counterparty. On the swap's notional the fund
pays the basket's return and a spread, and is paid the index's return, so
whatever the basket earns goes out again and the fund earns the index's
return less the spread. Real swap-based funds hold whatever liquid basket the
counterparty proposes within their collateral rules, so the basket is
required: an index definition, calculated over the run and held at its
weights, or fixed weights by identifier.

The basket still shapes the result:

- **Resets.** Between resets the swap is worth the index's performance less
  the basket's. The swap is reset, its value settled in cash and its notional
  set back to the fund's net assets, when its value owed to the fund reaches
  `exposure_limit` of net assets (0.10, the UCITS limit, by default), at each
  of the index's rebalances, and on any day money flows in or out. A basket
  unlike the index reaches the limit sooner, and each reset trades the basket.
- **Counterparty exposure.** What the fund would lose if the counterparty
  failed: the swap's value when it is owed to the fund.
  `result.counterparty_exposure` is that as a share of NAV each day.
- **Dividend tax.** The fund passes on the basket's dividends as it received
  them, after its own withholding, while the swap pays the index's return as
  the index measures it: price, total or net total return.
- **Costs.** The basket is traded at the run's costs, and the spread
  (`spread_bps`, a year) accrues on the notional every calendar day
  (ACT/360).

The NAV includes the swap's value, and `result.swap_value` and
`result.swap_resets` record it and each reset: the day, the value settled,
why, and the new notional.

```python
from beacon.strategy import SyntheticReplication

basket = {name: 1.0 for name in list(dataset.UNIVERSE)[:3]}
synthetic = backtest.run(
    IndexTracking(definition, SyntheticReplication(basket, spread_bps=15.0)),
    start="2024-01-02", end="2025-12-31")

print("tracking difference", round(synthetic.get_tracking_difference(), 4))
print(len(synthetic.swap_resets), "resets, most exposure",
      round(synthetic.counterparty_exposure.max(), 4))
```

## Futures-based replication

A futures-based fund holds no securities. Its cash earns the run's cash rate
(`ModellingAssumptions(cash_rate=...)`), and a long position in the index's
futures, sized to its net assets, carries the index's return. Profit and loss
is settled in cash every day, as variation margin is, so the NAV is the cash.

Each contract is priced at fair value from the index level S, the cash rate
r and the index's dividend yield q, with T the time to expiry over calendar
days (ACT/365):

```text
F = S x exp((r - q) x T)
```

The futures give up the cash rate the cash earns and are paid the dividend
yield, so the fund earns about the index's price return plus its dividend
yield, less the cost of rolling.

- **Contracts** expire quarterly, on the third Friday of March, June,
  September and December. The position rolls to the next one `roll_days`
  sessions before expiry (5 by default), paying `roll_cost_bps` on its
  notional each time, for the richness real rolls pay over fair value.
- **The dividend yield.** None for a total return or net total return index,
  whose level already counts its dividends. For a price index it is estimated
  at each roll from the constituents' dividends over the trailing twelve
  months, at their index weights, unless `dividend_yield` gives one.
- **Sizing.** The position is sized to the fund's net assets when it opens,
  at each rebalance and roll, and on any day money flows in or out, and
  drifts with the market between. Contracts are fractional, and initial
  margin is not modelled, since it is the fund's own cash.

`result.futures_notional`, `result.futures_rolls` and
`result.futures_exposure` record the position. The same futures can
equitise a physical fund's cash buffer: `Implementation(cash_buffer=0.03,
equitise=IndexFutures())`.

```python
from beacon.assumptions import ModellingAssumptions
from beacon.strategy import FuturesReplication

with_cash_rate = Backtest(initial_capital=10_000_000.0, data_provider=fetcher,
                          modelling_assumptions=ModellingAssumptions(cash_rate=0.04))
futures = with_cash_rate.run(
    IndexTracking(definition, FuturesReplication(roll_cost_bps=2.0)),
    start="2024-01-02", end="2025-12-31")

print("tracking difference", round(futures.get_tracking_difference(), 4))
print(len(futures.futures_rolls), "rolls, costing",
      round(sum(roll.cost for roll in futures.futures_rolls), 2))
```

## Active strategies

An `ActiveStrategy` builds its own portfolio from a signal and is measured
against a benchmark, an index calculated as any index is. At each rebalance
(the first session of each month by default) it:

1. takes the benchmark's weights in force that day;
2. picks its candidates: its `universe` (the benchmark's constituents by
   default), less any failing its `screen`, a condition such as
   `data.market.market_cap > 1e9`;
3. scores them with its signal;
4. estimates the covariance from a year of returns, as optimised replication
   does, holding a name too new to measure at its benchmark weight;
5. builds the long-only, fully invested portfolio with its construction
   method, within its constraints.

**Signals.** A signal's values are standardised across the candidates,
capped at three standard deviations, and negated when `higher_is_better` is
False; a name with no value scores 0.

| Signal | Its value for a name |
| --- | --- |
| `FieldSignal(field)` | A market, reference or feature field, such as `data.features.fundamentals.earnings_yield` |
| `Momentum(lookback_days=252, skip_days=21)` | The return over the lookback, leaving out the most recent month |
| `FunctionSignal(function)` | Whatever `function(name, date, fetcher)` returns |

**Construction.** Every method shares one interface, `Construction`, so
more can be added. Each is long-only and fully invested, within the
strategy's constraints. Two are built around the benchmark and the scores:

- `MaxAlpha(tracking_error=0.03)` holds the most exposure to the scores that
  a tracking-error budget allows: the budget is the target, and the active
  bets are as large as it lets them be. It is solved through the equivalent
  mean-variance problem, with the risk aversion found by bisection so the
  tracking error meets the budget, or falls short of it when the other
  constraints keep the portfolio closer to the benchmark.
- `MeanVariance(risk_aversion=10.0, alpha_per_score=0.02)` maximises the
  expected active return (each score times `alpha_per_score` a year) less
  `risk_aversion` times the active variance. One setting trades return for
  risk, and the tracking error is whatever results.

Three ignore the benchmark's weights and need no signal (the run is still
measured against the benchmark):

- `MinimumVariance()` holds the least risky portfolio the constraints allow.
- `MaximumDiversification()` holds the highest diversification ratio: the
  holdings' weighted average volatility over the portfolio's own.
- `RiskParity(budgets=None)` has each name contribute a set share of the
  portfolio's variance, equal shares by default. A name's contribution is
  its weight times its covariance with the portfolio, over the variance.
  `budgets` sets the shares by name, scaled to add up to one.

Two build from expected returns:

- `MaximumSharpe(base_return=0.05, alpha_per_score=0.02)` holds the highest
  expected excess return per unit of risk, with each name expected to earn
  `base_return` over cash plus its score times `alpha_per_score`.
- `BlackLitterman(risk_aversion=2.5, tau=0.05, alpha_per_score=0.02,
  confidence=1.0)` starts from the returns the benchmark implies (the
  returns that make it the best portfolio at that risk aversion), takes each
  scored name as a view that it will earn its implied return plus its score
  times `alpha_per_score`, blends the two, and holds the best trade-off of
  the blended returns against variance. With no scores it holds the
  benchmark; `confidence` moves the blend toward the views.

And two more:

- `FactorTilts(factors, targets, strength=1.0)` aims at target active
  exposures to factors while staying close to the benchmark. Each factor is
  a signal, whose scores are the names' exposures, such as
  `FactorTilts({"value": FieldSignal(data.features.fundamentals.earnings_yield)},
  {"value": 0.3})`. It minimises the active variance plus `strength` times
  the squared misses from the targets, and needs no signal of its own.
- `TopN(n=20, weighting="equal")` holds the top `n` names by score, equally
  or in proportion to their scores (`weighting="signal"`): a simple rule the
  optimised methods can be judged against. Its weights are moved only as far
  as the constraints require.

**Constraints,** measured against the benchmark where it matters:

| Constraint | The portfolio must |
| --- | --- |
| `TrackingErrorBudget(maximum)` | Have an ex-ante tracking error of at most *maximum* |
| `ActiveShare(minimum, maximum)` | Differ from the benchmark by an active share in the range |
| `RelativeSectorBounds(within)` | Hold each sector within *within* of its benchmark weight |
| `RelativePositionBounds(within)` | Hold each name within *within* of its benchmark weight |
| `HoldingsLimit(maximum)` | Hold no more than *maximum* names |
| `TurnoverLimit(maximum)` | Trade at most *maximum* one way from the last rebalance's weights |

Active share is half the summed absolute differences from the benchmark's
weights. The turnover limit is measured from the last rebalance's target
weights, not from the drifted holdings.

**Reporting.** `result.active` records each rebalance's weights, the
benchmark's, the ex-ante tracking error, the active share, the scores, each
holding's share of the portfolio's variance (`risk_contributions`) and what
the construction reports (`exposures`, a factor tilt's active exposure to
each factor).
The summary adds the `information_ratio` (annualised active return over
tracking error), the average active share and the average ex-ante tracking
error, and `result.active_attribution()` says which names produced the active
return: each name's active weight times its return, linked over the run, with
costs and cash in the residual.

```python
from beacon.strategy import ActiveStrategy, MaxAlpha, Momentum, RelativeSectorBounds

momentum = ActiveStrategy(
    benchmark=definition, signal=Momentum(),
    construction=MaxAlpha(tracking_error=0.03),
    constraints=[RelativeSectorBounds(within=0.10)])

active = backtest.run(momentum, start="2024-01-02", end="2025-12-31")
summary = active.summary()
print(round(summary["information_ratio"], 3),
      round(summary["average_active_share"], 3))
print(active.active_attribution().contributions[0])
```

```python
from beacon.strategy import RiskParity

balanced = backtest.run(ActiveStrategy(benchmark=definition, construction=RiskParity(),
                                       rebalancing="QUARTERLY"),
                        start="2024-01-02", end="2025-12-31")
shares = balanced.active[0].risk_contributions
print(round(min(shares.values()), 3), round(max(shares.values()), 3))
```

