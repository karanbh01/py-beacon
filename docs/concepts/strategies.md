---
title: Strategies
description: "What a backtest holds: an index tracked in full or through an optimised subset or a stratified sample, or an active strategy built from a signal against a benchmark."
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

**Construction.** Two methods share one interface, `Construction`, so more
can be added:

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
benchmark's, the ex-ante tracking error, the active share and the scores.
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

