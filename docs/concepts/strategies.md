---
title: Strategies
description: "What a backtest holds: an index tracked in full, or through an optimised subset or a stratified sample of it."
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
