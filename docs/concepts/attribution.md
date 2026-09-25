# Attribution

Where did the return come from, and where did the risk come from. Two
questions, both answered by splitting a total across the things that caused
it, and both with the property that the parts must add to the whole.

## Return attribution

`attribute()` takes series, not results: the return being explained, the
weights held, and the constituents' returns, all per period. Build them from
an `IndexResult` and prices. This example uses a capped market-cap index over
the five US-dollar names in the sample dataset, so local prices need no FX.

```python
from beacon.analysis import attribute, cap_drag, cost_drag, drifted_weights
from beacon.backtest.main import Backtest
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
definition = IndexDefinition(index_id="CAPPED",
                             index_name="Capped Index",
                             base_date="2024-01-02",
                             base_value=1000.0,
                             currency="USD",
                             eligibility_rules=[],
                             weighting_scheme=MarketCapWeighted(),
                             rebalancing_frequency="MONTHLY",
                             calendar="XNYS",
                             universe_identifiers=names,
                             max_constituent_weight=0.25)

fetcher = dataset.data_fetcher()
result = IndexCalculator(definition, fetcher).run(end_date="2024-06-28")
prices = dataset.prices()[names].loc["2024-01-02":"2024-06-28"]

weights = drifted_weights(result.weight_snapshots, prices)
asset_returns = prices.pct_change().reindex(weights.index)
period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

attribution = attribute(period_returns, weights, asset_returns)

attribution.contributions   # per constituent, largest first
attribution.residual        # should be machine epsilon
attribution.to_frame()
```

Each entry in `contributions` is a `Contribution` with four fields:

| Field | Holds |
| --- | --- |
| `asset_id` | The constituent |
| `contribution` | Its linked contribution; these sum to the total return |
| `average_weight` | Its mean weight across the window, for context |
| `total_return` | The constituent's own return over the window |

The result also carries `total_return`, `residual`, `periods`, `start` and
`end`. `reconciles()` checks the residual against a tolerance of `1e-9`.

### Within a period the split is exact

    R_t = Σ_i w_{i,t-1} × r_{i,t}

Each name's contribution is last period's weight times this period's return.
`attribute()` applies that shift itself, so pass the weights as held on each
date. Because the weighting scheme drives the index level, this holds against
the level to better than 1e-12 for every scheme, on rebalance days as well as
ordinary ones.

### Across periods it is not, and the fix has a name

**Returns compound; contributions add.** So summing daily contributions
undershoots the compounded total, by about a percentage point over a 130-day
window on a modest example, which is far too large to write off as rounding.

Carino linking is the correction. Scale each period's contributions by
`k_t / K`, where `k_t = ln(1+R_t)/R_t` and `K = ln(1+R)/R` for the total return
`R`. Then the scaled contributions sum to `R` exactly, because

    Σ_t (k_t/K) R_t = (1/K) Σ_t ln(1+R_t) = ln(1+R)/K = R

`attribute()` applies it. The residual is reported regardless and should sit at
machine epsilon. **A residual that is not tiny means an assumption has broken
upstream**, which is worth surfacing rather than rounding away. A period
return of -100% or worse cannot be linked and raises.

### Weights drift, and using the wrong ones is the classic error

Between rebalances the index holds fixed units, so weights move with relative
performance. `drifted_weights()` reconstructs them from the rebalance
snapshots and prices. Attributing with *target* weights instead of *held*
weights attributes a return the index did not earn to a position it did not
hold.

### The drags

`cap_drag` and `cost_drag` are not contributions. A contribution is a name's
share of what happened; a drag is what a decision cost against a
counterfactual: an uncapped index, or a frictionless one. `attribute()` takes
both as optional numbers and reports them alongside, never in the
contribution list, because that would put two different kinds of quantity in
one column. Two helpers compute them.

**`cap_drag(capped, uncapped, prices)`** is the capped index's return minus
the return of the same methodology uncapped, each path drifted from its own
snapshots. The uncapped weights are stored rather than estimated:
`result.cap_reports` holds a `CapReport` for every rebalance where the cap
bound, and its `uncapped_weights` is the counterfactual. A rebalance with no
report was not capped, so its published weights already are the uncapped
ones.

```python
uncapped = {date: (result.cap_reports[date].uncapped_weights
                   if date in result.cap_reports else snapshot)
            for date, snapshot in result.weight_snapshots.items()}

capping = cap_drag(result.weight_snapshots, uncapped, prices)
```

**`cost_drag(total_costs, initial_capital)`** is the transaction costs paid as
a fraction of starting capital, negated so it reads as a drag. It is the
direct effect only: it leaves out the compounding of the capital spent rather
than invested. For the full effect, compare against a zero-cost backtest.

```python
backtest = Backtest(initial_capital=1_000_000.0,
                    transaction_cost_bps=10.0,
                    data_provider=fetcher).run(definition, end="2024-06-28")
paid = sum(trade.transaction_cost for trade in backtest.portfolio.transactions)

attribution = attribute(period_returns, weights, asset_returns,
                        cap_drag=capping,
                        cost_drag=cost_drag(paid, 1_000_000.0))
attribution.cap_drag, attribution.cost_drag
```

## Risk contribution

The same idea applied to volatility rather than return, and the reason a
weights table is not a risk view.

**A name at 8% of a quiet utility might account for 3% of volatility; a name
at 4% of something volatile that moves with everything else accounts for
9%.** Weight tells you what you own. Contribution tells you what you are
exposed to.

```python
from beacon.risk import (
    active_risk_contributions, estimate_risk_model, risk_contributions,
)

model = estimate_risk_model(dataset.returns())
held = result.weight_snapshots[max(result.weight_snapshots)]
benchmark = dataset.equal_weights()

total = risk_contributions(held, model.covariance)
active = active_risk_contributions(held, benchmark, model.covariance)

total.volatility, total.contribution
active.volatility, active.contribution   # volatility here is the tracking error
```

Weights can be a mapping or a pandas Series, such as
`OptimisationResult.weights`. The result is a `RiskContributions` with
`volatility`, per-name `marginal` and `contribution`, `covered_weight` and
`uncovered`.

For weights `w` and covariance `S`, portfolio volatility is `√(w'Sw)`, and
each name's contribution is `w_i × (Sw)_i / σ`. **These sum to σ exactly.** It
follows from Euler's theorem, not from an approximation, so if the parts do
not add to the whole, something is wrong and there is no tolerance to hide
behind.

### Active risk

The identical arithmetic on active weights `w − b`, where the reported
volatility is the tracking error against that benchmark. Usually the more
useful number for an index product: it answers *which position is making me
diverge*.

**Contributions here can be negative, and that is the point.** An active
weight is signed, so a position pointing against the book's overall active
exposure genuinely reduces tracking error: it hedges. An absolute value would
hide whichever position is doing the most useful thing in the book.

Being underweight is not enough to contribute negatively: an underweight in a
name the portfolio is *also* underweight overall contributes positively,
because the active weight and its marginal share a sign.

Active weights are taken over the **union** of both universes. A benchmark
constituent you do not hold is routinely the largest active position there
is, and intersecting would drop it silently.

### Names the model cannot price

A constituent added last week has too little history. Rather than dropping it
and renormalising, which claims the portfolio holds more of the covered names
than it does, the decomposition runs over the covered names *at their actual
weights*, reports `covered_weight`, and lists the rest in `uncovered`. The
identity still holds exactly over the part it describes. For active risk,
`covered_weight` is a share of *gross* active weight, since active weights sum
to roughly zero.

A number that describes 94% of an index and says so beats one that describes
100% of a portfolio nobody holds.

## Where it sits

Downstream of everything: it reads a completed `IndexResult` (and a
`BacktestResult` for costs) plus prices, and computes rather than
re-simulates.

`AttributionResult` carries a `.plot` accessor: `contributions()` draws the
per-constituent bars, with the total, the residual and any drags stated in
the chart's note rather than drawn as bars. See [charts](charts.md).

```python
attribution.plot.contributions()
```

The full API is in the [analysis reference](../reference/analysis.md).
