# Attribution

Where did the return come from, and where did the risk come from. Two
questions, both answered by splitting a total across the things that caused it,
and both with the property that the parts must add to the whole.

## Return attribution

```python
from beacon.analysis import attribute, drifted_weights

weights = drifted_weights(result.weight_snapshots, prices)
asset_returns = prices.pct_change().reindex(weights.index)
period_returns = (weights.shift(1) * asset_returns).sum(axis=1)

attribution = attribute(period_returns, weights, asset_returns,
                        cap_drag=-0.003, cost_drag=-0.004)

attribution.contributions   # per constituent
attribution.residual        # should be machine epsilon
```

### Within a period the split is exact

    R_t = Σ_i w_{i,t-1} × r_{i,t}

Each name's contribution is last period's weight times this period's return.
Nothing subtle. Since the weighting scheme was made to drive the level, this
holds to better than 1e-12 for every scheme, on rebalance days as well as
ordinary ones.

### Across periods it is not, and the fix has a name

**Returns compound; contributions add.** So summing daily contributions
undershoots the compounded total — by over a percentage point across a 130-day
window on a modest fixture, which is far too large to write off as rounding.

Carino linking is the correction. Scale each period's contributions by
`k_t / K`, where `k_t = ln(1+R_t)/R_t` and `K = ln(1+R)/R` for the total return
`R`. Then the scaled contributions sum to `R` exactly, because

    Σ_t (k_t/K) R_t = (1/K) Σ_t ln(1+R_t) = ln(1+R)/K = R

`attribute()` applies it. The residual is reported regardless and should sit at
machine epsilon — **a residual that is not tiny means an assumption has broken
upstream**, which is worth surfacing rather than rounding away.

### Weights drift, and using the wrong ones is the classic error

Between rebalances the index holds fixed units, so weights move with relative
performance. `drifted_weights()` reconstructs them. Attributing with *target*
weights instead of *held* weights attributes a return the index did not earn to
a position it did not hold.

### The drags

`cap_drag` and `cost_drag` are not contributions. A contribution is a name's
share of what happened; a drag is what a decision cost you against a
counterfactual — an uncapped index, or a frictionless one. They are reported
separately because adding them to the constituent list would put two different
kinds of quantity in one column.

Cap drag is computable precisely because every rebalance snapshot keeps its
`uncapped_weights`: the counterfactual is stored, not estimated.

## Risk contribution

The same idea applied to volatility rather than return, and the reason a
weights table is not a risk view.

**A name at 8% of a quiet utility might account for 3% of volatility; a name at
4% of something volatile that moves with everything else accounts for 9%.**
Weight tells you what you own. Contribution tells you what you are exposed to.

```python
from beacon.risk import risk_contributions, active_risk_contributions

total = risk_contributions(weights, model.covariance)
active = active_risk_contributions(weights, benchmark_weights, model.covariance)
```

For weights `w` and covariance `S`, portfolio volatility is `√(w'Sw)`, and each
name's contribution is `w_i × (Sw)_i / σ`. **These sum to σ exactly** — it
follows from Euler's theorem, not from an approximation — so if the parts do
not add to the whole, something is wrong and there is no tolerance to hide
behind.

### Active risk

The identical arithmetic on active weights `w − b`, where the reported figure
is the tracking error against that benchmark. Usually the more useful number
for an index product: it answers *which position is making me diverge*.

**Contributions here can be negative, and that is the point.** An active weight
is signed, so a position pointing against the book's overall active exposure
genuinely reduces tracking error — it hedges. An absolute value would hide
whichever position is doing the most useful thing in the book.

Being underweight is not enough to contribute negatively: an underweight in a
name the portfolio is *also* underweight overall contributes positively,
because the active weight and its marginal share a sign.

Active weights are taken over the **union** of both universes. A benchmark
constituent you do not hold is routinely the largest active position there is,
and intersecting would drop it silently.

### Names the model cannot price

A constituent added last week has too little history. Rather than dropping it
and renormalising — which claims the portfolio holds more of the covered names
than it does — the decomposition runs over the covered names *at their actual
weights* and reports `covered_weight`. The identity still holds exactly over
the part it describes, and the response says which part that is.

A number that describes 94% of an index and says so beats one that describes
100% of a portfolio nobody holds.

## Where it sits

Downstream of everything: it reads a completed `IndexResult` or
`BacktestResult` plus prices, and computes rather than re-simulates.

`AttributionResult` carries a `.plot` accessor — `contributions()` draws the
per-constituent bars with the drags alongside. See the
[gallery](../gallery.md).
