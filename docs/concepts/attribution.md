---
title: Attribution
description: "Splitting an index's or a run's return across its names, groups and decisions, with the cap and cost drags."
---

# Attribution

Where did the return come from, and where did the risk come from. Two
questions, both answered by splitting a total across the things that caused
it, and both with the property that the parts must add to the whole.

## Return attribution

`attribute()` takes series, not results: the return being explained, the
weights held, and the constituents' returns, all per period. Build them from
an `IndexResult` and prices. This example uses a capped market-cap index over
the five US-dollar names in the sample dataset, so local prices need no FX.
For a backtest, `result.attribution` does all of this from the run's own
records; see [A run's attribution](#a-runs-attribution).

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
than invested. For the full effect, compare against a zero-cost backtest. A
run's `implementation()` measures each cost as a share of the NAV on the day
it was paid instead, linked so it adds up with the rest of the tracking
difference (see [Implementation](#implementation)).

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

## A run's attribution

A backtest result carries its own attribution: `result.attribution` is an
`Attribution` (`beacon.analysis.attribution.Attribution`) over the records the
run kept, so there are no weights or prices to rebuild. It reads each book's
daily weights, each name's daily total return in the fund's currency, the
reference data at each rebalance, and the run's cash and costs.

A book is `"fund"`, `"index"` (the index the fund tracked), `"optimised"`
(the optimised index, for a strategy that tracks one) or `"benchmark"` (the
benchmark of record). Every method takes `start` and `end`, both
included, and covers the whole run by default. The days are linked as above:
by Carino's `ln(1+R)/R` for one book's return, and for a return over a
comparator by the active factor

    k_t = (ln(1+R_t) - ln(1+R^b_t)) / (R_t - R^b_t)

over the same for the window (`1/(1+R_t)` where the two returns are equal),
so the parts add up to the fund's compounded return less the comparator's.

The examples run the capped index above as a fund, with an equal-weighted
index over the same names as its benchmark. A benchmark given as a level
series has no weights, so every method that reads them raises a `KeyError`
against it; only the returns-based factor attribution works there.

```python
from beacon.index.methodology import EqualWeighted

equal = IndexCalculator(IndexDefinition(index_id="EQUAL",
                                        index_name="Equal Index",
                                        base_date="2024-01-02",
                                        base_value=1000.0,
                                        currency="USD",
                                        eligibility_rules=[],
                                        weighting_scheme=EqualWeighted(),
                                        rebalancing_frequency="MONTHLY",
                                        calendar="XNYS",
                                        universe_identifiers=names),
                        fetcher).run(end_date="2024-06-28")

run = Backtest(initial_capital=1_000_000.0,
               transaction_cost_bps=10.0,
               benchmark=equal,
               data_provider=fetcher).run(definition, end="2024-06-28")

run.attribution.contribution().to_frame()                   # each name
run.attribution.contribution("benchmark", field="SECTOR").to_frame()
run.attribution.active("benchmark").to_frame()
```

### By name and by group

`contribution(book="fund", field=None)` is each name's contribution to a
book's return, as an `AttributionResult` like `attribute()`'s. With `field`, a
column of the reference data the run kept (`SECTOR`, `CURRENCY` and
`EXCHANGE` in the sample), it is each group's instead: a name counts in the
group it was in at the close before, which is its group at the last rebalance
before the day, and a group's return is what its names earned over what it
held.

`active(against="index", field=None)` does the same for the fund against a
comparator: the fund's weights less the comparator's, both as they drift and
over the union of their names, against the fund's daily return less the
comparator's. The days are linked by the active Carino factor, so its
`total_return` is the fund's compounded return less the comparator's, the same
total the Brinson, currency, factor and implementation results below explain.

Either way, what the names do not explain (cash, costs, fees, futures, the
timing of dividends) is the residual, never folded into a name.

### Brinson-Fachler

`brinson(field, against="index")` splits the fund's return over a comparator,
group by group, into how it weighted each group (allocation), what it held
within each (selection) and the two together (interaction). It is worked out
each day, with `w` and `b` the fund's and the comparator's weights at the
close before and `R` and `R^b` their returns, by group `g` and in all:

    allocation_g  = (w_g - b_g) (R^b_g - R^b)
    selection_g   = b_g (R_g - R^b_g)
    interaction_g = (w_g - b_g) (R_g - R^b_g)

and the days are linked by the active factor. **Interaction has a line of its
own** rather than being folded into selection, and **cash is a group of its
own**: a book's weight in it is one less its names', and its return is the
interest the fund earned over the cash it held at the close before.

A group the fund does not hold takes the comparator's group return as its
own, so it has no selection or interaction, only the allocation of staying
out. A group the comparator does not hold takes the comparator's whole
return, so the fund's bet on it is all interaction.

The `BrinsonResult` carries `fund_return`, `comparator_return` and
`active_return` (the fund's compounded return less the comparator's),
`explained` (what the effects add up to: the holdings' active return) and
`residual` (the active return less that: costs, fees, dividends not yet paid,
a price index's missing dividends), with `start`, `end` and `periods`.
`groups` has a row a group: each book's mean weight and compounded return in
it, each effect and their `total`, largest first. A group neither book held is
left out; cash is always shown. `totals` adds each effect over the groups, and
`reconciles()` checks the residual against `1e-9`.

```python
sectors = run.attribution.brinson("SECTOR", against="benchmark")

sectors.active_return, sectors.explained, sectors.residual
sectors.groups[["allocation", "selection", "interaction", "total"]]
sectors.names.loc["Technology"]   # AAA and BBB, adding up to the group's row
```

**`names` opens each group into its names**, indexed by group and name. With
`W_g` and `B_g` the two books' weights in the group, `w_i` and `b_i` a name's
and `r_i` its return:

    d_i           = (w_i / W_g - b_i / B_g) (r_i - R^b_g)
    selection_i   = B_g d_i
    interaction_i = (W_g - B_g) d_i

A group's `d_i` add up to `R_g - R^b_g`, and the days are linked by the group
effects' factors, so **a group's names add up to its selection and
interaction**. Allocation is a decision about the group and stays on its row.
A name the comparator holds and the fund does not has a negative share, so it
shows what not holding it cost. A name that changed group has a row in each.

`brinson_by_period(field, against="index", frequency="Y")` repeats the
attribution over each year, or each month with `"M"`, each period linked
within itself from the close before its first day: a row a period, with the
three effects, `explained`, `active_return` and `residual`. Periods compound,
so the rows do not add up to the whole run's figures.

### Currency

A name's daily return in the fund's currency `r` is its return in its own
currency `l` and the move of that currency `x`: `1 + r = (1 + l)(1 + x)`, so

    r = l + x + l x

`currency(against=None, by_currency=False)` gives each part the name's weight
at the close before and links the days: `local`, `currency`, and the `cross`
term on a line of its own. Local returns are worked out from the run's own
closes and corporate actions in each name's currency, as the fund's returns
are. With `against`, it explains the active return from the active weights;
with `by_currency=True`, its `table` splits each part by the names'
currencies. The sample's names are all in US dollars, so here it is all local.

It returns an `EffectsResult`, as `factors()` and `implementation()` do:
`total_return` (a book's compounded return, or the fund's less the
comparator's), `effects` (each effect linked, largest first), `explained`,
`residual`, `table` (empty unless a breakdown was asked for), `start`, `end`,
`periods` and `reconciles()`.

```python
run.attribution.currency(against="benchmark", by_currency=True).table
```

### Factors

`factors(analytics, against=None, method="holdings")` explains the return by
the run's factors. `analytics` is the run's `FactorAnalytics`,
`run.risk.factors`: each name's exposures at each rebalance, standardised over
the universe, and each day's factor returns, from a cross-sectional
regression on those exposures, weighted by market cap, with a market
intercept. `run.risk` needs the data source the run read.

**Holdings-based** (`"holdings"`) splits every name's return as

    r_i = f_market + Σ_k z_ik f_k + e_i

with `z_ik` its exposures at the last rebalance before the day and `e_i`
defined as what is left, **so the split is exact**. Holding `w` at the close
before, the fund earns `(Σ w) f_market` from the market, `X_k f_k` from each
factor (`X_k = Σ w_i z_ik`) and `Σ w_i e_i` of its own: the lines `market`,
one a factor, and `specific`.

**Returns-based** (`"returns"`) regresses the fund's daily return over the
risk-free rate on the factor returns (the market's among them) over the
window, by ordinary least squares with an intercept. It splits the return into
`risk_free`, `alpha`, each beta times its factor's return, and the
regression's `residual`.

With `against`, the holdings method takes the active weights, and the returns
method regresses the active return, with no risk-free rate taken off. Any
other method raises a `ValueError`.

```python
factors = run.risk.factors

run.attribution.factors(factors).effects
run.attribution.factors(factors, method="returns").effects
```

### Implementation

`implementation(against="index")` splits the fund's tracking difference
against the index it tracked into what caused it, day by day from the run's
own accounting, linked by the active factor. Each day, `V` is the base the
fund's return is measured on: the NAV at the close before, grown by any units
dealt that day at yesterday's NAV per unit. `w` and `b` are the fund's and the
index's weights at the close before (the fund's scaled onto `V`, so money
dealt that day is held as cash), `c = 1 - Σ w` is the fund's cash, `r` the
names' total returns, and `g = w - (1 - c) b` the gap between the fund's
invested weights and the index's, `g^R` as it was at the last rebalance.

| Effect | What it is |
| --- | --- |
| `cap_and_screens` | `Σ g^R r`: what the cap, the screens, the smallest position and rounding left the fund holding differently |
| `drift` | `Σ (g - g^R) r`: how that gap has moved since, as flows, dividends and the cash band traded the fund and not the index |
| `cash_drag` | `-c Σ b r` plus the interest earned over `V`: the return the cash gave up against the index, less what it earned |
| `equitisation` | The futures' profit and loss before their costs, over `V`, which wins back the cash drag |
| `fixed_costs`, `market_impact`, `futures_rolls`, `futures_trading` | Each day's cost, over `V` |
| `fees` | The management fee charged, over `V` |
| `withholding_tax` | The tax withheld from dividends, over `V` |
| `creation_fees` | What dealing investors paid into the fund, over `V` |
| `dividend_treatment` | `Σ b r` less the index's own return: the dividends a price index leaves out, or a total return index's own timing |

They add up to the fund's return from its names, cash and futures, less its
costs and fees, less the index's return. `explained` is that sum; the
`residual` is whatever else moved the NAV, such as a currency hedge or trades
dealt away from the close. By default the breakdown starts on the run's first
day, whose close before is the starting capital all in cash, so the cost of
buying in is counted; with a `start`, it starts from that day's close.

```python
costs = run.attribution.implementation()

costs.total_return   # the tracking difference
costs.effects        # here almost all fixed_costs: 10 bps on every trade
costs.residual
```

Against `"benchmark"` the accounting is the same, but the gap at each
rebalance is then the difference between two methodologies, so it reads as
`cap_and_screens`, and its movement since as `drift`.

### Weights

`weights(book="fund")` is a book's daily weights, dates by names, as every
method above reads them: names as strings, blanks as zero. It raises a
`KeyError` if the run kept no weights for that book, as for a benchmark given
as levels, or if the book is none of the four.

```python
run.attribution.weights("benchmark").tail()
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
weight is signed, so a position pointing against the portfolio's overall active
exposure genuinely reduces tracking error: it hedges. An absolute value would
hide whichever position is doing the most useful thing in the portfolio.

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

Downstream of everything. The functions read a completed `IndexResult` (and
a `BacktestResult` for costs) plus prices; a run's attribution reads the
records the run kept. Both compute rather than re-simulate.

`AttributionResult`, from `attribute()`, `contribution()` or `active()`,
carries a `.plot` accessor: `contributions()` draws the
per-constituent bars, with the total, the residual and any drags stated in
the chart's note rather than drawn as bars. See [charts](charts.md).

```python
attribution.plot.contributions()
```

The full API is in the [analysis reference](../reference/analysis.md).
