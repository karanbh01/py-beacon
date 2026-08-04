# Optimiser

Where `EqualWeighted` and `MarketCapWeighted` are closed-form rules, the
optimiser derives weights **numerically**: it finds the portfolio that best
meets an objective subject to constraints you state.

Lives in `beacon.optimise`, behind the `optimise` extra (scipy).

```bash
pip install "py-beacon[optimise]"
```

## The shape of a problem

Three things: what you are trying to achieve, what you are not allowed to do,
and a risk model saying how the assets move together.

```python
from beacon.optimise import (
    FullInvestment, GroupBounds, PositionBounds, minimise_tracking_error,
)
from beacon.risk import estimate_risk_model

risk = estimate_risk_model(returns)

result = minimise_tracking_error(
    target_weights,
    [FullInvestment(),
     PositionBounds(0.0, 0.25),
     GroupBounds("Technology", technology_names, maximum=0.30)],
    risk)

result.weights          # the solution
result.binding          # which constraints actually bit
result.diagnostics      # what the solver did, and whether to trust it
```

## The objectives

| Function | Finds |
| --- | --- |
| `minimise_tracking_error` | The portfolio closest to a target, in risk terms |
| `minimum_variance_portfolio` | The lowest-volatility feasible portfolio |
| `maximum_return_portfolio` | The highest expected return the constraints allow |
| `efficient_frontier` | A set of portfolios tracing the risk/return trade-off |

## The constraints

Each is a class you construct and pass in a list, so a constraint set is data
rather than a list of arguments to remember.

| Constraint | Limits |
| --- | --- |
| `FullInvestment` | Weights sum to a total, normally one |
| `PositionBounds` | Individual weights, optionally for named assets only |
| `GroupBounds` | The combined weight of a set — a sector, a country, a bucket |
| `TurnoverBudget` | How far the solution may move from current holdings |
| `ExpectedReturnTarget` | Pins the expected return, which is what traces a frontier |
| `Cardinality` | How many names may be held |

`Cardinality` is the odd one out and says so: counting holdings is not convex,
so it is solved by a heuristic and the answer is not provably optimal. Every
other constraint keeps the problem convex, which is what makes a solution
trustworthy rather than merely returned.

`GET /optimise/constraint-types` publishes all of this — every constraint with
its parameters, types, defaults and labels — so an editor renders from the same
source the solver reads rather than from a copy that drifts.

## Reading the result

**`binding` is the part most worth looking at.** A constraint that bit changed
your answer; one that did not was never relevant. An optimiser returning only
weights leaves you unable to tell whether the position limit shaped the
portfolio or merely watched it.

**`diagnostics` is the solver's account of itself** — whether it converged, how
many iterations, the final objective. A run that did not converge is returned
rather than raised, because a near-miss is often informative where a bare
exception is not; but it is labelled, so nothing silently passes for a
solution.

## Feasibility

A constraint set can rule out every portfolio, and that is worth catching
before the solver rather than during it. A cap of 5% across ten names
distributes at most 50%, so it cannot coexist with full investment — that is
arithmetic, not a solver outcome, and the server reports it as a validation
finding while a user is still editing.

One case is worth knowing about because it looks like a bug: a cap of exactly
`1/n` makes the feasible set a single point. Different scipy builds report that
differently, so Beacon answers it directly rather than solving — the answer is
determined, and asking a solver to discover it invites a status code that
varies by machine.

## Where it sits

Alongside the [methodology](methodology.md) layer's weighting step, consuming a
[risk model](risk-model.md), producing weights that flow onward exactly as any
other scheme's do — into `IndexCalculator`, then `BacktestEngine`.

`OptimisationResult` carries a `.plot` accessor: `exposures()` for active
weights against the target, `frontier()` for the efficient frontier with its
named points and the capital market line. See the [gallery](../gallery.md).
