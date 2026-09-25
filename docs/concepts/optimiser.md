# Optimiser

`EqualWeighted` and `MarketCapWeighted` are closed-form rules. The optimiser
finds weights **numerically**: the portfolio that best meets an objective while
satisfying the constraints you state.

It lives in `beacon.optimise`. Describing a problem (constraints, configs,
results) needs only the core install; solving one needs scipy, from the
`optimise` extra. Without it, a solve raises `MissingDependencyError` naming
the extra.

```bash
pip install "py-beacon-kit[optimise]"
```

## The shape of a problem

Three things: what you want to be close to, what you are not allowed to do,
and optionally a [risk model](risk-model.md) saying how the assets move
together.

```python
from beacon.optimise import (
    FullInvestment, GroupBounds, PositionBounds, minimise_tracking_error,
)
from beacon.risk import estimate_risk_model
from beacon.testing import dataset

risk = estimate_risk_model(dataset.returns())
target_weights = {"AAA": 0.30, "BBB": 0.25, "CCC": 0.15,
                  "DDD": 0.10, "EEE": 0.10, "FFF": 0.10}
technology = dataset.sectors()["Technology"]    # AAA and BBB

result = minimise_tracking_error(
    target_weights,
    [FullInvestment(),
     PositionBounds(0.0, 0.25),
     GroupBounds("Technology", technology, maximum=0.40)],
    risk)

result.weights          # the solution, by asset id
result.binding          # which constraints the solution sits on
result.diagnostics      # what the solver did
```

The target weights define the universe: the solver allocates over exactly
those names, in that order. With a risk model, the objective is squared
tracking error, `(w - b)ᵀ Σ (w - b)`. Without one, `Σ` is the identity and the
objective is the squared distance between the two weight vectors: every unit
of active weight costs the same wherever it is taken. With no constraints at
all, the default is full investment alone.

## The objectives

| Function | Finds | Returns |
| --- | --- | --- |
| `minimise_tracking_error` | The feasible portfolio closest to a target | `OptimisationResult` |
| `minimum_variance_portfolio` | The least risky feasible portfolio | `FrontierPoint` |
| `maximum_return_portfolio` | The highest expected return the constraints allow | `FrontierPoint` |
| `efficient_frontier` | A grid of portfolios tracing the risk/return trade-off | `EfficientFrontier` |

The last three need a risk model, and default to long-only and fully invested
(`FullInvestment()` plus `PositionBounds(0.0, 1.0)`) when no constraints are
given. The maximum-return portfolio refuses a problem where nothing caps the
weights, because its return would be unbounded.

```python
from beacon.optimise import efficient_frontier

expected = (dataset.returns().mean() * 252).to_dict()   # annualised, like the covariance

frontier = efficient_frontier(risk, expected, points=10,
                              constraints=[FullInvestment(),
                                           PositionBounds(0.0, 0.40)])

frontier.minimum_variance.volatility   # the left-hand end
frontier.tangency.sharpe_ratio         # the highest Sharpe ratio available
frontier.to_frame()                    # one row per point
```

Each `FrontierPoint` carries its weights, volatility, expected return, Sharpe
ratio and the labels of the constraints it sits on. `is_monotonic()` checks
that risk rises with return across the grid; a `False` means a point did not
solve to optimality.

## The constraints

Each is a class you construct and pass in a list, so a constraint set is data
rather than a list of arguments to remember.

| Constraint | Limits |
| --- | --- |
| `FullInvestment` | Weights sum to a total, normally one |
| `PositionBounds` | Individual weights, optionally for named assets only |
| `GroupBounds` | The combined weight of a set: a sector, a country, a bucket |
| `TurnoverBudget` | One-way turnover from the current holdings |
| `ExpectedReturnTarget` | Pins the expected return, which is how a frontier is traced |
| `Cardinality` | How many names may be held |

Several `PositionBounds` compose, and the tightest limit on each name wins.
Group members outside the universe are ignored, but a group with no members
in it is an error. Turnover is one-way: half the sum of absolute weight
changes.

`Cardinality` is the odd one out. Counting holdings is not convex, so it is
met by a heuristic: solve, keep the largest positions, then solve again with
the rest pinned at zero. The answer satisfies the limit but is not proven
optimal, and `result.heuristic` says when this happened. Every other
constraint is convex, so without `Cardinality` the problem is convex and a
local optimum is the global one.

The server publishes every constraint type with its parameters, types,
defaults and labels at `GET /optimise/constraint-types`, so an editor renders
from the same source the solver reads.

## Reading the result

`OptimisationResult` carries more than the weights:

| Field | Holds |
| --- | --- |
| `weights` | The solution, indexed by asset id |
| `target_weights` | What was tracked, on the same index |
| `binding` | Constraints the solution sits on, tightest first |
| `slacks` | Every constraint's room at the solution, binding or not |
| `diagnostics` | A `SolverDiagnostics`: converged, iterations, evaluations, objective, status, message |
| `heuristic` | Whether `Cardinality` forced the restricted re-solve |

**`binding` is the part most worth reading.** A constraint that binds changed
the answer, and relaxing it is the only way to improve the objective. One that
does not bind never mattered. Each `BindingConstraint` has a `label`, a `kind`
(equality or inequality), its `slack` and the `unit` the slack is measured in
(`"fraction"`, or `"count"` for `Cardinality`). `slacks` shows what nearly
bound as well.

```python
result.binding_labels()   # ['full investment at 100.0000%', "maximum 40.0000% in group 'Technology'"]
result.tracking_error()   # square root of the objective
result.turnover()         # one-way, against the target unless you pass current weights
result.to_frame()         # target, optimal and active weights, largest active first
```

`tracking_error()` is annualised tracking error only when a risk model was
used. Without one it is a Euclidean distance between weight vectors, not a
volatility.

**A solve either returns a verified answer or raises `CalculationError`.**
After solving, every constraint is re-evaluated against the returned weights,
and an answer that breaks one is refused. A solver that does not converge also
raises, because a stalled solve leaves a point that is feasible but not
necessarily optimal, and returning it would make "the best answer" and "an
answer" look the same. So `diagnostics.converged` is always `True` on a
returned result; the diagnostics tell you how the solve went, not whether to
trust it.

## Feasibility

A constraint set can rule out every portfolio. Where arithmetic on the bounds
alone proves that, the solver raises before solving, with a message naming
what is impossible rather than a solver exit code. It checks for:

- position bounds on one name that cross each other;
- bounds that cannot sum to the amount `FullInvestment` requires;
- a group whose limits contradict its members' bounds;
- a holding limit smaller than the number of names that must be held, or too
  small to reach full investment.

```python
from beacon.exceptions import CalculationError

try:
    minimise_tracking_error(target_weights,
                            [FullInvestment(), PositionBounds(0.0, 0.10)])
except CalculationError as error:
    print(error)   # the maximum weights total 60.0000%, which cannot reach the 100.0000% ...
```

One case looks like a bug, so it is handled directly: when the bounds sum to
exactly the amount to be invested (a cap of `1/n` on `n` names, for instance),
the feasible set is a single point. Different scipy builds report that
differently, so py-beacon returns the determined weights without a search,
still checking them against every other constraint.

The server checks a stored constraint set while you edit it, but only its
shape: unknown types or parameters, a parameter the class rejects, two
`FullInvestment` rows, no investment target, or a non-convex `Cardinality`.
Infeasibility against a universe surfaces when the solve runs. A weight cap on
an index (`max_constituent_weight`) is a different mechanism with its own
check; see [capping](methodology.md).

## Optimised indices

An optimised index is a new index derived from one you already built: a
source index, an objective and constraints. It stores no weights. At each of
the source's rebalance dates, its published weights are solved under the
constraints, and the solved weights are chained into daily levels on the
source's calendar.

```python
from beacon.index.constructor import IndexDefinition
from beacon.index.derived import OptimisedIndexDefinition, calculate_derived_index
from beacon.index.methodology import MarketCapWeighted

parent = IndexDefinition(index_id="PARENT",
                         index_name="Parent Index",
                         base_date="2024-01-02",
                         base_value=1000.0,
                         currency="USD",
                         eligibility_rules=[],
                         weighting_scheme=MarketCapWeighted(),
                         rebalancing_frequency="MONTHLY",
                         calendar="XNYS",
                         universe_identifiers=list(dataset.UNIVERSE))

optimised = OptimisedIndexDefinition(index_id="PARENT-OPT",
                                     index_name="Parent Index, optimised",
                                     source=parent,
                                     constraints=[FullInvestment(),
                                                  PositionBounds(0.0, 0.20)])

derived = calculate_derived_index(optimised, dataset.data_fetcher(),
                                  end_date="2024-06-28")
derived.weight_snapshots    # the solved weights, at the parent's rebalance dates
```

Optimised weights go through `calculate_derived_index`, never through
`IndexCalculator`: `OptimisedIndexDefinition` is not an `IndexDefinition` and
has no eligibility rules, weighting scheme or schedule of its own. The result
is a normal `IndexResult`, so it backtests, charts and attributes like any
other index. Details worth knowing:

- Rebalancing, calendar and universe follow the source. Base date, base value
  and currency inherit the source's unless you set them.
- The source is referenced, not copied, so editing it changes the optimised
  index at its next calculation. A source can itself be optimised.
- The only objective is `"min_tracking_error"`, solved without a risk model,
  so each rebalance finds the feasible weights closest to the source's by
  squared weight distance. `risk_model` is accepted but not yet used, and
  setting it makes the calculation uncacheable.
- `solve_snapshot(optimised, weights)` runs one rebalance's solve and returns
  its full `OptimisationResult`, binding constraints included.
- `Backtest.run(definition, optimised=True, optimisation_config=...)` builds
  the same derivation ad hoc from an `OptimisationConfig`, with identical
  numbers to a stored one. See [backtest](backtest.md).

## Where it sits

The optimiser consumes a [risk model](risk-model.md) and produces weights.
Those weights reach an index through an optimised index, and a portfolio
through the [backtest](backtest.md).

`OptimisationResult` carries a `.plot` accessor. `exposures()` draws the active
weights against the target. `frontier(frontier)` draws an `EfficientFrontier`
you pass in, with the minimum-variance and tangency points and the capital
market line; pass `risk_free_rate` to start that line at the rate the frontier
was traced with. See [charts](charts.md).

```python
result.plot.exposures()
result.plot.frontier(frontier)
```

The full API is in the [optimiser reference](../reference/optimise.md).
