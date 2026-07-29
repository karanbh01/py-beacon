# Concepts overview

Beacon separates *what an index is* from *how it is computed* from *how a
portfolio that tracks it behaves*. Each concept page below covers one part
of that separation:

- **[Universe](universe.md)** — the raw set of identifiers an index draws
  from, and how it is resolved into `Asset` objects each rebalance.
- **[Methodology](methodology.md)** — the rules layer: Selection
  (eligibility), Weighting (schemes), and Treatment (corporate-action
  divisor adjustments).
- **[Backtest](backtest.md)** — how `BacktestEngine` simulates a tracking
  portfolio against a target weight schedule, and why its NAV is not the
  same number as the index level.
- **[Optimiser](optimiser.md)** — *not yet implemented.* Where a
  weight-optimisation layer will sit once it exists.
- **[Risk Model](risk-model.md)** — what risk analytics exist today (scalar
  metrics) versus what a portfolio-level risk model would add
  (covariance, shrinkage, factor decomposition) — *not yet implemented.*

These map onto the three-layer pipeline described on the [Home](../index.md)
page:

```
Methodology  ->  Calculator (IndexCalculator.run() -> IndexResult)  ->  Backtest (BacktestEngine.run() -> BacktestResult)
```

Universe and Methodology together define the Methodology layer; the
Calculator layer is `IndexCalculator` walking business days and applying
that methodology; the Backtest layer is `BacktestEngine` simulating a real
portfolio against the resulting weight schedule.
