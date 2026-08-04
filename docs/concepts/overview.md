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
- **[Optimiser](optimiser.md)** — deriving weights numerically rather than
  by a closed-form rule: objectives, constraints, and reading which of them
  actually bit.
- **[Risk Model](risk-model.md)** — how assets move *together*: covariance
  estimation, shrinkage and why it is on by default, and factor models.
- **[Attribution](attribution.md)** — where the return came from and where
  the risk came from, and why in both cases the parts must add to the whole
  exactly.

These map onto the three-layer pipeline described on the [Home](../index.md)
page:

```
Methodology  ->  Calculator (IndexCalculator.run() -> IndexResult)  ->  Backtest (BacktestEngine.run() -> BacktestResult)
```

Universe and Methodology together define the Methodology layer; the
Calculator layer is `IndexCalculator` walking business days and applying
that methodology; the Backtest layer is `BacktestEngine` simulating a real
portfolio against the resulting weight schedule.
