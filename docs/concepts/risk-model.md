# Risk model

!!! warning "Not yet implemented"
    `beacon.analysis.risk` today holds only scalar, single-series metrics.
    There is no covariance estimation, no shrinkage, and no factor model —
    the portfolio-level risk model described below does not exist yet. It
    will be delivered by **BN-73**
    ([GitHub issue #89](https://github.com/karanbh01/py-beacon/issues/89)).

## What exists today

`src/beacon/analysis/risk.py` provides three scalar functions (also exposed
as methods on the thin `RiskMetricsCalculator` wrapper class):

- `calculate_volatility(price_series, window=252)` — annualised standard
  deviation of returns derived from a single price series.
- `calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year=252)` —
  annualised Sharpe ratio for a single return series.
- `calculate_max_drawdown(price_series)` — maximum peak-to-trough drawdown
  of a single price series.

Each of these operates on **one** series at a time — a single asset, a
single portfolio's NAV, or a single index's level history. They say nothing
about how multiple assets move *together*, which is what a risk model
proper is for.

## What a risk model would add

A portfolio-level risk model would sit alongside the [Methodology](
methodology.md) and [Optimiser](optimiser.md) layers, providing the
cross-sectional inputs neither currently has:

- **Covariance estimation** across the universe's constituents — the input
  a mean-variance optimiser needs and that a single-series metric cannot
  provide.
- **Shrinkage estimators** (e.g. Ledoit-Wolf) to stabilise a sample
  covariance matrix estimated from a short or noisy return history.
- **Factor models** decomposing portfolio risk into systematic factor
  exposures plus idiosyncratic risk, rather than treating every asset pair
  independently.

Once built, this would most plausibly consume the same `DataFetcher`-sourced
price histories that `IndexCalculator` and `BacktestEngine` already use, and
would be a natural input to the optimiser described on the previous page —
but none of this exists yet, so there is no API to document.
