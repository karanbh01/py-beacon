# Risk model

`beacon.analysis.risk` answers questions about **one** series: this
portfolio's volatility, this index's drawdown. `beacon.risk` answers questions
about how assets move **together**, which is a different thing and the input
an [optimiser](optimiser.md) needs.

It needs only numpy, so it is part of the core rather than behind an extra.

```python
from beacon.risk import estimate_risk_model
from beacon.testing import dataset

returns = dataset.returns()                # dates on the index, assets on the columns
model = estimate_risk_model(returns)

model.covariance      # annualised, indexed and columned by asset id
model.correlation     # derived from it, unit diagonal
model.volatilities()  # the square root of the diagonal
model.diagnostics     # how it was produced, and how well conditioned
```

Rows with any missing value are dropped first, so every entry is estimated
over the same periods. The estimate is annualised with `periods_per_year`,
252 by default for daily returns.

The model also answers portfolio questions directly:

```python
held = dataset.equal_weights()
benchmark = {"AAA": 0.30, "BBB": 0.25, "CCC": 0.15,
             "DDD": 0.10, "EEE": 0.10, "FFF": 0.10}

model.portfolio_volatility(held)
model.tracking_error(held, benchmark)   # volatility of the active weights
```

## Why shrinkage

A sample covariance estimated from a short history is noisy, and the noise is
worst exactly where it matters: the smallest eigenvalues, which an optimiser
inverts. The result is a portfolio that looks brilliant on the estimate and
falls apart out of sample.

Shrinkage blends the sample toward a structured target, trading a little bias
for a lot less estimation error:

- `CONSTANT_CORRELATION` (the default) keeps each asset's own variance and
  gives every pair the average sample correlation.
- `SCALED_IDENTITY` puts the average variance on the diagonal and zero
  elsewhere, discarding every estimated relationship.

`estimate_risk_model` shrinks by default. If you do not name an intensity, it
uses `assets / (assets + observations)`: hard when assets outnumber
observations, light when the history is long.

```python
from beacon.risk import SCALED_IDENTITY

estimate_risk_model(returns, intensity=0.0)          # the raw sample covariance
estimate_risk_model(returns, target=SCALED_IDENTITY)
```

This is a shape-based rule, **not** the optimal Ledoit-Wolf intensity, which
is not implemented. If you have computed an optimal intensity elsewhere, pass
it as `intensity`.

## Diagnostics, and why they are reported rather than fixed

`RiskDiagnostics` records the observations and assets used, the target and
intensity, the average correlation, the condition number, the smallest
eigenvalue, whether the matrix is positive semi-definite, and whether it was
repaired. A badly conditioned matrix is not repaired silently:

```python
estimate_risk_model(returns, repair=True)   # eigenvalue clipping, opt-in
```

Repair runs only when the estimate is not positive semi-definite. It is off by
default because shrinkage should make it unnecessary, and because clipping
shifts the variances: quietly changing an estimate to make it usable is how a
number nobody chose ends up in a portfolio.

## Factor models

`fit_factor_model` models covariance as `Σ = BFBᵀ + D`: common factor
exposures `B` with factor covariance `F`, plus a diagonal `D` of
asset-specific variance. Factor returns are fitted by cross-sectional
regression each period, with a `market` intercept added by default.
Standardise raw exposures with `z_scores` first.

```python
import math

import pandas as pd

from beacon.risk import fit_factor_model, z_scores

raw = pd.DataFrame({
    "beta": {c.identifier: c.beta for c in dataset.CONSTITUENTS},
    "size": {c.identifier: math.log(c.initial_price * c.shares_outstanding)
             for c in dataset.CONSTITUENTS},
})
factors = fit_factor_model(returns, z_scores(raw))

split = factors.decompose_active_risk(held, benchmark)
split.factor_variance     # from factor bets
split.specific_variance   # from asset-specific residuals
split.tracking_error      # square root of the total
split.to_frame()          # per-factor exposure and contribution
```

`decompose_active_risk` returns an `ActiveRiskDecomposition`, which splits the
**active variance** (squared tracking error), not the tracking error itself:
`factor_variance + specific_variance = total_variance` exactly. `factor_share`
is the fraction from factors, and the per-factor contributions sum to
`factor_variance`. The identity holds because the model defines `Σ` as
`BFBᵀ + D`, so it reconciles to this model's tracking error, which is not the
one a shrunk sample covariance gives for the same portfolio.

Factor contributions **can be negative**, and are reported that way: a factor
position that hedges another genuinely reduces risk, and an absolute value
would misreport what the portfolio is doing.

`r_squared` is the fraction of return variance the factors explain. Read it
against roughly `k/n` (factors over assets) rather than against zero, since
`k` free parameters always fit something.

## Risk contribution

Which holdings actually drive the risk, a different question from which are
largest. See [attribution](attribution.md#risk-contribution), where the
decomposition and its exactness are covered alongside return attribution.

## Where it sits

Between the data and the [optimiser](optimiser.md). It takes a returns panel,
such as `prices.pct_change()` over prices from the same `DataFetcher`
everything else uses, and produces the covariance a mean-variance problem is
stated against.

`RiskModel` carries a `.plot` accessor: `correlation()` draws the matrix on
the `beacon_corr` scale, shaded from 0.2 upward. The scale is the same in
light and dark mode, so two screenshots of one matrix cannot disagree. See
[charts](charts.md).

```python
model.plot.correlation()
```

The full API is in the [risk reference](../reference/risk.md).
