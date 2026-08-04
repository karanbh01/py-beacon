# Risk model

`beacon.analysis.risk` answers questions about **one** series — this
portfolio's volatility, this index's drawdown. `beacon.risk` answers questions
about how assets move **together**, which is a different thing and the input an
optimiser needs.

Needs only numpy, so it is part of the core rather than behind an extra.

```python
from beacon.risk import estimate_risk_model

model = estimate_risk_model(returns)     # dates on the index, assets on the columns

model.covariance      # annualised, asset-indexed
model.correlation     # derived from it, unit diagonal
model.volatilities()  # the square root of the diagonal
model.diagnostics     # how it was produced, and how well conditioned
```

## Why shrinkage

A sample covariance estimated from a short history is noisy, and the noise is
worst exactly where it matters: the smallest eigenvalues, which an optimiser
inverts. The result is a portfolio that looks brilliant on the estimate and
falls apart out of sample.

Shrinkage pulls the sample toward a structured target — a constant-correlation
matrix, or a scaled identity — trading a little bias for a lot of variance.
`estimate_risk_model` shrinks by default and picks an intensity from the
panel's shape if you do not name one.

```python
estimate_risk_model(returns, intensity=0.0)   # raw sample, if you want it
estimate_risk_model(returns, target=SCALED_IDENTITY)
```

The optimal Ledoit-Wolf intensity is **not** implemented; the heuristic is a
shape-based rule, and it says so rather than implying otherwise.

## Diagnostics, and why they are reported rather than fixed

`RiskDiagnostics` carries the condition number, whether the matrix is positive
semi-definite, and how it was estimated. A badly conditioned matrix is not
repaired silently:

```python
estimate_risk_model(returns, repair=True)   # eigenvalue clipping, opt-in
```

Repair is off by default because shrinkage should make it unnecessary, and
because clipping shifts the variances — quietly changing an estimate to make it
usable is how a number nobody chose ends up in a portfolio.

## Factor models

`fit_factor_model` decomposes risk as `Σ = BFBᵀ + D` — common factor exposures
plus asset-specific residual — rather than treating every asset pair
independently. `ActiveRiskDecomposition` then splits a tracking error into the
part explained by factor bets and the part that is idiosyncratic.

Factor contributions **can be negative**, and are reported that way: a factor
position that hedges another genuinely reduces risk, and an absolute value
would misreport what the portfolio is doing.

## Risk contribution

Which holdings actually drive the risk — a different question from which are
largest. See [attribution](attribution.md#risk-contribution), where the
decomposition and its exactness are covered alongside return attribution.

## Where it sits

Between the data and the [optimiser](optimiser.md): it consumes the same
`DataFetcher`-sourced returns everything else uses, and produces the covariance
a mean-variance problem is stated against.

`RiskModel` carries a `.plot` accessor: `correlation()` draws the matrix on the
`beacon_corr` scale, which is mode-independent by design so two screenshots of
one matrix cannot disagree. See the [gallery](../gallery.md).
