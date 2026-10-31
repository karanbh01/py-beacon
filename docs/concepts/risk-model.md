---
title: Risk model
description: "Covariance estimation and shrinkage, factor risk models and risk contributions."
---

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

## A global factor model

Names in many markets need more than styles: country and currency effects are
a large share of a global stock's risk, and a styles model counts them as each
name's own. `fit_global_model` adds them:

- **Countries and sectors.** Each name has an exposure of 1 to its country's
  factor and its sector's. A country with too few names for a factor of its
  own joins its region's (`assign_groups`), named like `Europe (other)`.
- **Measured against the market.** The countries' returns, weighted by their
  market caps, add up to zero, and so do the sectors'. Without that, the
  market, a country and a sector would each explain the same moves. So a
  portfolio holding the market by cap has no country or sector risk; they
  show in what it holds differently, its active risk.
- **Currencies.** A name priced in another currency than the base has an
  exposure of 1 to that currency's factor, whose return is the currency's own
  return against the base. The other factors are fitted on the names' returns
  in their own currencies, each name weighted by the square root of its cap.
- **Weekly returns.** Markets close at different times, so the same day's
  returns in Tokyo and New York see different news. Over a week that is a
  small part of the period, so `weekly_returns` compounds days into weeks
  ending on Fridays.

```python
import numpy as np

from beacon.risk.model.global_fit import (
    COUNTRY, SECTOR, assign_groups, fit_global_model, group_dummies, weekly_returns)

rng = np.random.default_rng(7)
names = [f"S{i}" for i in range(60)]
domicile = pd.Series(["US"] * 30 + ["DE"] * 24 + ["NZ"] * 6, index=names)
region = domicile.map({"US": "Americas", "DE": "Europe", "NZ": "Asia Pacific"})
sector = pd.Series(["Energy", "Financials", "Utilities"] * 20, index=names)
caps = pd.Series(rng.lognormal(10.0, 1.0, len(names)), index=names)
currency_of = domicile.map({"DE": "EUR", "NZ": "NZD"})
days = pd.bdate_range("2023-01-02", "2024-12-27")
daily = pd.DataFrame(rng.normal(0.0004, 0.01, (len(days), len(names))),
                     index=days, columns=names)
fx_returns = pd.DataFrame(rng.normal(0.0, 0.005, (len(days), 2)),
                          index=days, columns=["EUR", "NZD"])

# NZ has 6 names, too few for its own factor: it joins Asia Pacific (other).
countries = group_dummies(assign_groups(domicile, region, minimum_names=10), COUNTRY)
sectors = group_dummies(sector, SECTOR)
exposures = pd.concat([countries, sectors], axis=1)

world = fit_global_model(weekly_returns(daily), exposures,
                         [list(countries.columns), list(sectors.columns)],
                         caps=caps, currencies=currency_of,
                         currency_returns=weekly_returns(fx_returns))
world.factor_names  # market, the countries, the sectors, then each currency
world.dimensions    # independent factors: one fewer per constrained set
```

A run's analysis takes the global model at each rebalance whose names span
more than one country, fitted on the 104 weeks before it. Each part is a
modelling assumption in the report options' `risk_model` section: the
factors (automatic, global or styles only), where a name's country is read
from (its domicile by default, its listing, its region), the fewest names a
country needs, whether sectors and currencies are factors, whether returns
are weekly, and how many weeks.

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
