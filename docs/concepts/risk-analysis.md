---
title: Risk analysis
description: "A run's factor exposures, value at risk, expected risk at each rebalance, concentration and liquidity."
---

# Risk analysis

The [risk model](risk-model.md) page covers `beacon.risk`, the library that
estimates how assets move together from a returns panel. This page covers
`beacon.analysis.risk`, which applies it to a finished run: what the
portfolio was exposed to, how much it could lose, where its risk came from,
how concentrated and how liquid it was. It is reached as `result.risk` on a
[backtest](backtest.md) result.

```python
import logging

from beacon.backtest import Backtest
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

logging.getLogger("beacon").setLevel(logging.ERROR)  # keep the output short

definition = IndexDefinition(
    index_id="SAMPLE", index_name="Sample Market-Cap Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)
backtest = Backtest(initial_capital=1_000_000.0, transaction_cost_bps=5.0,
                    data_provider=dataset.data_fetcher())
result = backtest.run(definition, start="2023-01-03", end="2024-12-31")
risk = result.risk
```

`result.risk` is a `RiskAnalytics` with five parts, each worked out the first
time it is read and kept:

| Part | Class | Answers |
| --- | --- | --- |
| `risk.factors` | `FactorAnalytics` | What the portfolio was exposed to, and what the factors earned |
| `risk.loss` | `RiskOfLoss` | Value at risk, expected shortfall and stress windows |
| `risk.over_time` | `RiskOverTime` | Expected risk at each rebalance, and where it came from |
| `risk.concentration` | `ConcentrationAnalytics` | Concentration by weight, by risk and by outcome |
| `risk.liquidity` | `LiquidityAnalytics` | How fast the portfolio could be sold |

The factors and traded values are measured from the data the run read, so a
result that kept no data source raises `ValueError` on `result.risk`.

## Factors

A factor is a standardised score for each name. Unless the run's analysis
names its own (`factors=` on `beacon.analysis.request.AnalysisRequest`,
passed as `analysis=` to `Backtest.run`), the factors are
`beacon.strategy.default_factors()`:

| Factor | Score | A higher score means |
| --- | --- | --- |
| `size` | The log of the market cap, negated | A smaller company |
| `value` | Book to price: one over the price-to-book ratio | A cheaper company |
| `momentum` | The return over the last 252 trading days, leaving out the last 21 | A stronger trend |

Momentum needs at least 115 days of returns in its window, or the name has no
score. The sample dataset has no price-to-book ratios, so its `value`
exposures and factor returns are all 0.

**Standardising.** At each rebalance, each factor's values are centred on the
universe's mean weighted by market cap, then divided by the standard
deviation of the centred values across the universe and capped at 3 either
side. The universe is the index's names that day. Centring on the
cap-weighted mean means a portfolio holding the universe by market cap has
no exposure to any factor, so an exposure reads as a tilt against the
market; the sample's market-cap fund has exposures of 0 throughout.
A name held outside the universe is scored on the universe's figures, a
name with no value scores 0, and with fewer than two values in the universe
every name scores 0.

A `Factor` can instead centre on the equally weighted mean
(`centre="equal"`), or within groups (`centre_by="sector"`). A group of
fewer than 5 names is centred on the group one level up, and so on to the
whole universe; `fallbacks(date)` lists which groups fell back and to what.

```python
factors = risk.factors
day = factors.dates[-1]          # the run's rebalances, a list

factors.exposures(day)           # names by factors, at that rebalance
factors.caps(day)                # the market caps the centring used
factors.holdings_exposures()     # the fund's exposure at each rebalance
factors.active_exposures()       # the fund's less the index's
```

**Holdings-based exposures** are the weights times the names' exposures,
from the weights at the close of each rebalance (or each day with
`daily=True`, on the last rebalance's exposures). Cash and names without
exposures add nothing. `holdings_exposures` takes `"fund"`, `"index"` or
`"benchmark"`; `active_exposures(against="index")` is the fund's less a
comparator's.

**Factor returns** are estimated each day from the day after the first
rebalance, by a cross-sectional regression of the universe's returns on
their exposures at the last rebalance before the day, with a market
intercept, each name weighted by its market cap:

    r_i = f_market + Σ_k z_ik f_k + e_i

So a factor's return is what one unit of exposure earned that day with the
other factors held fixed. A day with no more names priced than there are
unknowns is left out. A factor naming its own series (`returns=` on
`Factor`, a vendor's factor index, say) takes that series' daily change
instead.

**Returns-based betas** regress a portfolio's daily returns on the factor
returns, the market's included, by ordinary least squares with an
intercept. The fund and the comparators are taken in excess of the
risk-free rate; `"active"` (the fund's return less the index's) is not. At
least 20 common days are needed.

```python
factors.factor_returns()           # a row a day, the market's first
fit = factors.regression()         # "fund", a comparator such as "index", or "active"
fit.betas, fit.r_squared, fit.alpha_a_year
factors.rolling_betas(window=126)  # each beta over the trailing 126 days
```

`regression` returns a `FactorRegression`: `coefficients` gives each
factor's (and the intercept's, `alpha`) coefficient, standard error and
t-statistic; `alpha_a_year` is the daily intercept times the days in a year.
`rolling_betas` defaults to a 252-day window, a row for each window's last
day.

## Value at risk and expected shortfall

Both are losses, reported as positive numbers: a one-day 99% value at risk of
0.02 says the portfolio loses more than 2% on one day in a hundred. Expected
shortfall is the mean loss on the days worse than that.

```python
risk.loss.books()           # ['fund', 'index', 'active']
risk.loss.table()           # the fund's, every method, level and horizon
risk.loss.table("active")   # the fund's return less the index's
```

The table has a row for each measure and method and a column for each level
and horizon:

    level                     0.95            0.99
    days                        1       10      1       10
    measure method
    var     historical      0.0132  0.0417  0.0186  0.0588
            normal          0.0126  0.0398  0.0181  0.0574
            cornish_fisher  0.0126  0.0399  0.0183  0.0578
    es      historical      0.0163  0.0515  0.0203  0.0641
            normal          0.0160  0.0506  0.0209  0.0661

| Method | Value at risk | Expected shortfall |
| --- | --- | --- |
| `historical` | Minus the returns' `1 - level` quantile, linearly interpolated | Minus the mean of the returns at or below that quantile |
| `normal` | `-(mean + z × sd)`, with `z` the normal's `1 - level` quantile | `-mean + sd × φ(z) / (1 - level)` |
| `cornish_fisher` | The normal's, with `z` adjusted for the returns' skew and excess kurtosis | Not given |

Cornish-Fisher lets fat tails and a lean to one side count. The expansion
holds for moderate tails (skew within about 1 either side, excess kurtosis up
to about 6); far beyond them it can fall below even the normal figure, and
the historical one is the better guide.

The ten-day figure is the one-day figure times √10, the Basel convention,
not a measurement on ten-day returns. Each figure is measured over the
analysis window (`start` and `end` in the analysis settings) and is `None`
with fewer than 20 returns. `table` takes other `levels` and `horizons`.

`stress_windows()` gives each crisis the run saw: its `crisis`, `start`,
`end` and whether it was `partial`, with each portfolio's return and
deepest fall over it (`fund_return`, `fund_drawdown`, `index_return`, ...).
Crises are the market's falls of 15% or more from a peak, the market being
the benchmark when the run has one and the index otherwise, or named
episodes passed as `named=`. The sample's market never fell 15%, so its
table is empty.

## The risk model at each rebalance

Expected risk comes from a factor model estimated once at each rebalance on
the returns before it, never including that day, for every name the fund,
its index or its benchmark held:

- **Styles.** The market and the factors above, fitted on the 252 sessions
  of daily returns before the rebalance, on the exposures that day.
- **Global.** At a rebalance whose names span more than one country, the
  styles with country, sector and currency factors beside them, fitted on
  the 104 weeks of weekly returns before it. See
  [a global factor model](risk-model.md#a-global-factor-model) for how it is
  estimated. A run whose reference data has no country field, like the
  sample's, always uses the styles model.

A name whose returns do not cover the whole window (one that listed during
it, say) is left out of that rebalance's model rather than shortening the
window for every name. `risk.over_time.models.uncovered(date)` lists them,
and `covered_weight` below says how much of the portfolio the model
describes.

**The first rebalances may have no expected risk.** A model needs its whole
window of returns before the rebalance, and the run's returns start with the
run, so the styles model starts a year in and the weekly global model two
years in. `risk.over_time.dates()` gives the rebalances that have one
(`models.has_history(date)` for any one); asking for another raises
`CalculationError`. The sample starts in January 2023, so its first is
2024-04-01.

`RiskModelSettings` (in `beacon.analysis.risk.model`) holds the choices, and
a [report](reports.md) takes the same ones as its options' `risk_model`
section:

| Setting | Default | What it sets |
| --- | --- | --- |
| `window` | 252 | Sessions of daily returns before each rebalance, when fitted on daily returns |
| `structure` | `"auto"` | `"global"`, `"styles"`, or `"auto"`: global when the names span more than one country |
| `country_field` | `"COUNTRY_DOMICILE"` | The reference field a name's country is read from (`COUNTRY_LISTING`, `REGION`, ...); none for no country factors |
| `minimum_names` | 10 | The fewest names a country needs for its own factor; one with fewer joins its region's |
| `sectors`, `currencies` | `True` | Whether the global model has sector and currency factors |
| `weekly` | `True` | Whether the global model is fitted on weekly returns, or on `window` sessions of daily ones |
| `weeks` | 104 | Weeks of returns before each rebalance, for the weekly model |

`result.risk` uses the defaults. For other choices, build a `RiskOverTime`
on the run's factors:

```python
from beacon.analysis.risk.contributions import RiskOverTime
from beacon.analysis.risk.model import RiskModelSettings

half_year = RiskOverTime(risk.factors, settings=RiskModelSettings(window=126))
half_year.dates()   # from 2023-10-02: half a year of returns is enough
```

## Expected risk over time

```python
over_time = risk.over_time
over_time.summary()             # a row a rebalance with a model
day = over_time.dates()[-1]
over_time.expected(day)         # the same figures for one rebalance
over_time.contributions(day)    # by name
over_time.group_contributions(day, "SECTOR")
over_time.measured()            # over the run, from its returns
```

`expected(date, book="fund", against="index")` holds the portfolio at its
weights at the close of the rebalance. The model's covariance is
`B F Bᵀ + D`, so the variance splits exactly into a factor part and a
specific part:

| Figure | Meaning |
| --- | --- |
| `volatility` | Expected annualised volatility |
| `factor_volatility`, `specific_volatility` | The square roots of its two variance parts: their squares add up to the volatility's square, not the figures themselves |
| `factor_share` | The factor part's share of the variance |
| `tracking_error`, `factor_tracking_error`, `specific_tracking_error` | The same on the active weights against `against` |
| `diversification_ratio` | `Σ w_i σ_i / σ`: the weighted names' own volatilities over the portfolio's; 1 means no diversification |
| `average_correlation` | The single correlation that, with these weights and volatilities, gives the portfolio's variance |
| `covered_weight` | The share of the portfolio's weight in names the model covers |

`summary()` is `expected` at every rebalance in `dates()`.

`contributions(date, book="fund", against=None)` splits the expected
volatility by name (or, with `against`, the expected tracking error):
`weight`, `marginal` (the change in volatility per unit of weight),
`contribution` (weight times marginal, adding up to the volatility) and
`share` (adding up to 1), largest contribution first. A contribution can be
negative: a name that hedges the rest reduces risk. `group_contributions`
adds them up by a reference field as it stood on the day, `"Unknown"` for a
name without one. See [risk contribution](attribution.md#risk-contribution)
for the decomposition itself.

`measured(group=None)` looks back instead. Each name's daily contribution to
the fund's return is its weight at the previous close times its return that
day, and `other` is whatever else moved the fund (cash, costs, futures). A
contribution's `share` of the fund's measured variance is its covariance
with the fund's return over that variance, and the shares add up to 1
exactly; `volatility` is the share times the fund's measured annualised
volatility. Set it beside the expected contributions to see whether the
names that were expected to carry the risk did.

## Concentration

```python
concentration = risk.concentration
concentration.by_weight()                    # at each rebalance's close
concentration.by_weight(field="SECTOR")      # across sectors
concentration.curve(day)                     # names largest first
concentration.by_risk()                      # from the expected risk
concentration.absorption_ratio(day)
concentration.by_outcome()                   # over the whole run
```

**By weight.** The weights above 0 are scaled to add up to 1, so cash is
left out:

| Figure | Meaning |
| --- | --- |
| `names` | Names held |
| `herfindahl` | `H = Σ w_i²` |
| `normalised_herfindahl` | `(H - 1/N) / (1 - 1/N)`: 0 equally weighted, 1 a single name |
| `effective_number` | `1 / H`, the equally weighted names that would be as concentrated |
| `largest`, `top_5` ... `top_25` | The largest weight, and the top 5, 10, 20 and 25 together |
| `names_for_50`, `names_for_80`, `names_for_90` | The fewest names making that share of the portfolio |
| `gini`, `entropy`, `entropy_effective_number` | With `optional=True`: the Gini coefficient, `-Σ w_i ln w_i` and its exponential |

`by_weight` takes `book` (`"fund"`, `"index"`, `"benchmark"`), `field` for
group weights and `daily=True` for every day; `against("index")` is the
fund's figures less the index's. `curve(date)` lists the names largest
first with each one's weight, the cumulative weight, and the Herfindahl and
effective number of the top k on their own, with the change the k-th name
made to each. `by_period("Y")` (or `"M"`) and `in_crises()` average the
daily figures over each year or month, and inside each crisis and on the
calm days outside them. `drift()` is each day's half sum of the absolute
gaps between the weights and the last rebalance's: the share of the
portfolio that would trade to put it back.

**By risk**, at each rebalance in `risk.over_time.dates()`, from each name's
share of the expected volatility: their Herfindahl (`risk_herfindahl`), the
effective number of risk sources (`1 / Σ p_i²`), the top names' share of the
risk, and the diversification ratio and average correlation.

**The absorption ratio** is the share of the total variance of the model's
names (all the fund, index and benchmark held) in the risk model's largest
eigenvalues, one per independent factor by default: the factor covariance's
rank, which is one fewer per constrained set in a global model, and leaves
out a factor whose exposures are all 0, as the sample's value is. It reads
how much of the risk the common drivers absorb. It depends on the names held,
not on their weights; pass `components=` to count a different number.

**By outcome**, over a window (`start`, `end`, the whole run by default):
`return`, the daily contributions added up (not compounded), the top 10
contributors' share of it (`top_10_share`), and `breadth`, the share of the
names with a contribution whose contribution was positive.

## Liquidity

At each rebalance and month end (`risk.liquidity.dates()`), a holding worth
`V` is sold at no more than a share of the name's average daily traded value
`ADV` a day:

    days = V / (participation × ADV)

and within `d` days the portfolio can sell `Σ min(V, d × participation × ADV)`.
The traded value is the close times the volume, in the run's currency, over
the name's last 20 days with a volume, read from the data the run used. A
holding's value is its weight times the fund's net assets, for the index and
benchmark as well, so all three read as if the fund held them. A name with no
traded value cannot be sold at all: its days are infinite and it is counted
apart.

```python
liquidity = risk.liquidity
liquidity.days_to_sell(day)   # each holding's value, traded value and days, slowest first
liquidity.profile()           # a row a date
liquidity.traded_values()     # dates by names
```

`profile()` gives `sellable_1d`, `sellable_5d` and `sellable_20d` (the
shares sellable within each horizon), `days_to_sell_all` (the slowest name
with a traded value), `mean_days` (weighted by value) and
`no_traded_value` (the share in names that cannot be sold). The defaults are
the liquidity options' (participation 20%, 20 days, horizons 1, 5 and 20);
for others, build one directly:

```python
from beacon.analysis.risk.liquidity import LiquidityAnalytics

cautious = LiquidityAnalytics(result.run_data, dataset.data_fetcher(),
                              participation=0.05)
cautious.profile(horizons=(1, 2))
```

## Where it sits

Downstream of the run: it reads a finished `BacktestResult` (its weights,
returns, reference snapshots and comparators) and the `DataFetcher` it read,
and calculates rather than re-simulates. The covariance and factor model
estimation is [`beacon.risk`](risk-model.md); return attribution and the
risk decomposition are in [attribution](attribution.md); the return,
drawdown and ratio figures are in [performance](performance.md), and what
the portfolio held and traded in [portfolio](portfolio.md). The analysis
options behind each choice above are shown and set in [reports](reports.md).

The full API is in the [analysis reference](../reference/analysis.md).
