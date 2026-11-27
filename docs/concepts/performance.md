---
title: Performance
description: "A run's returns, risk-adjusted ratios, drawdowns and figures against its comparators."
---

# Performance

`beacon.analysis.performance` measures how a run did: its returns, its risk,
its drawdowns, and its figures against the index it tracked or any other
series. `PerformanceAnalytics` gives the run's own figures; `RelativeAnalytics`,
`CrisisAnalytics` and `RegimeAnalytics` measure it against one comparator.

Every figure is computed with `beacon.analysis.performance.formulas`, which
holds each formula once as a pure function of returns or levels. A backtest's
`summary()`, a report and a stored run's analysis all call the same functions,
so they cannot disagree.

## A run's analytics

A finished backtest's analytics are `result.analytics`, the way in to every
calculation on the run, built on first read and kept. Its performance is
`result.analytics.portfolio.performance`: each figure a method named by its
statistic, giving the fund's over the whole run, with keywords for another
portfolio, window or form.

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

performance = result.analytics.portfolio.performance
print(performance.total_return())            # 0.457, the fund's over the run
print(performance.sharpe_ratio(of="all"))    # the fund's and its index's
print(performance.total_return(by="year"))   # a Series, a year a row
```

`PerformanceAnalytics` gives the same figures from a `RunData`, together:

```python
from beacon.analysis.performance import PerformanceAnalytics

analytics = PerformanceAnalytics(result.analytics.run_data)
print(analytics.headline())   # total_return 0.457, sharpe_ratio 1.53, ...
print(analytics.summary())    # with the monthly hit rate and tracking figures
```

The analytics read a `RunData`, everything the analysis of a run needs:
`result.analytics.run_data` for a finished run. Written to a folder, it reads
back exactly as it was, whatever has happened to the data since, so a run
stored months ago gives the same figures as when it finished.

```python
from pathlib import Path

from beacon.analysis import RunData

result.analytics.run_data.write(Path("stored-run"))
stored = PerformanceAnalytics(RunData.from_record(Path("stored-run")))
print(stored.headline() == analytics.headline())   # True
```

## What the figures are measured over

`AnalysisSettings` says what a run's figures are measured over and against.
Every field left as None takes the run's own.

| Setting | Means | The run's own |
| --- | --- | --- |
| `start`, `end` | The first and last day measured | The run's first and last day |
| `periods_per_year` | How many daily returns make a year | The run's `ModellingAssumptions`, 252 by default |
| `risk_free_rate` | The annual rate the Sharpe ratio and excess returns are measured over | The run's `ModellingAssumptions`, 0 by default |
| `tracked` | The comparator tracking is measured against: `index`, `optimised` or `benchmark` | The optimised index when the run solved one, else the index |

```python
from beacon.analysis.performance import AnalysisSettings

year = PerformanceAnalytics(result.analytics.run_data,
                            AnalysisSettings(start="2024-01-01", risk_free_rate=0.04))
print(year.headline()["sharpe_ratio"])
```

For `result.analytics` to use settings, pass them in the run's analysis
request: `Backtest.run(..., analysis=AnalysisRequest(options=AnalysisOptions(window=settings)))`,
with `AnalysisRequest` from `beacon.analysis.request` and `AnalysisOptions`
from `beacon.analysis.reports.options`. See [reports](reports.md) for what
else a request decides.

Over the whole run, the first return runs from the initial capital to the
first day's close, so **the cost of buying in on the first day counts** in the
total return, volatility and drawdown. Over a window that starts later, the
figures run from the close before its first day. With flows, every return is
a unit's (time-weighted), so money arriving is not a return.

## Definitions

The definitions a reader needs to read the figures correctly:

| Figure | Definition |
| --- | --- |
| Annualised return | `(1 + total) ^ (periods_per_year / days) - 1` |
| Volatility | The sample standard deviation of the daily returns, times `√periods_per_year` |
| Sharpe ratio | The mean daily excess return over its standard deviation, times `√periods_per_year` |
| Tracking error | The volatility of the active returns (the fund's daily return less the comparator's) |
| Tracking difference | The fund's compounded return less the comparator's |
| Information ratio | The mean active return over its standard deviation, times `√periods_per_year` |
| Maximum drawdown | The largest fall from a running peak, measured from the starting value |
| Correlation, beta | Of the daily returns over the days both series have; beta is their covariance over the comparator's variance |
| Money-weighted return | The annual internal rate of return of the investors' cash flows, over calendar days (ACT/365); only for a run with flows, and None when the flows do not change sign |

**The Sharpe and information ratios are a daily average over a daily spread**,
annualised by the square root of the days a year. Both halves come from the
same returns, so the ratio does not change with leverage, and multiplied by
the square root of the years it is the t-statistic of the mean.

**The Sharpe ratio's risk-free rate is the daily rate that compounds to the
annual one**, `(1 + rate) ^ (1 / periods_per_year) - 1`, subtracted from each
day's return. Every excess return in the module (Sortino, the regressions,
the bootstrap) uses the same daily rate.

**A series that does not vary has nothing to relate.** A spread of at most
`1e-12` a day counts as none (a perfectly smooth series has a standard
deviation of about `1e-17` in floating point). Then the Sharpe ratio,
correlation and beta are 0, and the information ratio is None, as there is no
active risk to measure it against. A figure that cannot be measured, from too
few returns, is None, unless its definition says otherwise: volatility and the
Sharpe ratio are 0 with fewer than two.

## Ratios and drawdowns

`ratios()` gives the risk-adjusted ratios beyond the Sharpe ratio. A ratio
over a risk that is not there (no day below the target, no drawdown) is None.

| Ratio | What it is |
| --- | --- |
| `sortino` | The mean excess return over the downside deviation (the root mean square of the shortfalls below the risk-free rate, every day counted), annualised |
| `kappa_3` | The mean excess return over the cube root of the third lower moment, which weighs larger shortfalls more. Daily, not annualised |
| `upside_potential` | The mean gain above the risk-free rate over the downside deviation |
| `omega` | The sum of gains above the risk-free rate over the sum of shortfalls below it |
| `calmar` | The annualised return over the deepest drawdown |
| `martin` | The annualised mean excess return over the ulcer index |
| `sterling` | The annualised return over the mean of the three deepest drawdowns |
| `burke` | The annualised return over the root of the sum of every drawdown squared |
| `gain_to_pain` | The sum of monthly returns over the losing months' sum |
| `probabilistic_sharpe` | The probability the true Sharpe ratio is above `sharpe_benchmark` (0 a year unless given), allowing for skew and fat tails |
| `deflated_sharpe` | The same against the best Sharpe ratio `trials` strategies would show by luck; None unless `trials` and `trial_variance` are given |
| `lo_sharpe` | The Sharpe ratio corrected for serial correlation over 10 days of lags |

```python
ratios = analytics.ratios(trials=20, trial_variance=0.25)
print(ratios["sortino"], ratios["deflated_sharpe"])

drawdowns = analytics.drawdowns()
deepest = drawdowns.worst[0]
print(deepest.peak.date(), deepest.trough.date(), deepest.depth, deepest.recovery)
print(drawdowns.time_below_peak, drawdowns.ulcer_index)
```

A drawdown starts on a peak, reaches its trough and ends on the first day the
level is back at the peak; one still open on the last day has no recovery and
is `current_drawdown`. Lengths are in trading days from the peak.
`drawdowns(worst=5)` returns a `DrawdownFigures` with the deepest `worst`,
the `current` fall below the peak, the average depth and length, the share of
days below a peak, the longest drawdown, the recovery factor (total return
over the deepest), the ulcer index (the root mean square of the daily
drawdown) and the pain index (its mean size). `drawdown_series()` is the
daily drawdown itself, and `levels()` the level everything is measured along.

## Return figures

| Method | Gives |
| --- | --- |
| `trailing_returns()` | `1M`, `3M`, `6M`, `1Y`, `3Y`, `5Y`, `10Y`, `MTD`, `QTD`, `YTD` and `inception`, each a `WindowReturn`; None for a window longer than the run |
| `trailing_table()` | The same as a table, for the fund, its index and benchmark, and the fund's less each's |
| `periods(frequency)` | Each calendar year, quarter or month (`annual`, `quarterly`, `monthly`) with the headline figures and each comparator's |
| `period_table(frequency)` | The same as a table |
| `monthly_grid(of)` | The calendar grid of monthly returns, a row a year, with each year's; `of` is `fund`, a comparator's name, or `difference index` |
| `month_of_year()` | The mean return in each calendar month over the years |
| `best_and_worst()` | The best and worst day, week, month, quarter and year, each marked `partial` when the run covered only part of it |
| `positive_shares()` | The share of days, months and years that gained |
| `gains_and_losses(frequency)` | The average gain, average loss and their ratio |
| `longest_streaks()` | The longest run of months that all gained, and that all lost |
| `means()` | The arithmetic and geometric mean daily return, and each a year |

A trailing return runs from the close on or before the same date that many
months back from the last day. **Only windows longer than a year are
annualised** (3 years and more, and since inception when it spans more than a
calendar year), since a shorter return annualised overstates what was earned.
Since inception is the headline's total return, so over the whole run it
includes the first day's cost. Returns to date run from the close before the
month, quarter or year began. Calendar periods compound their own days, so the
months compound to the year and the years to the run.

The main return and risk figures over the same windows, of every portfolio,
are `result.analytics.table([...], trailing=[...], of="all")`: each figure is
worked out as the whole run's is, on the window's days, and an annualised one
is blank over a window too short for it (below). The figures that take
`trailing` are the total and annualised return, volatility, maximum drawdown,
downside deviation, skew, excess kurtosis, tail ratio, Sharpe ratio, and
against a comparator the tracking difference, tracking error, downside
tracking error, beta and up and down capture. Every analysis holds two such tables
(`beacon.analysis.statistic_tables`): its `summary`, the main figures over
every window, and `risk`, the risk figures over the last 1, 3 and 5 years and
since inception.

## Annualised figures over short windows

A figure annualised over a few weeks says more about the scaling than the
fund, so **it is left blank over a window too short for it**, wherever it is
asked for: over the whole run, a `start` and `end`, each period of `by`, each
window of `trailing` and each `rolling` window, in `result.analytics` and in
the analysis the server gives.

| Figures | Need a window of at least | Set by |
| --- | --- | --- |
| Rates compounded or scaled to a year: the annualised return and annualised relative return, alpha, and the Calmar, Martin, Sterling, Burke and Treynor ratios | A year | `min_annualised_rate_window` |
| Ratios scaled by the square root of a year: the Sharpe, Sortino, information and Lo Sharpe ratios | Six months | `min_annualised_ratio_window` |
| Spreads: volatility, tracking error, residual risk and downside tracking error | Any window | |

Both thresholds are modelling assumptions, recorded with each run, and
`AnalysisSettings` sets them for one analysis. Each is a number of months or
years, such as `"3M"` or `"2Y"`, and `"0M"` turns it off. A window counts as
long enough when it holds the threshold's daily returns less a week's, since
a calendar year holds 250 to 253 trading days. Only the annualised figure of
alpha is left blank: its t-statistic and p-value are given over any window.

The rule holds too where figures are given together: `result.summary()` and
the backtest metrics the server gives with a run, a comparison of the run with
its index or benchmark, and each calendar period's figures
(`PerformanceAnalytics.periods()`, the period table and
`GET /strategies/{id}/periods`), where each quarter's and month's Sharpe ratio
is blank, as a year's under six months is. There the figure is null.

Every annualised analytic takes `annualise`, which asks for it another way:

```python
performance = result.analytics.portfolio.performance

performance.sharpe_ratio(by="quarter")                    # blank: a quarter is too short
performance.sharpe_ratio(by="quarter", annualise=True)    # annualised all the same
performance.sharpe_ratio(by="quarter", annualise=False)   # per day, never blank
```

`annualise=False` gives the figure per day: it is measured with a year of one
period, so the Sharpe ratio is the mean daily excess return over its spread,
the volatility the daily standard deviation, and the annualised return the
daily rate that compounds to the total. The risk-free rate is taken as the
daily rate that compounds to it.

A name's risk over a short period says little for a like reason: a
volatility, beta or correlation over a few days is mostly noise. A name's
profile (`instrument.profile(name).period_statistics`) leaves its volatility, beta,
correlation and share of volatility blank over a period with fewer daily
returns than `min_risk_days`, 5 unless set, and says so in the part's
`attrs["unavailable"]`. It is a modelling assumption too, recorded with each
run, and `AnalysisSettings(min_risk_days=...)` sets it for one analysis; it
is a whole number of days, 2 or more, as a standard deviation needs two.

```python
print(analytics.trailing_table().loc[["1Y", "YTD", "inception"]])
print(analytics.monthly_grid().round(4))
best, worst = analytics.best_and_worst()["monthly"]
print(best.period, best.value, worst.period, worst.value)
```

## Distribution and bootstrap

`distribution()` describes the shape of the daily returns: the downside
deviation (below the risk-free rate) and semi-variance (below the mean), both
annualised; skew and excess kurtosis, bias-corrected as a spreadsheet's are;
the tail ratio (the 95th percentile over the size of the 5th); the serial
correlation at lags of 1 to 5 days; and a `Histogram` of 50 bins with the
counts a normal distribution of the same mean and spread would give.

The `return_distribution` statistic counts the fund's, the index's and the
benchmark's returns into the same bins, so they can be drawn on one chart.
Each portfolio's count is a share of its returns. A `normal` column gives the
share a normal distribution of the fund's mean and standard deviation would
put in each bin. The returns are daily unless the `histogram_horizon` setting
(in the `returns` options) asks for more days. Each return then compounds that
many days, in periods that do not overlap, and a last period that falls short
is dropped.

`bootstrap()` says how far the annualised return and the Sharpe ratio could
have come out otherwise. It draws the run's own days again, 1,000 runs of the
same length by default, in blocks of `block` days to keep any dependence
between neighbouring days, and returns each figure as an `Interval` holding
95% of the resampled runs. The same `seed` draws the same runs.

```python
interval = analytics.bootstrap(block=5)["sharpe_ratio"]
print(interval.value, interval.low, interval.high)
```

## Rolling figures

`rolling(window)` gives each day's return, volatility, Sharpe ratio, Sortino
ratio and maximum drawdown over the `window` days ending that day (252, a
year, by default; 63 is a quarter), blank until the first full window. Each uses
the same definition as the static figure. With a `comparator`, by name or by
its levels, it adds the relative return, tracking error, information ratio,
beta and correlation. `rolling_outcomes()` summarises every full window: the
worst, median and best return, the share that gained and the share ahead of
the comparator.

As with the static figures, a rolling Sharpe ratio, beta or correlation over
a spread of none is 0, and a rolling information ratio is blank. Against its
own index the sample is one such case: after the first day's buying cost it
tracks to rounding, so every full window's information ratio is blank.

```python
print(analytics.rolling(63).dropna().tail(2))
```

## Risk rating

`beacon.analysis.performance.risk_rating` rates a fund as a UCITS factsheet
does, on the synthetic risk and reward indicator (SRRI). The volatility of
its weekly returns over the last five years, annualised, is placed on a
scale of 1 to 7. A run shorter than five years uses all of its weeks. Weeks
end on Friday, each compounding its days, so the first and last may be part
weeks.

| Rating | Volatility |
| --- | --- |
| 1 | under 0.5% |
| 2 | 0.5% to under 2% |
| 3 | 2% to under 5% |
| 4 | 5% to under 10% |
| 5 | 10% to under 15% |
| 6 | 15% to under 25% |
| 7 | 25% and over |

Its tracking error against the tracked comparator, over the same weeks,
puts it in a band: passive under 0.5%, enhanced from 0.5% to under 2%,
active from 2% to under 5%, and unconstrained 5% and over. The
`risk_rating` statistic gives the fund's rating and band and each
comparator's rating. `risk_distribution` counts the days the fund spent at
each rating and band, on the volatility and tracking error of daily returns
over the year ending each day. The run's first year is not counted, and
because these are daily figures, the last day can sit at a different rating
from the headline one. Both need a year of daily returns, and are
`too_short` without it. The `risk_rating` options section holds the
volatility bounds, the tracking error bounds and the years measured.

```python
from beacon.analysis.performance.risk_rating import RiskRatingSettings, risk_rating

table = risk_rating(analytics.returns, analytics.tracked, {}, RiskRatingSettings(),
                    analytics.periods_per_year)
print(table)
```

## Risk and return

Every name's risk and return, for one chart of risk against return, is the
per-name table at weekly returns,
`result.analytics.instrument.table(frequency="weekly")`: a row for every name
any portfolio held on any day of the run, not only its latest members. Daily
returns are compounded into weeks ending on Friday. A name's annualised
return is its mean weekly return times 52, and its volatility is the standard
deviation of its weekly returns times the square root of 52, each over the
weeks it has. The mean rather than the compounded return is what a
mean-variance frontier is drawn in, so a single name sits on or inside the
frontier. A name with under a year of weeks keeps its row, with its
annualised return blank. Each row also gives when the name was held, its
weights and whether it was in the index at the end.

`portfolio_risk_return` (`beacon.analysis.performance.members`) measures the
fund, the index and the benchmark the same way, so their markers sit on the
names' scale. `efficient_frontier` traces the
long-only, fully invested mean-variance frontier at 50 points, from the
minimum-variance portfolio, which is flagged, to the best name. It uses the
100 largest latest index members that have a return in every week, with their
sample covariance and mean returns, so a late listing cannot shorten every
name's weeks. It is drawn with returns the run went on to see, so it is a
frontier in hindsight, not one an investor could have held. It needs a year
of weeks, and more weeks than names, and is `too_short` otherwise. Its
summary gives the names it was drawn from and the weeks they cover. The frontier
reads the run's name returns and index weights. Over 6,000 synthetic names,
5,000 of them in the index, it took about six seconds.

## Against a comparator

The sample run tracks its index almost exactly, so its relative figures are
near zero. An equal-weighted index over the same names gives something to
compare against.

```python
from beacon.index.calculation import IndexCalculator
from beacon.index.methodology import EqualWeighted

equal = IndexDefinition(
    index_id="EQUAL", index_name="Sample Equal-Weighted Index",
    base_date="2023-01-03", base_value=1000.0, currency="USD",
    eligibility_rules=[], weighting_scheme=EqualWeighted(),
    rebalancing_frequency="QUARTERLY", calendar="XNYS",
    universe_identifiers=["AAA", "BBB", "CCC", "DDD", "EEE"],
)
levels = IndexCalculator(equal, dataset.data_fetcher()).run(end_date="2024-12-31").index_levels

comparison = analytics.comparison(levels)
print(comparison.own["total_return"])                 # the comparator's own figures
print(comparison.relative["information_ratio"])       # the run's against it
print(analytics.rolling(63, comparator=levels)["information_ratio"].dropna().tail(2))
```

`comparison(comparator)` takes a comparator's name (`index`, `optimised`,
`benchmark`) or its levels, and measures over the days both have. Its `own`
holds the comparator's headline figures and monthly hit rate; its `relative`
the relative and annualised relative return, tracking error, information
ratio, correlation, beta and the share of months the run beat it. The
comparator's first day returns nothing, from its level that day, and when it
starts after the run, the run's first day does too.

`RelativeAnalytics(run, comparator)` goes further against one comparator:

| Method | Gives |
| --- | --- |
| `regression()` | Jensen's regression of the daily excess returns on the comparator's: `alpha` (a year), `beta`, `r_squared`, `alpha_t`, its two-sided `alpha_p`, and `residual_risk` a year. None with fewer than three days. Against a comparator that does not vary the `beta` is 0, as `formulas.beta` gives it, and `alpha` is the mean excess return |
| `treynor()` | The annual mean excess return per unit of beta |
| `capture()` | Up and down capture on months, as Morningstar measures them: the fund's geometric mean monthly return in the months the comparator rose (fell) over the comparator's, and their ratio |
| `batting_average()` | The share of months the fund beat the comparator |
| `information_ratio_t()` | The information ratio times the square root of the years, and its two-sided probability by chance |
| `downside_tracking_error()` | The root mean square of the days the fund fell behind, every day counted, a year |
| `relative_levels()`, `relative_drawdowns()` | The fund's wealth over the comparator's, and its stretches of falling behind |
| `tracking_difference_by_year()` | Each calendar year's return less the comparator's, and the standard deviation of those differences |
| `figures()` | All of these by name, with the comparison's relative figures |

```python
from beacon.analysis.performance import RelativeAnalytics

relative = RelativeAnalytics(result.analytics.run_data, levels)
print(relative.regression())
print(relative.capture(), relative.batting_average())
```

## Crises and regimes

`CrisisAnalytics` measures how the run held up in crises. By default a crisis
is each fall of a broad market series deeper than 15%, from its peak to its
trough. Pass `named=` a list of `Episode(name, start, end)` to use your own
instead; `NAMED_CRISES` holds eight, S&P 500 closing peak to closing trough.
Both are in `beacon.analysis.performance.crises`. The market is the run's
benchmark when it has one, else its index, or any series given as `market=`;
it need not be the comparator.

Each `Crisis` gives the fund's and comparator's return over the days after
the start through the end, the outperformance, each's deepest fall inside,
the beta over the window, the alpha beyond what the whole run's beta
predicts, the returns 3 and 6 months after the trough, and the trading days
the fund took to get back to its value at the start, and the day it did. A
window the run covers only part of is marked `partial`. Its `status` is
`covered` or `partial`. `unseen()` lists the episodes the run saw none of, each
as `before_run` or `after_run`. `summary()` gives the average
outperformance, the share of crises outperformed, the crisis capture and the
pooled alpha over every crisis day, a year.

`RegimeAnalytics` splits the run's days by the market's behaviour: months it
rose and fell, the calmest and most volatile fifth of days by 63-day rolling
volatility, bear markets (falls of more than 20%) and corrections (10% to
20%). Each `Regime` compounds the fund's and comparator's returns over its
days, as if invested on those days alone, annualises their mean so regimes of
different lengths compare, and gives the beta and Jensen's alpha (over the
risk-free rate) over them. A regime the run never saw is None. A crisis's
alpha, by contrast, is a total over the window with no risk-free rate.

```python
from beacon.analysis.performance import CrisisAnalytics, RegimeAnalytics

run = result.analytics.run_data

crises = CrisisAnalytics(run, levels, market=levels, threshold=0.05)
print(len(crises.crises()), crises.summary())

regimes = RegimeAnalytics(run, levels, market=levels).regimes()
print(regimes["down months"])
```

The sample's market never fell 15%, so the example lowers the threshold to 5%
and has no bear markets or corrections.

In a report, the `crises` options choose the list with `episodes`: `market`
(the market's falls, the default) or `named` (`NAMED_CRISES`). With the named
list, `crisis_returns` keeps a row for every crisis, with its status, and a
crisis the run saw none of has no figures. `crisis_paths` gives the fund's,
the index's and the benchmark's levels through each crisis, rebased to 100 at
its start. A crisis the run joined late starts at 100 on the run's first day.
Each path runs on past the crisis's end: for at least three months, then to a
month after the fund's recovery, and for at most eighteen months, stopping at
the run's end. Each day's `phase` is `crisis` through the end and `after`
beyond it.

## Where it sits

Downstream of the [backtest](backtest.md): it reads a finished or stored run
and recomputes nothing. What the run held, traded and paid is in
[portfolio](portfolio.md); its risk in [risk analysis](risk-analysis.md); the
sources of its return in [attribution](attribution.md). Which figures a
report shows, and how it lays them out, is in [reports](reports.md).

The full API is in the [analysis reference](../reference/analysis.md).
