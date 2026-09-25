# Charts

Results draw themselves. Each result object has a `.plot` accessor whose
methods draw one chart with matplotlib and return the `Axes` they drew on.
[`beacon.plot.compare`](#comparing-results) puts several results on one
chart, and two styles, light and dark, match the py-beacon application.

Charts need the `plot` extra (matplotlib):

```bash
pip install "py-beacon-kit[plot]"
```

`import beacon` does not import matplotlib. It is loaded the first time a
chart is drawn, so code that never plots never needs it.

Pictures of every chart, in both styles, are in the [Gallery](../gallery.md).

## What each result can draw

| Result | Method | Draws |
| --- | --- | --- |
| `IndexResult` | `level(benchmark=None, ax=None, label="Index")` | The index level rebased to 100, with an optional benchmark series |
| `IndexResult` | `weights(date=None, ax=None)` | Constituent weights at a rebalance (the latest by default), with the weight cap marked |
| `BacktestResult` | `performance(ax=None)` | Growth of 100 with a linked drawdown panel beneath |
| `BacktestResult` | `annual_returns(ax=None)` | Calendar-year returns as green and red bars |
| `AttributionResult` | `contributions(ax=None)` | Each constituent's contribution to return, with the cap and cost drags in the footnote |
| `OptimisationResult` | `exposures(ax=None)` | Active weights (optimal minus target), with tracking error and turnover |
| `OptimisationResult` | `frontier(frontier, risk_free_rate=0.0, ax=None)` | An efficient frontier, its minimum-variance and tangency points and the capital market line |
| `RiskModel` | `correlation(ax=None)` | The correlation matrix as a heatmap |

Typing `result.plot` at a prompt lists the methods, and
`result.plot.methods()` returns their names.

Bar charts of weights, contributions and exposures show at most 25 bars, the
largest by size, and say in the footnote how many were left out.

## Drawing and saving

This builds a small index and a backtest that tracks it from the sample
dataset, then saves two charts.

```python
import matplotlib.pyplot as plt

import beacon.plot
from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.testing import dataset

END = "2024-12-31"
fetcher = dataset.data_fetcher()

definition = IndexDefinition(index_id="CANON",
                             index_name="Canonical Index",
                             base_date=dataset.START,
                             base_value=1000.0,
                             currency="USD",
                             eligibility_rules=[],
                             weighting_scheme=MarketCapWeighted(),
                             rebalancing_frequency="QUARTERLY",
                             calendar="XNYS",
                             universe_identifiers=list(dataset.UNIVERSE),
                             max_constituent_weight=0.20)

index = IndexCalculator(definition, fetcher).run(start_date=dataset.START,
                                                 end_date=END)

backtest = BacktestEngine(start_date=dataset.START,
                          end_date=END,
                          initial_capital=10_000_000.0,
                          data_provider=fetcher,
                          index_result=index,
                          transaction_cost_bps=10.0).run()

beacon.plot.use("light")
print(index.plot)

ax = index.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
ax.figure.savefig("level.png", dpi=110)
plt.close(ax.figure)

ax = backtest.plot.performance()
ax.figure.savefig("performance.png", dpi=110)
plt.close("all")
```

Every method returns an `Axes`, so save through `ax.figure.savefig(...)` or
`plt.savefig(...)`. Close figures you are done with (`plt.close("all")`),
since matplotlib keeps each one in memory until then.

When a method creates the figure, its size is fixed per chart kind: wide for
time series, taller for bar charts of names. The styles save with a standard
bounding box rather than a tight one, so a saved image's pixel size depends
only on the figure size and `dpi`.

## Composing figures

Pass `ax=` to draw into a figure you laid out yourself:

```python
figure, (left, right) = plt.subplots(1, 2, figsize=(13, 4.5))

index.plot.level(ax=left)
index.plot.weights(ax=right)
backtest_axes = backtest.plot.annual_returns()

figure.savefig("index-overview.png", dpi=110)
backtest_axes.figure.savefig("annual-returns.png", dpi=110)
plt.close("all")
```

`performance()` is the exception. It always creates its own two-panel figure,
because the drawdown panel shares the level panel's dates, and it logs a
warning if you pass `ax`. It returns the upper (level) panel.

## Attribution, optimisation and risk

These results come from [Attribution](attribution.md),
[Optimiser](optimiser.md) and [Risk model](risk-model.md). The optimiser
needs the `optimise` extra.

```python
from beacon.analysis import attribute, drifted_weights
from beacon.optimise import (
    FullInvestment, GroupBounds, PositionBounds, minimise_tracking_error,
)
from beacon.optimise.frontier import efficient_frontier
from beacon.risk import estimate_risk_model

prices = dataset.prices().loc[:END]
weights = drifted_weights(index.weight_snapshots, prices)
asset_returns = prices.pct_change().reindex(weights.index)
portfolio_returns = (weights.shift(1) * asset_returns).sum(axis=1)
attribution = attribute(portfolio_returns,
                        weights,
                        asset_returns,
                        cap_drag=-0.003,
                        cost_drag=-0.004)

risk = estimate_risk_model(dataset.returns(), intensity=0.1)

optimisation = minimise_tracking_error(
    dataset.equal_weights(),
    [FullInvestment(),
     PositionBounds(0.0, 0.25),
     GroupBounds("Technology", dataset.sectors()["Technology"], maximum=0.20)],
    risk)

expected = {name: 0.04 + 0.02 * position
            for position, name in enumerate(dataset.UNIVERSE)}
frontier = efficient_frontier(risk, expected, points=12, risk_free_rate=0.02)

charts = {
    "contributions.png": lambda: attribution.plot.contributions(),
    "exposures.png": lambda: optimisation.plot.exposures(),
    "frontier.png": lambda: optimisation.plot.frontier(frontier, risk_free_rate=0.02),
    "correlation.png": lambda: risk.plot.correlation(),
}

for filename, draw in charts.items():
    draw().figure.savefig(filename, dpi=110)
    plt.close("all")
```

A few details worth knowing:

- **`contributions`** adds up the bars it actually drew for the total in its
  footnote, and prints the residual as `0` when the attribution reconciles.
  The cap and cost drags appear in the footnote, not as bars, because they
  are not terms in the decomposition.
- **`frontier`** takes an `EfficientFrontier` built over the same universe.
  Pass the same `risk_free_rate` used to build it, since that is where the
  capital market line starts.
- **`correlation`** shades from 0.2 to 1.0 on the `beacon_corr` colour map.
  Correlations below 0.2 all get the lowest colour. The map is the same in
  both styles.

## Comparing results

`beacon.plot.compare(*results, labels=None, ax=None)` draws two or more
`IndexResult` or `BacktestResult` objects on one axis:

```python
from beacon.plot import compare

ax = compare(index, backtest, labels=["Index", "Fund"])
ax.figure.savefig("compare.png", dpi=110)
plt.close("all")
```

Every series is cut to the dates they all share and rebased to 100 on the
first shared date, so the lines start together and differences in history
length do not distort the comparison. Beneath the chart a table gives each
series' total return, annualised volatility (252 days a year) and maximum
drawdown over that window, and how many shared observations it rests on.

Labels default to each result's index id or portfolio id. `compare` raises
`ValueError` for fewer than two results or when they share no dates.

## Light and dark styles

`beacon.plot.use("light")` or `beacon.plot.use("dark")` applies a style to
every chart drawn afterwards. The colours come from the same design tokens as
the py-beacon application, so a chart matches the screen it sits on.

```python
beacon.plot.use("dark")
ax = index.plot.level()
ax.figure.savefig("level-dark.png", dpi=110)
plt.close("all")

beacon.plot.use("light")
```

Call `use()` before drawing. The chart methods choose some colours (the
series accent, green and red for signs, the cap marker) by reading the
figure's background: the light style's background gets the light colours
and any other background gets the dark ones. Drawing without a beacon style
applied therefore puts the dark colours on matplotlib's white default.

Once any chart method, `use()` or `compare()` has run, the styles are
registered with matplotlib as `beacon` and `beacon-dark`, and the colour map
as `beacon_corr`. They work on charts py-beacon did not draw, and
`beacon.plot.palette(mode)` returns the named colours for your own series:

```python
colours = beacon.plot.palette("light")

with plt.style.context("beacon"):
    figure, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dataset.prices()["AAA"], color=colours["accent"], label="AAA")
    ax.plot(dataset.prices()["BBB"], color=colours["series-2"], label="BBB")
    ax.legend()
    figure.savefig("own-chart.png", dpi=110)

plt.close("all")
```

`palette` returns `canvas`, `surface`, `border`, `divider`, `text-primary`,
`text-secondary`, `text-muted`, `accent`, `success`, `danger`, `series-2` and
`series-3`. The series colour cycle is accent, `series-2`, `series-3`, then
`text-secondary`. Green and red are left out of it because they mean up and
down elsewhere in the application.

## Interactive charts

There is no interactive backend yet. The `plot-interactive` extra is a
placeholder: it installs plotly, but nothing in py-beacon uses it.

The full API is in the [Charts reference](../reference/plot.md).
