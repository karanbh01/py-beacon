---
title: Charts
description: "Drawing each chart from a result, comparing results, and the light and dark styles."
---

# Charts

Results draw themselves. Each result object has a `.plot` accessor whose
methods draw one chart with matplotlib and return the `Axes` they drew on.
[`beacon.plot.compare`](#comparing-results) puts several results on one
chart, and two styles, light and dark, match Beacon desktop.

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
| `IndexResult` | `constituents(limit=8, ax=None)` | The index drawn over its constituents' closes, all rebased to 100, each constituent named at its line's end; the largest `limit` by weight |
| `IndexResult` | `weights(date=None, ax=None)` | Constituent weights at a rebalance (the latest by default), with the weight cap marked |
| `BacktestResult` | `performance(ax=None)` | Growth of 100 with a linked drawdown panel beneath |
| `BacktestResult` | `constituents(limit=8, ax=None)` | The same chart for the index the run tracked |
| `BacktestResult` | `annual_returns(ax=None)` | Calendar-year returns as green and red bars, each year from the previous year's close and the first from the initial capital (the same figures as `get_annual_returns()`) |
| `AttributionResult` | `contributions(ax=None)` | Each constituent's contribution to return, with the cap and cost drags in the notes |
| `OptimisationResult` | `exposures(ax=None)` | Active weights (optimal minus target), with tracking error and turnover |
| `OptimisationResult` | `frontier(frontier, risk_free_rate=None, ax=None)` | An efficient frontier, its minimum-variance and tangency points and the capital market line |
| `RiskModel` | `correlation(shading="squares", ax=None)` | The correlation matrix as a heatmap, each pair a square or, with `shading="blended"`, the colours blended from cell to cell |

Every method, and `compare`, also takes `title=`, `subtitle=` and `notes=`,
described [below](#titles-notes-and-the-mark). Typing `result.plot` at a
prompt lists the methods, and `result.plot.methods()` returns their names.

Bar charts of weights, contributions and exposures show at most 25 bars, the
largest by size, and say in the notes how many were left out.

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

When a method creates the figure, its size is fixed per chart kind. Every
chart is designed 6in wide: 6 by 4.5 for the time series, annual returns and
the bar charts of names, and 6 by 5.5 for the frontier and the correlation
heatmap. The performance chart is taller, so its upper panel matches the
level chart's plot with the drawdown panel beneath. Charts are drawn at 80%
of these sizes (`beacon.plot.style.SCALE`), lines and spacing with them;
the heading and the small text keep their own sizes, so they stay readable.
The styles save with a standard bounding box rather than a tight one, so a
saved image's pixel size depends only on the figure size and `dpi`; save at
`dpi=300` for a sharp image on a high-density screen.

## Titles, notes and the mark

A chart that creates its own figure is framed the way Beacon desktop frames
one:

- **A title and a subtitle** at the top left: the title in Inter Tight and the
  subtitle beneath it in Source Serif 4 italic, with a thin accent bar down
  their left, level with the y axis.
- **The beta mark**, the glyph Beacon desktop shows in its menu bar, just left
  of the accent bar.
- **The legend**, when the chart has one, at the right of the heading, ending
  where the x axis does and level with the subtitle. It takes as many columns
  as fit beside the title and subtitle and wraps onto more rows when they do
  not; a legend too wide to sit beside the heading goes under it.
- **Notes** at the bottom left, starting under the y axis: one italic line
  beginning "Notes:" that says where the data came from, the dates it covers,
  and anything the chart needs said, such as what its drawdown measures or
  how many bars it left out. The source is named when the data recorded it,
  as a data store does, for example "Source: Synthetic data, backtest dated
  from 03/01/2023 to 31/12/2024."

Each chart has its own title and subtitle. Pass `title=`, `subtitle=` or
`notes=` to replace them, or an empty string to leave one out.
`beacon.plot.frame_text(ax, part)` reads one back, with `part` one of
`"title"`, `"subtitle"` or `"notes"`:

```python
from beacon.plot import frame_text

ax = backtest.plot.performance(subtitle="The CANON tracker, 10 bps a trade",
                               notes="Sample data. Costs of 10 bps on every trade.")

print(frame_text(ax, "subtitle"))
print(frame_text(ax, "notes"))
plt.close("all")
```

Every axis that measures something (a level, a return, a date) ends on its
outermost ticks, so its line stops where its scale does. Where both axes
measure, the lowest label on the y axis is left blank, since it would sit on
the first label of the x axis. The level charts (`level`, `constituents`,
`performance` and `compare`) have no gridlines and draw one line at 100,
where every series starts, so a glance says whether it has gained or lost;
the bar charts and the frontier have no gridlines either, with a line at zero
where a bar can fall below it. The ticks point inwards, and the axes, their
labels, the legend and the notes are in the title's colour. Date axes label
the year at each January and the month between ("2023, Apr, Jul, Oct,
2024"), and a chart's side margins widen to fit its labels, so nothing runs
off its edges.

The fonts (Inter Tight and Source Serif 4) ship with py-beacon under the SIL
Open Font License, so a chart looks the same on every machine without
installing anything.

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

A chart drawn into your own axes gets a plain title and its notes under the
axes rather than the frame, because one figure can hold several charts.

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
  notes, and prints the residual as `0` when the attribution reconciles. The
  cap and cost drags appear in the notes, not as bars, because they are not
  terms in the decomposition.
- **`frontier`** takes an `EfficientFrontier` built over the same universe.
  The capital market line starts at the rate the frontier was traced at, so
  it is tangent at the tangency portfolio. Pass `risk_free_rate` only to draw
  it from another rate.
- **`correlation`** shades from 0.2 to 1.0 on the `beacon_corr` colour map.
  Correlations below 0.2 all get the lowest colour, and the map is the same
  in every theme. `shading="squares"` (the default) draws each pair as its
  own cell; `shading="blended"` runs the colours smoothly from cell to cell,
  which shows the matrix's shape rather than each value.

## Comparing results

`beacon.plot.compare(*results, labels=None, ax=None)` draws two or more
`IndexResult` or `BacktestResult` objects on one axis, and takes `title=`,
`subtitle=` and `notes=` like the chart methods:

```python
from beacon.plot import compare

ax = compare(index, backtest, labels=["Index", "Fund"])
ax.figure.savefig("compare.png", dpi=110)
plt.close("all")
```

Every series is cut to the dates they all share and rebased to 100 on the
first shared date, so the lines start together and differences in history
length do not distort the comparison. The notes say how many shared
observations the comparison rests on.

Labels default to each result's index id or portfolio id. `compare` raises
`ValueError` for fewer than two results or when they share no dates.

## Themes

`beacon.plot.use(theme)` applies a theme to every chart drawn afterwards:

| Theme | Background | Text and axes |
| --- | --- | --- |
| `beacon-light` (or `light`) | Beacon's light canvas | dark |
| `beacon-dark` (or `dark`) | Beacon's dark canvas | light |
| `github-dark` | GitHub's dark page | GitHub's light greys |
| `white` | white | dark |
| `transparent-dark-axes` | none, for a light page | dark |
| `transparent-light-axes` | none, for a dark page | light |

```python
beacon.plot.use("transparent-dark-axes")
ax = index.plot.level()
ax.figure.savefig("level-for-a-light-page.png", dpi=110)
plt.close("all")

beacon.plot.use("light")
```

Every theme draws in Beacon's accent, series and sign colours, which come
from the same design tokens as Beacon desktop, so a chart matches the screen
it sits on; only the background and the text change. A chart records the
theme it was drawn in, so its own colours (the series accent, green and red
for signs, the cap marker, the frame) follow that theme even on a transparent
background. A chart drawn on a figure py-beacon did not make reads the theme
off the figure's background, so one drawn on matplotlib's white default still
gets colours meant for a light page.

Once any chart method, `use()` or `compare()` has run, each theme is
registered with matplotlib as a style (`beacon` and `beacon-dark` for the
Beacon canvases, `beacon-github-dark`, `beacon-white`,
`beacon-transparent-dark-axes` and `beacon-transparent-light-axes`), and the
colour map as `beacon_corr`. They work on charts py-beacon did not draw, and
`beacon.plot.palette(theme)` returns the named colours for your own series:

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

`palette` returns `canvas` (`"none"` for a transparent theme), `surface`,
`border`, `divider`, `text-primary`,
`text-secondary`, `text-muted`, `accent`, `success`, `danger`, `series-2` and
`series-3`. The series colour cycle is accent, `series-2`, `series-3`, then
`text-secondary`. Green and red are left out of it because they mean up and
down elsewhere in the application.

## Interactive charts

There is no interactive backend yet. The `plot-interactive` extra is a
placeholder: it installs plotly, but nothing in py-beacon uses it.

The full API is in the [Charts reference](../reference/plot.md).
