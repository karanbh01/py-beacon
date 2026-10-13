---
title: Charts
description: "Results that draw themselves, chart classes for any data, tables, interactive charts, and every setting of the chart design."
---

# Charts

Results draw themselves. Each result object has a `.plot` accessor whose
methods each return a chart, drawn with matplotlib the first time it is
shown or saved. [`beacon.plot.compare`](#comparing-results) puts several
results on one chart, the [chart classes](#chart-classes) draw any DataFrame
or Series in the same frame, [`Table`](#tables) sets a DataFrame as a table,
every chart becomes an [interactive chart](#interactive-charts) with
`.interactive()`, and every part of the design is a [setting](#styles).

`import beacon` does not import matplotlib, and neither does building a
chart. It is loaded the first time a chart is drawn, so code that never
plots never pays for it.

Pictures of every chart, in every theme, beside a live interactive version,
are in the [Gallery](../gallery.md). The [chart styles page](../chart-styles.html)
changes every setting on sample charts and shows the code for each change.

## What each result can draw

| Result | Method | Draws |
| --- | --- | --- |
| `IndexResult` | `level(benchmark=None, label="Index")` | The index level rebased to 100, with an optional benchmark series |
| `IndexResult` | `constituents(limit=8)` | The index drawn over its constituents' closes, all rebased to 100, each constituent named at its line's end; the largest `limit` by weight |
| `IndexResult` | `weights(date=None)` | Constituent weights at a rebalance (the latest by default), with the weight cap marked |
| `BacktestResult` | `performance()` | Growth of 100 with a linked drawdown panel beneath |
| `BacktestResult` | `constituents(limit=8)` | The same chart for the index the run tracked |
| `BacktestResult` | `annual_returns()` | Calendar-year returns as green and red bars, each year from the previous year's close and the first from the initial capital (the same figures as `get_annual_returns()`) |
| `AttributionResult` | `contributions()` | Each constituent's contribution to return, with the cap and cost drags in the notes |
| `OptimisationResult` | `exposures()` | Active weights (optimal minus target), with tracking error and turnover |
| `OptimisationResult` | `frontier(frontier, risk_free_rate=None)` | An efficient frontier, its minimum-variance and tangency points and the capital market line |
| `RiskModel` | `correlation(shading="squares")` | The correlation matrix as a heatmap, each pair a square or, with `shading="blended"`, the colours blended from cell to cell |

Every method, and `compare`, also takes `title=`, `subtitle=` and `notes=`,
described [below](#titles-notes-and-the-mark), and `style=` for that chart
alone ([Styles](#styles)). Typing `result.plot` at a
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

chart = index.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
chart.save("level.png", dpi=110)

backtest.plot.performance().save("performance.png", dpi=110)
```

Every method returns a chart. `.save(path)` writes it as a PNG, SVG or PDF,
`.show()` shows it, and in a notebook a chart that is a cell's last value is
shown beneath the cell. `.figure` and `.ax` are its matplotlib figure and its
top panel's axes, for anything else matplotlib can do. A chart is drawn the
first time one of these is used, in the style in force when it was built.

A chart's size is fixed per chart kind. Every
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

Every chart is framed the way Beacon desktop frames one:

- **A title and a subtitle** at the top left: the title in Inter Tight and the
  subtitle beneath it in Source Serif 4 italic, with a thin accent bar down
  their left, level with the y axis. A title with no subtitle is centred on
  the bar, which is the same height either way.
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
`beacon.plot.frame_text(chart, part)` reads one back from a drawn chart, with
`part` one of `"title"`, `"subtitle"` or `"notes"`:

```python
from beacon.plot import frame_text

chart = backtest.plot.performance(subtitle="The CANON tracker, 10 bps a trade",
                                  notes="Sample data. Costs of 10 bps on every trade.")

print(frame_text(chart, "subtitle"))
print(frame_text(chart, "notes"))
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

## Chart classes

The result charts are built on chart classes, which draw any DataFrame or
Series in the same frame. Each takes `title=`, `subtitle=`, `notes=`,
`source=` and `style=`, and returns a chart like the result methods do: it is
drawn the first time it is shown, saved or asked for its figure, and
`.interactive()` makes it a [plotly figure](#interactive-charts).

| Class | Draws |
| --- | --- |
| [`Line`](#line) | One or more series over dates or numbers, with an optional second axis |
| [`Area`](#area) | Parts of a whole stacked, or one series filled from zero |
| [`Column` and `Bar`](#column-and-bar) | Values by category, upright or across |
| [`Combo`](#combo) | Columns on the left axis and a line on the right |
| [`Scatter`](#scatter) | One measure against another, with a fitted line or a curve |
| [`Histogram`](#histogram) | Counts in equal bins, with a normal curve and marked values |
| [`Heatmap`](#heatmap) | A table of values as colour |
| [`Grid`](#grids) | Several charts under one heading |
| [`Table`](#tables) | A DataFrame as a table, as an image or as HTML |

Every chart has a figure of its own and is always framed. Charts are not
drawn into an axes you laid out; `chart.figure` is an ordinary matplotlib
figure, so anything else can be done with it. Pictures of every class, in
every theme and live, are in the [Gallery](../gallery.md).

The examples below draw from the sample dataset.

```python
import pandas as pd

from beacon.plot import Area, Bar, Column, Combo, Heatmap, Histogram, Line, Scatter

prices = dataset.prices().loc[:END]
rebased = prices / prices.iloc[0] * 100
```

### Line

![Three holdings, each rebased to 100, the first in the accent colour with its last value labelled.](../gallery/line.light.png)
![Three holdings, each rebased to 100, the first in the accent colour with its last value labelled.](../gallery/line.dark.png)

```python
Line(rebased[["AAA", "BBB", "CCC"]],
     title="Three holdings",
     subtitle="Each rebased to 100",
     ylabel="Level",
     reference=100,
     label_last=True,
     periods={"First half": ("2023-03-01", "2023-06-30")},
     source="Synthetic data").save("line.png", dpi=110)
```

The first series is drawn in the accent colour and heavier, over the others.

- `reference=` draws a line the chart is read against, such as 100.
- `label_last=` dots and labels the last value: `True` for the first series,
  or the names of the series to label. `end_names=` writes series' names at
  their lines' ends instead of a legend.
- `secondary=` takes series measured on a second axis on the right, drawn
  lighter, with `ylabel_secondary=` as its title.
- `periods=` marks periods as boxes from the bottom of the plot to the top,
  behind the series: `{label: (start, end)}`. Each is labelled at its top
  unless `period_labels=False`, and `period_legend=` names them once in the
  legend.
- `percent=True` writes shares as percentages with as few decimals as tell
  the ticks apart (5%, 10%, 15%), and `invert=True` runs the axis from high at
  the bottom to low at the top.
- `xlim=`, `ylim=` and `ylim_secondary=` set an axis's (low, high); either may
  be `None`. While axes end on their ticks a limit moves out to the nearest
  tick, never in.
- `colors=` sets each series' colour, by token name or hex.

### Area

![Four holdings' shares of the whole, stacked to 100 percent.](../gallery/area.light.png)
![Four holdings' shares of the whole, stacked to 100 percent.](../gallery/area.dark.png)

```python
held = prices[["AAA", "BBB", "CCC", "DDD"]]
shares = held.div(held.sum(axis=1), axis=0)

Area(shares, title="Shares", subtitle="Each holding's share of the four",
     ylabel="Share", percent=True).save("area.png", dpi=110)

drawdown = prices["AAA"] / prices["AAA"].cummax() - 1
Area(drawdown.rename("Drawdown"), title="Drawdown", percent=True).save("drawdown.png", dpi=110)
```

Several series are stacked one on another; `stacked=False` fills each from
zero. One series, such as a drawdown, is filled from zero with its line on
top. The horizontal axis runs from the first value to the last, so the fill
meets the axes.

### Column and Bar

![Calendar-year returns for two holdings, side by side.](../gallery/column.light.png)
![Calendar-year returns for two holdings, side by side.](../gallery/column.dark.png)

![The largest holdings' weights as bars across, with a dashed cap at 8 percent.](../gallery/bar.light.png)
![The largest holdings' weights as bars across, with a dashed cap at 8 percent.](../gallery/bar.dark.png)

```python
monthly = prices["CCC"].resample("ME").last().pct_change().dropna().loc["2024"]
monthly.index = monthly.index.strftime("%b")

Column(monthly.rename("CCC"), title="Monthly returns", subtitle="CCC in 2024",
       ylabel="Return", percent=True).save("column.png", dpi=110)

weights = pd.Series([0.0792, 0.0741, 0.0688, 0.0617],
                    index=["Northwind Systems", "Halcyon Health", "Arden Industrials",
                           "Meridian Software"], name="Weight")
Bar(weights, title="Largest holdings", subtitle="Capped at 8%", percent=True,
    value_labels=True, decimals=2, cap=0.08).save("bar.png", dpi=110)
```

`Column` draws values upright and `Bar` across; both take a Series or a
DataFrame whose index names the categories. Several series are grouped side
by side in their own colours.

- One series takes the signed colours, green up and red down.
  `signed=False` gives every bar one colour, grey by default or `color=`, and
  `up_color=` and `down_color=` change the signs' colours for one chart.
- `gap=` sets the space between categories and `inner_gap=` between the bars
  in one, each as a share; both can be 0.
- `value_labels=True` writes each value at its bar's end, with `decimals=`
  decimals, and `cap=` marks a limit with a dashed line.
- Categories that do not fit side by side are labelled every few (every
  other, every third: a year's quarters), staying level.

### Combo

![Daily net flows as green and red columns, with assets as a line on the right axis.](../gallery/combo.light.png)
![Daily net flows as green and red columns, with assets as a line on the right axis.](../gallery/combo.dark.png)

```python
flows = prices["AAA"].diff().loc["2024"].rename("Net flows")
assets = prices["BBB"].loc["2024"].rename("Assets")

Combo(flows, assets, gap=0, title="Daily flows and assets",
      ylabel="Net flows", ylabel_secondary="Assets").save("combo.png", dpi=110)
```

Columns on the left axis and a line on the right, for a flow and a stock on
one calendar. The columns are signed by default. On daily data with `gap=0`
each column runs to the next trading day, so the columns meet edge to edge.

### Scatter

![AAA's daily returns against DDD's, with a fitted line and its slope and fit written on the chart.](../gallery/scatter.light.png)
![AAA's daily returns against DDD's, with a fitted line and its slope and fit written on the chart.](../gallery/scatter.dark.png)

```python
returns = prices[["AAA", "DDD"]].pct_change().dropna().loc["2024"]

Scatter(returns, x="DDD", y="AAA", fit="linear", percent=True,
        title="Sensitivity", subtitle="AAA's daily returns against DDD's").save(
    "scatter.png", dpi=110)

bets = pd.DataFrame({"Active weight": [-0.0003, 0.0049, -0.0041, 0.0007],
                     "Excess return": [0.12, 0.07, -0.05, -0.04]},
                    index=["AAA", "DDD", "CCC", "EEE"])
Scatter(bets, x="Active weight", y="Excess return", labels=bets.index.to_series(),
        axes="origin", signed=True, percent=True,
        quadrants=["Over, ahead", "Under, ahead", "Under, behind", "Over, behind"],
        title="Did the bets pay off?").save("bets.png", dpi=110)
```

`x=` and `y=` name two columns, or pass two Series.

- `labels=` names the points, and `name=` says what they are in the legend.
- `fit="linear"` draws a fitted line and writes its slope (beta) and R² on
  the chart. `curve=` draws a curve through the same axes, such as a frontier,
  with `curve_label=` in the legend.
- `axes="origin"` draws the axes through zero, for a chart read as four
  quadrants, with optional `quadrants=` names from the top right
  anticlockwise. `signed=True` colours each point by whether its two values
  share a sign.
- `percent=` is `True` for both axes, or `"x"` or `"y"` for one.

### Histogram

![Daily returns in 40 bins, with the normal curve of the same mean and deviation and a dashed line at the 95 percent VaR.](../gallery/histogram.light.png)
![Daily returns in 40 bins, with the normal curve of the same mean and deviation and a dashed line at the 95 percent VaR.](../gallery/histogram.dark.png)

```python
daily = prices["AAA"].pct_change().dropna().rename("Daily return")

Histogram(daily, bins=40, normal=True, percent=True,
          mark={"95% VaR": float(daily.quantile(0.05))},
          title="Daily returns", subtitle="How often AAA moved by each amount").save(
    "histogram.png", dpi=110)
```

Counts in `bins=` equal bins. `normal=True` draws the normal curve with the
values' mean and standard deviation, and `mark=` marks values with a dashed
line and names them, such as a VaR.

### Heatmap

![Six holdings' correlations as coloured squares, each value written in its cell, with the colour scale beside.](../gallery/heatmap.light.png)
![Six holdings' correlations as coloured squares, each value written in its cell, with the colour scale beside.](../gallery/heatmap.dark.png)

```python
correlation = prices.pct_change().dropna().corr()

Heatmap(correlation, title="Correlation", subtitle="Between daily returns",
        colormap="beacon_corr", vmin=0.2, vmax=1.0).save("heatmap.png", dpi=110)
```

A table of values as colour: the index names the rows, top to bottom, and the
columns the columns. The chart is cropped to what it draws at the same scale,
so a square table does not leave the frame's width empty.

- `colormap=` takes a [colour map's name](#colour-maps) (`_r` on the end
  reverses it) or a list of hex colours from low to high. By default values
  either side of zero diverge about it in `blue-red`, and others take
  `blues`. `vmin=` and `vmax=` set the map's ends.
- `values=` writes each value in its cell, in dark or light text to read on
  the cell, with `decimals=` decimals; `percent=True` writes shares as
  percentages.
- `lower=True` draws the lower triangle of a symmetric table.
- `shading="blended"` blends each colour into the next; `gap=` sets the space
  between cells, and `gap=0` leaves no seam.
- `colorbar=` draws the scale beside, labelled at its two ends.

## Grids

![Four panels under one heading: volatility, drawdown, beta and tracking error, each titled.](../gallery/grid.light.png)
![Four panels under one heading: volatility, drawdown, beta and tracking error, each titled.](../gallery/grid.dark.png)

`Grid` puts several charts in rows and columns under one heading and one
notes line. Each chart draws one panel, titled with its own title.

```python
from beacon.plot import Grid

growth = Line(rebased[["AAA", "BBB"]], title="Growth of 100", reference=100)
ranks = Bar(prices.loc["2024"].iloc[[0, -1]].pct_change().iloc[-1].rename("2024"),
            title="Returns in 2024", percent=True, value_labels=True)
falls = Area(drawdown.rename("Drawdown"), title="AAA's drawdown", percent=True)

Grid([[growth, ranks],
      [falls, ranks]],
     heights=(3, 1),
     title="Two holdings",
     subtitle="A tall chart beside two").save("grid.png", dpi=110)
```

- Rows run top to bottom, each left to right. `None` leaves a cell empty, and
  one chart named in a rectangle of cells spans them, as `ranks` does here.
- `heights=` and `widths=` set the rows' and columns' shares, such as
  `heights=(3, 1)` for a tall panel over a short one.
- Each column is given the room its labels take and each row room for its
  titles and tick labels, and the heading and legend line up with the outer
  panels.
- `sharex=True` runs every panel on dates over the same dates. A panel with
  another on dates directly beneath it leaves its dates to that one.
- One legend is shown, in the heading band, when every panel's legend names
  the same series, and none otherwise.

`BacktestResult.plot.performance()` is such a grid: its level over its
drawdown, three to one.

## Tables

![A table of returns and risk under the Beacon heading, with column groups and green and red returns.](../gallery/table.light.png)
![A table of returns and risk under the Beacon heading, with column groups and green and red returns.](../gallery/table.dark.png)

`Table` draws a DataFrame under the same frame as a chart: the mark, the
heading, the table and the notes line. `.save()` and `.show()` draw it as an
image, for reports; `.to_html()` writes it as HTML, which is also what a
notebook shows.

```python
from beacon.plot import Table

performance = pd.DataFrame(
    {"1Y": [8.42, 7.91, 6.10, -2.35], "3Y": [5.12, 4.80, 3.95, 0.40],
     "Volatility": [14.2, 13.8, 15.1, 18.4], "Sharpe": [0.61, 0.58, 0.44, -0.12]},
    index=["Beacon Global", "MSCI World", "Peer median", "Peer low"])

table = Table(performance,
              title="Performance and risk",
              subtitle="To 31 December 2024",
              groups={"Return (%)": ["1Y", "3Y"], "Risk": ["Volatility", "Sharpe"]},
              formats={"Return (%)": "{:.1f}", "Risk": "{:.2f}"},
              signed=["Return (%)"],
              highlight=["Beacon Global"],
              row_groups={"Ours": ["Beacon Global", "MSCI World"],
                          "Peers": ["Peer median", "Peer low"]},
              row_group_headings=True,
              source="Synthetic data")
table.save("table.png", dpi=110)

with open("table.html", "w", encoding="utf-8") as page:
    page.write(table.to_html())
```

- **Formats.** `formats=` takes a format per column or per column group: a
  format string such as `"{:,.1f}"`, a mapping of `decimals`, `prefix`,
  `suffix`, `thousands` and `percent`, or a function. Negatives take a true
  minus sign, or brackets with `negatives="brackets"`; a value that rounds to
  zero has no sign, and a missing value is a dash.
- **Signs and emphasis.** `signed=` colours columns or groups green and red by
  sign, and `highlight=` sets rows in bold.
- **Column groups.** `groups=` puts columns under headings, as
  `{heading: [columns]}` or `{heading: how many columns}` in order. Each
  heading is centred over its columns on a rule that starts a little in from
  the group's left, so neighbouring groups' rules stay apart.
- **Row groups.** `row_groups=` keeps rows together, and sorting the HTML
  keeps each group together. `row_group_headings=True` names each group in a
  column at the left, centred beside its rows, with a rule between the
  headings that stops short of the rows' rule; the row names then take the
  column headings' colour, so the group headings stand out.
- **Heatmap.** `heatmap=` colours a column, `{"column": name}`, or a column
  group, `{"group": heading}`, by value. It also takes `colormap`,
  `reverse`, `scale` (`"column"` or `"group"`), `center` (such as 0), `fade`
  (near-neutral cells take the table's background, so no pale cell stands
  out on a dark page), `values` (`False` hides the numbers), and `shading`
  (`"squares"` or `"blended"`) with `blend` (`"columns"`, `"rows"` or
  `"both"`). Text on a heatmap cell is dark or light to read on it.

The look is the style's `table` group, so a saved style carries it.
`preset=` chooses the rules and boxes:

| Preset | Draws |
| --- | --- |
| `lines` (the default) | A rule under the header, light rules between rows, a rule under the table, and rules under the column group headings |
| `plain` | No rules or boxes at all |
| `boxed` | A box around the values, with the row names to its left and the headings above, and the group headings' rules |

Each rule and box can still be changed on its own, after the preset:
`table_header_rule`, `table_row_rules`, `table_bottom_rule`,
`table_group_rules`, `table_column_rules`, `table_box`, `table_group_boxes`
(a box around each column group, with a narrow gap so neighbouring boxes do
not merge) and `table_row_boxes` (a box around each row group).

```python
boxed = Table(performance, title="Boxed by group", preset="boxed",
              groups={"Return (%)": ["1Y", "3Y"], "Risk": ["Volatility", "Sharpe"]},
              style={"table_box": False, "table_group_boxes": True, "table_stripes": True})
boxed.save("table-boxed.png", dpi=110)
```

The row names' column is as wide as its names need, with `table_name_space`
before the longest, and the value columns share the rest. In HTML a column's
values sort when its heading is clicked. The HTML is laid out as the image
is: its columns take the image's widths, its rules take no room, and it
carries the fonts it is set in, so it looks the same offline and wherever it
is opened. That makes each table's HTML about 1.4MB.

A [report](reports.md) holds a chart or a table as an image block:
`beacon.report.blocks.Chart.of(table, "table.png")` saves it and returns the
block.

## Interactive charts

Every chart's `.interactive()` returns a plotly figure drawn from the same
layout as its image, so the two look the same.

```python
figure = Line(rebased[["AAA", "BBB", "CCC"]], title="Three holdings",
              subtitle="Each rebased to 100", reference=100).interactive()

figure.write_html("three-holdings.html")
figure.write_html("three-holdings-light.html", include_plotlyjs="directory")
```

- **The same chart.** The figure is the image's size; each axis takes the
  image's domain, range, ticks and labels; and the mark, heading, legend and
  notes are where the image puts them. Zooming and the toolbar are off: the
  figure is the chart as designed.
- **The popup.** Moving the pointer over a chart shows a popup in the style
  of Beacon desktop's popovers, a blurred glass fill with a hairline edge, in
  the chart's theme, with the app's crosshair: a dashed line down through
  the date under the pointer with a dot on each line, or both ways through a
  scatter's point. It says what the pointer is on, with the units the axis
  writes (see the table below).
- **Offline.** `.write_html()` and `.to_html()` carry the chart's fonts,
  plotly.js and the popup's script, so the file looks and works the same
  offline. Each file is about 5MB; `include_plotlyjs="directory"` writes
  plotly.js once beside several files instead.
- **Notebooks.** A figure that is a cell's last value is shown by plotly.
  Where the notebook does not run the popup's script, plotly's own labels
  show instead.

| Chart | The popup shows |
| --- | --- |
| Time series | The date and every series' value |
| Columns and bars | The category and each series |
| Scatter | The point's name and both values |
| Heatmap | The cell and its value |
| Histogram | The bin's range and its count |

plotly is loaded only when `.interactive()` is called. A table's interactive
form is its HTML, so `Table.interactive()` points to `to_html()`.

## Attribution, optimisation and risk

These results come from [Attribution](attribution.md),
[Optimiser](optimiser.md) and [Risk model](risk-model.md).

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
    draw().save(filename, dpi=110)
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

`beacon.plot.compare(*results, labels=None)` draws two or more
`IndexResult` or `BacktestResult` objects on one axis, and takes `title=`,
`subtitle=` and `notes=` like the chart methods:

```python
from beacon.plot import compare

compare(index, backtest, labels=["Index", "Fund"]).save("compare.png", dpi=110)
```

Every series is cut to the dates they all share and rebased to 100 on the
first shared date, so the lines start together and differences in history
length do not distort the comparison. The notes say how many shared
observations the comparison rests on.

Labels default to each result's index id or portfolio id. `compare` raises
`ValueError` for fewer than two results or when they share no dates.

## Themes

`beacon.plot.use(theme)` applies a theme to every chart built afterwards,
changing nothing else in the [style](#styles):

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
index.plot.level().save("level-for-a-light-page.png", dpi=110)

beacon.plot.use("light")
```

Every theme draws in Beacon's accent, series and sign colours, which come
from the same design tokens as Beacon desktop, so a chart matches the screen
it sits on; only the background and the text change. A chart records the
theme it was built in, so its own colours (the series accent, green and red
for signs, the cap marker, the frame) follow that theme even on a transparent
background.

Once a chart has been drawn, or `use()` has run, each theme is
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
`border`, `divider`, `period`, `text-primary`, `text-secondary`,
`text-muted`, `accent`, `success`, `danger`, `series-2` and `series-3`. The
series colours are the [series palette](#the-series-palette).

## Styles

Every part of a chart's design is a setting, with Beacon's design as the
default: its size, the heading, the text, the line weights, the axes, the
legend, the notes and the colours. Change any of them for every chart, for a
block of code, or for one chart.

- **`Style`** is a look: one value holding every setting. `Style.beacon()` is
  the default and `Style.plain()` drops the beta mark and draws on white, for
  work that is not a Beacon report. A style saves to a TOML file and loads
  back, so a project can keep its look in one place.
- **`configure()`** makes a look the default for every chart drawn afterwards.
  It takes a `Style`, a file a style was saved to, `preset="plain"`, or
  settings by name. Settings by name change the current style rather than
  replacing it, and `reset()` goes back to Beacon's design.
- **`settings()`** does the same for one `with` block, then puts the previous
  look back.
- **`style=`** on any chart changes that chart alone: a `Style`, or settings
  by name.

The most specific wins: a chart's `style=`, then a `settings()` block, then
`configure()`, then the defaults. `use(theme)` is a shortcut for
`configure(theme=...)` and changes nothing else.

```python
from beacon.plot import Style, configure, reset, settings

configure(mark=False, heading_scale=1.2)
index.plot.level().save("level-no-mark.png", dpi=110)

with settings(theme="white", legend="bottom"):
    index.plot.level().save("level-white.png", dpi=110)

index.plot.level(style={"scale": 1.25, "series_width": 2.0}).save("level-larger.png", dpi=110)

house = Style.plain().replace(lines={"series_width": 2.0}, axes={"grid": "y"})
house.save("house.toml")
configure(Style.load("house.toml"))
reset()
```

The settings, by the short names `configure()` and `style=` take, with the
group each lives in for `Style.replace()`:

| Setting | Group | Default | What it does |
| --- | --- | --- | --- |
| `theme` | | `beacon-light` | The theme (see [Themes](#themes)). |
| `scale` | size | 1.0 | Zooms the chart: figure, text and lines together. The heading keeps its own scale. |
| `width` | size | the chart's own (6in) | The designed width: more room at the same text size. |
| `aspect` | size | the chart's own | Width to height, such as `"16:9"`. |
| `dpi` | size | 300 | For saved images. |
| `mark`, `accent_bar` | heading | on | The beta mark and the bar beside the title. |
| `heading_scale` | heading | 1.0 | Scales the mark, accent bar, title and subtitle together. |
| `title_font`, `title_size` | heading | Inter Tight, 9pt | |
| `subtitle_font`, `subtitle_size`, `subtitle_italic` | heading | Source Serif 4, 6.3pt, italic | |
| `font` | text | Inter Tight | The chart's own text. |
| `text_scale` | text | 1.0 | The chart's small text alone. |
| `label_size` | text | 5.5pt | Legend entries and labels on the chart. |
| `series_width`, `secondary_width` | lines | 1.5pt, 1.1pt | The leading series, and the others. |
| `axis_width`, `reference_width`, `grid_width` | lines | 0.6pt, 0.5pt, 0.6pt | |
| `x_axis_line`, `y_axis_line` | axes | on | The axis lines. |
| `x_ticks`, `y_ticks` | axes | on | The tick marks. |
| `tick_direction`, `tick_length` | axes | in, 3.5pt | |
| `tick_label_size`, `axis_title_size` | axes | 7.8pt, 8pt | |
| `grid` | axes | each chart's own | `"none"`, `"x"`, `"y"` or `"both"`. |
| `end_on_ticks` | axes | on | Each measuring axis ends on its outermost ticks. |
| `blank_lowest_label` | axes | on | The lowest y label is left blank where both axes measure. |
| `reference_lines` | axes | on | The lines a chart is read against, such as 100 or zero. |
| `legend` | legend | `"heading"` | `"heading"`, `"bottom"`, `"right"` or `"none"`. |
| `notes`, `notes_size`, `notes_italic` | notes | on, 4.5pt, italic | The notes line. |
| `series_colors` | colors | the palette | Hex codes replacing the series colours in order. |
| `up_color`, `down_color` | colors | the theme's | Gains and losses. |
| `accent_bar_color`, `period_color`, `period_label_color` | colors | the theme's | |
| `text_color`, `muted_color`, `background` | colors | the theme's | |
| `table_preset` | table | `lines` | A table's rules and boxes: `lines`, `plain` or `boxed` (see [Tables](#tables)). |
| `table_header_rule`, `table_row_rules`, `table_bottom_rule`, `table_group_rules`, `table_column_rules` | table | the preset's | Each rule on its own. |
| `table_box`, `table_group_boxes`, `table_row_boxes`, `table_group_box_scope` | table | the preset's, `values` | Boxes round the values, each column group or each row group; a column group's box encloses the `values` or the `headings` too. |
| `table_rule_width`, `table_row_rule_width`, `table_box_width` | table | 0.48pt, 0.48pt, 0.36pt | |
| `table_row_padding`, `table_cell_padding`, `table_name_space` | table | 2pt, 3pt, 3pt | Above and below each row, either side of each cell, and before the longest row name. |
| `table_body_size`, `table_header_size`, `table_group_size` | table | 6.3pt, 5.5pt, 5.5pt | The values, the column headings and the group headings. |
| `table_header_weight`, `table_name_weight` | table | 300, 400 | |
| `table_number_align`, `table_header_align`, `table_name_align` | table | center, center, right | `table_header_align="numbers"` aligns the headings as the numbers. |
| `table_stripes` | table | off | Every other row shaded. |
| `table_rule_color`, `table_row_rule_color`, `table_stripe_color`, `table_header_color` | table | the column headings', the theme's divider, the divider, muted text | |

Sizes are in points as designed. A font that is not bundled is given as the
path of its file (`.ttf` or `.otf`), so its text can be measured.

### The series palette

Series take the palette in order: blue, deep blue, magenta, pink, lime, green,
olive, red and oxblood, then black (white on a dark canvas) and five greys for
context lines such as peers. `beacon.plot.themes.SERIES` holds both modes'
lists, and `series_colors` replaces them. Green and red for gains and losses
are separate (`up_color` and `down_color`), so a series never reads as a
sign.

### Colour maps

Heatmaps and a table's heatmap take a colour map by name, `_r` on the end to
reverse it, or a list of hex colours from low to high:

| Name | Runs |
| --- | --- |
| `beacon_corr` | Green through amber to red, the same in every theme: correlations |
| `blue-red` | Blue through a light middle to red: values either side of zero |
| `blues`, `greens`, `purples`, `greys` | Light to deep in one colour: values of one sign |
| `viridis` | Dark blue through green to yellow |

`beacon.plot.colormaps.names()` lists them.

### Trying every setting

The [chart styles page](../chart-styles.html) draws sample charts and a table
and changes them as each setting changes, showing the code for the look on
screen. It is a sketch in the browser of what the library draws, not the
library itself, so check a look in Python before relying on it.

The full API is in the [Charts reference](../reference/plot.md).
