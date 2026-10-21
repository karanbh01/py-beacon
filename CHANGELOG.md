# Changelog

What changed in each release of py-beacon. The newest release is first.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the version numbers follow [Semantic Versioning](https://semver.org/).
Before 1.0, a breaking change raises the middle number, as in 0.1 to 0.2.

## [Unreleased]

## [0.9.4] - 2026-10-21

A store saves in well under half the time with its progress shown, and the API refuses requests it used to misread: a parameter given twice, a boolean for a number, a price column it does not hold.

### Fixed

- `GET /data/prices/{identifier}` reads `columns` as the other list parameters do, so `?columns=CLOSE,VOLUME` names two columns, and an unknown column is refused with 422 `INVALID_ARGUMENT` rather than failing with a 500.
- `POST /derivatives/trs/price` and `/derivatives/futures/price` refuse inputs whose result overflows to infinity with 422 `INVALID_ARGUMENT`, where they answered 200 with `null` in place of a number.

### Changed

- Saving a data store takes well under half as long: on 1.16M market rows, 11 s rather than about 24 s, for a file 2% larger. A synthetic store at the API's default size should save in about 110 s rather than four minutes (estimated from the measured tenth). Each file is written a chunk of rows at a time and compressed as it goes, at gzip level 6 rather than 9. `store.save` takes a `progress` callback, and generating synthetic data reports the save's progress by dataset and chunk ("Saving market data") instead of sitting at one stage, with each stage's share of the progress bar matching the time it takes.
- A query parameter that takes one value is refused with 422 `INVALID_ARGUMENT` when it is given more than once. The last value used to win, so `?limit=10&limit=500` answered with 500 rows.
- A decimal number in a request body must be a JSON number: `true`, `false` and numbers sent as strings are refused with 422 where they were read as numbers, in a strategy, a backtest request, an index, a risk model or optimisation request, feature rows and synthetic data. A risk model's `index_id` and `strategy_id` must be ids like any other.

## [0.9.3] - 2026-10-20

A strategy's holdings at any date, from the universe to the book, with each step explained, and a backtest editor that can offer pickers and rows for every structured setting.

### Added

- `GET /strategies/{id}/holdings?date=` shows a strategy's whole holdings workflow at a date, one row per name: the universe narrowed by the index's rules, the index's cap, the benchmark's weight, the strategy's choice, the weight after each implementation screen, the cash buffer and the caps, the weight traded to, and as of the date the held weight, its drift and the units held, with every rebalance date of the run and each stage's label and count. A backtest now keeps how each rebalance's universe narrowed (`IndexResult.selections`) and each name's weight after the screens (`RebalanceStep.screened`).
- `universe_weight` on holdings rows and constituent preview assets: each universe name's share of the universe's total market cap, free-float where the index weights by free-float market cap, full otherwise.
- Each holdings stage and constituent preview step says what it is and does: its `type`, a plain-sentence `description` of what it does with its settings, and its settings as labelled `params` formatted with their units. The market cap and liquidity rules' thresholds now carry their units (money, shares) in `/indices/rule-types`.

- Every structured setting in `GET /backtest/options` names the shape it takes in `ref`, a schema published under `components.schemas`: names, a number per name, amounts by date, a field, an expression, optimiser constraint rows, signals by factor and targets by factor, and a vehicle's pricing, limits and withholding rates. A list of reference values names its `column`, and a setting whose entries are configurable types names the family they come from in `options`. `replication_constraints` lists the constraints an optimised replication takes, and `GET /data/reference-values?column=` lists the values a reference column holds. The optimiser's constraints in `/optimise/constraint-types` name their shapes too.

### Fixed

- A constituent preview's stale-price step no longer takes another rule's id.

### Changed

- `/data/fields` labels keep acronyms in capitals: "GICS industry", "ISIN", "IPO date", rather than "Gics industry".

## [0.9.2] - 2026-10-19

Every figure a run shows comes from the engine: a run compared with its index and its benchmark, its figures by year, quarter and month, and each holding's reference columns as they stood on the row's date.

### Added

- A run's `comparisons` compare it with its index and its benchmark: each comparator's level and own figures over the days both have (total and annualised return, volatility, Sharpe ratio, maximum drawdown and monthly hit rate) and the run's figures against it (relative and annualised relative return, tracking error, information ratio, correlation, beta and the share of months it beat it), for every kind of strategy. The run's metrics add `hit_rate_monthly`. `beacon.analysis.comparison` computes them in Python.
- `GET /strategies/{id}/periods?frequency=annual|quarterly|monthly` gives a run's figures in each calendar year, quarter or month, and against its index and its benchmark, each period from the previous period's close so they compound to the whole run.
- Reference columns on every row that names a holding, as they stood on the row's date: the weights view, each rebalance of a run with its unfilled orders and price gaps, attribution, an optimisation run's rows and frontier, and a constituent preview. `reference_fields` chooses the columns; without it a row carries the name, asset class, sector and country. A missing `ASSET_CLASS` answers `Equity`, as the engine reads it. `GET /optimise/runs/{id}` serves a run's rows, and `GET /data/fields` gives each field a `label`. `beacon.data.reference_as_of.ReferenceAsOf` answers many names on many dates from one read.

## [0.9.1] - 2026-10-18

For a backtest form and its runs: a strategy's latest run result by strategy, every setting a backtest request takes described in the options, and a run's settings complete enough to start a new backtest from.

### Added

- `GET /strategies/{id}/result` gives a strategy's latest run payload, every series and record in it, without the job's id, and answers 404 until the strategy has a run.
- `GET /backtest/options` describes the run's own settings in `run_settings`: `start` and `end` (a new `date` field type), `initial_capital`, `transaction_cost_bps`, `currency`, `dividends` and `benchmark`, with their defaults, units and choices. A request always took them, but the options never said so, so a form built from them could not set the dates, capital or cost.
- A run's resolved `settings` carry its `start` and `end` (the dates it ran over, defaults filled in), `initial_capital`, `transaction_cost_bps` and `benchmark`, so a new backtest can start from a run's settings.

## [0.9.0] - 2026-10-16

Strategies you save and backtest: a stored strategy holds what a backtest holds and how it trades, and every backtest runs one, keeping its runs apart from any other strategy's on the same index. An index can be calculated on its own, with no backtest. Share classes, synthetic replication and futures reach the API, and a hedged share class is measured against the index hedged the same way. **Breaking** for API clients: backtests and their results moved from `/beacon/{index_id}/...` to `/strategies/{id}/...`, and runs saved before this release are no longer served.

### Added

- `GET /indices/{id}/level` calculates a stored index on its own, with no backtest: its level, its total and annualised return, volatility, Sharpe ratio and maximum drawdown, and its weights at each rebalance, over `start` and `end` or the whole of the loaded data. `Backtest.calculate(definition, start, end)` does the same in Python, through the cache a run uses.
- Stored strategies: `/strategies` keeps what a backtest holds and how it trades, beside the index definitions, so a strategy can be saved, listed, read, checked and deleted, and travels with the indices. A strategy is `{id, name, description, index_id, strategy, implementation}`, the last two in the shapes a backtest request gives them. A save is checked as a backtest checks its settings, with a finding for each problem at its path, and `POST /strategies/validate` checks one without saving it. A strategy whose index has gone lists with a `problem`, and deleting an index a strategy holds is refused with 409 `IN_USE`, naming the strategies in `detail.used_by`.
- An optimisation run can aim at a stored strategy rather than an index: `POST /optimise/runs` takes `strategy_id` to solve for the weights the strategy's latest run chose at a rebalance, so an active strategy's answers how close a portfolio can get to it under the constraints. A risk model can take its names from a `strategy_id` too.
- Share classes on the API: a backtest request takes `share_classes`, each with its currency, hedge, fee and distribution policy, and `initial_capital` and `flows` by class name. `GET /backtest/options` describes the class fields, and a problem is a finding at its path, such as `share_classes.1.hedge_band`. The result carries each class's record in its own currency on `level`'s dates: net assets, units, NAV per unit, fees, distributions, flows and, for a hedged class, its hedge's value and trades, with the fund's `hedge_value`. Each class's total return, tracking difference and tracking error are measured against the index in its currency.
- A share class's benchmark: `index_in_class_currency` in `beacon.backtest.class_benchmark` gives the index converted into a class's currency, or hedged into it the way the class hedges, at the first session of each month and at covered interest parity, so a hedged class's tracking difference leaves out the interest rate differential. `class_tracking` measures a class against it.
- A synthetic fund's swap and a fund's index futures in the API's backtest result: `swap_value`, `counterparty_exposure` and `swap_resets`, and `futures_notional`, `futures_exposure` and `futures_rolls`, on `level`'s dates, and null or empty for a fund without them. Each futures roll now records the `contracts` held and their `price`, in `FuturesRoll` and on the API.

### Changed

- **Breaking:** every backtest runs a stored strategy. `POST /strategies/{id}/backtest` and `POST /strategies/{id}/backtest/validate` take what the strategy does not hold (the period, capital, costs, currency, dividends, benchmark, modelling assumptions, flows and vehicle), and refuse a body carrying `strategy` or `implementation`. `POST /beacon/{index_id}/backtest` and its validate route are gone.
- **Breaking:** runs are kept per strategy, so two strategies on one index each keep their own. The views of a run moved from `/beacon/{index_id}/...` to `/strategies/{id}/overview`, `/record`, `/weights`, `/attribution` and `/assets/{identifier}`, each naming its `strategy_id` and the `index_id` it holds; `/beacon/compare` is `/strategies/compare`, comparing strategies (`strategy_ids`), and `/beacon/backtests` is `/strategies/runs`, whose rows give `strategy_id`, `strategy_name` and `index_id`. A run's `settings` and its job kind, `backtest:{strategy_id}`, name the strategy. Runs saved before this release, filed under an index, are no longer served: backtest each strategy again.
- **Breaking:** an optimisation run takes exactly one of `index_id` and `strategy_id`. An index's weights are now calculated, so optimising against an index no longer needs a backtest of it. Each row's `index_weight` is `target_weight`, and the result adds `strategy_id` and `target_date`. A risk model's `index_id` takes the names from the index's calculation rather than a run.
- **Breaking:** a built-in report template is generated from a `strategy_id`'s latest run, not an `index_id`'s, and named after the strategy; the render result gives `strategy_id`.
- Deleting a strategy removes its runs, and its response says how many went. Deleting an index no longer touches runs, since none belong to it, so `DELETE /indices/{id}` no longer reports `backtest_record_deleted` or `backtest_results_deleted`.
- `compare` and `runs` are reserved and cannot be used as a document's id.

## [0.8.1] - 2026-10-14

Documentation only: the library is unchanged from 0.8.0.

### Changed

- Each chart in the gallery has a switch between its image and its live interactive version, and a list of the six themes. A chart starts as its image in the theme matching the page's light or dark setting. The live versions are now drawn in every theme, not only the two Beacon themes, and the row of the other four themes under each chart is gone.

## [0.8.0] - 2026-10-13

Charts for any data: chart classes (`Line`, `Area`, `Column`, `Bar`, `Combo`, `Scatter`, `Histogram`, `Heatmap`, `Grid` and `Table`), every part of their design a setting in a `Style`, and an interactive version of every chart that looks the same as its image. A plain install now draws charts, writes PDF and Excel reports, solves optimisations and reads Postgres stores; only the server and Yahoo Finance downloads are extras. The result charts return a chart rather than a matplotlib `Axes`, and `ax=` is gone.

### Added

- Chart styles: every part of the chart design is a setting, with Beacon's design as the default. `beacon.plot.Style` holds them (size and zoom, the heading and its scale, the beta mark and accent bar, fonts, text sizes, line weights, axis lines and ticks, gridlines, the legend's position, the notes and every colour), and saves to and loads from a TOML file. `configure()` sets a style for every chart drawn afterwards, `settings()` for a `with` block, and every chart method takes `style=` for that chart alone. `Style.plain()` drops the beta mark and draws on white. `use(theme)` is now a shortcut for `configure(theme=...)`.
- Chart classes in `beacon.plot`, each taking a DataFrame or Series and drawing it in the Beacon frame, with `title=`, `subtitle=`, `notes=`, `source=` and `style=`. A chart is drawn only when it is used: `.show()`, `.save(path)` (PNG, SVG or PDF at its designed size), its `.figure` or `.ax`, or as a notebook cell's last value. Building one needs no matplotlib.
  - `Line`: one or more series, the first in the accent colour and heavier. `reference=` for a line such as 100, `label_last=` to dot and label the last value, `secondary=` with `ylabel_secondary=` for series on a second axis on the right, and `periods=` for marked periods (boxes from the bottom of the plot to the top over date ranges, behind the series, labelled at the top unless `period_labels=False`; `period_legend=` names them once in the legend).
  - `Area`: stacked parts of a whole, or one filled series such as a drawdown. Its horizontal axis runs from the first value to the last, so the fill meets the axes.
  - `Column` (vertical) and `Bar` (horizontal): values by category, grouped where there are several series. A single series takes the signed colours (`signed=False` gives every bar one colour, grey by default, or `color=`; `up_color=` and `down_color=` change the signs' colours for one chart). `gap=` sets the space between categories and `inner_gap=` between the bars in one; both can be 0. `value_labels=` writes each value at its bar's end, and `cap=` marks a limit.
  - `Combo`: columns on the left axis and a line on the right. On daily data with `gap=0`, each column runs to the next trading day, so the columns meet edge to edge.
  - `Scatter`: one measure against another, by column name or as series. `labels=` names the points, `fit="linear"` draws a fitted line and writes its slope and R² on the chart, `curve=` draws a curve through the same axes such as a frontier, and `axes="origin"` draws the axes through zero with optional `quadrants=` names; `signed=True` colours each point by whether its two values share a sign.
  - `Histogram`: counts in equal bins, with an optional `normal=` curve and `mark=` lines for values such as a VaR.
  - `Heatmap`: a table of values as colour, cropped to what it draws at the same scale, so a square table does not leave the frame's width empty. `colormap=` takes a map's name (`beacon_corr`, `blue-red`, `blues`, `greens`, `purples`, `greys`, `viridis`, with `_r` to reverse) or a list of hex colours; by default values either side of zero diverge about it in `blue-red`, and others take `blues`. `vmin=` and `vmax=` set the map's ends, `values=` writes each value in its cell (in dark or light text to read on the cell), `decimals=`, `lower=True` draws the lower triangle, `shading="blended"` blends each colour into the next, `gap=` sets the space between cells (with `gap=0` they overlap a little, so no seam shows), and `colorbar=` draws the scale beside, labelled at its two ends.
  - `Grid`: several charts in rows and columns under one heading and one notes line, each panel titled with its chart's title (`None` leaves a cell empty). A chart named in a rectangle of cells spans them, so a tall chart can run beside two short ones. `heights=` and `widths=` set the rows' and columns' shares, such as `heights=(3, 1)` for a tall panel over a short one. Each column is given the room its labels take and each row room for its titles and tick labels, and the heading and legend line up with the outer panels. `sharex=True` runs every panel on dates over the same dates, and a panel with another on dates directly beneath it leaves its dates to that one. One legend is shown, in the heading band, when every panel's legend names the same series, and none otherwise. `BacktestResult.plot.performance()` is now such a grid: its level over its drawdown, three to one.
  - On a narrow axis, dates are labelled less often so they do not run together. A column chart's categories that do not fit side by side are labelled every few (every other, every third: a year's quarters), staying level, and a heatmap's names, where every one matters, are slanted or stood upright.
  - `percent=True` writes shares with as few decimals as tell the ticks apart: 5%, 10%, 15%.
  - Limits: `xlim=`, `ylim=` and `ylim_secondary=`. While axes end on their ticks a limit moves out to the nearest tick, never in. Limits given the wrong way round are swapped, and a single limit beyond the data keeps the data's span on the other side.
- `Table` in `beacon.plot`: a DataFrame under the Beacon frame (the mark, the heading and the notes line), drawn as an image with `.show()` and `.save()`, or written as HTML with `.to_html()`, which is also what a notebook shows. Its look is the style's `table` group, so a saved `Style` carries it.
  - `preset=` sets the rules and boxes: `"lines"` (the default: rules under the header, between rows, under the table and under the column group headings), `"plain"` (none at all) or `"boxed"` (a box around the values, and the group headings' rules). Each rule and box can still be changed: `table_header_rule`, `table_row_rules`, `table_bottom_rule`, `table_group_rules`, `table_column_rules`, `table_box`, `table_group_boxes` (a box around each column group, with a narrow gap so neighbouring boxes do not merge, enclosing the values or the headings too) and `table_row_boxes` (a box around each row group). Each kind of line has its own weight and colour, and the rules follow the column headings' colour unless given their own. `table_stripes` shades every other row. The row names' column is as wide as they need, `table_name_space` before the longest, and the value columns share the rest.
  - `formats=` per column or per column group: a format string, a mapping of decimals, prefix, suffix, thousands and percent, or a function. Negatives take a true minus sign, or brackets with `negatives="brackets"`; a value that rounds to zero has no sign. A missing value is a dash.
  - `groups=` puts columns under headings, each centred over its columns on a rule that starts a little in from the group's left, so neighbouring groups' rules stay apart. `row_groups=` keeps rows together, and `row_group_headings=True` names each group in a column at the left, centred beside its rows, with a group rule between the headings that stops short of the rule between the rows; the row names then take the column headings' colour, so the group headings stand out. `signed=` colours columns or groups by sign, and `highlight=` sets rows in bold.
  - `heatmap=` colours a column or a column group by value: `colormap=`, `reverse=`, `scale=` ("column" or "group"), `center=` (such as 0), `fade=` (near-neutral cells take the table's background, so no pale cell stands out on a dark page), `values=False` to hide the numbers, and `shading="blended"` with `blend=` "columns", "rows" or "both". Text on a heatmap cell is dark or light to read on it.
  - In HTML, a column's values sort when its heading is clicked, each row group staying together. Without scripts the table still reads, its heatmap cells each their own colour. The HTML is laid out as the image is: its columns take the image's widths, its rules take no room, and it carries the fonts it is set in (the bundled ones, or a font given as a file), so it looks the same offline and wherever it is opened. That makes each table's HTML about 1.4MB.
- Interactive charts: every chart's `.interactive()` returns a plotly figure drawn from the same layout as its image, so the two look the same: the figure is the image's size, each axis takes the image's domain, range, ticks and labels, and the mark, heading, legend and notes are where the image puts them. Moving the pointer over a chart shows a popup in the style of Beacon desktop's popovers (a blurred glass fill, a hairline edge and a soft shadow, in the chart's theme) and the app's crosshair: a dashed line down through the date under the pointer with a dot on each line, or both ways through a scatter's point. The popup gives the date and every series' value on a time series, the category and each series on columns and bars, the point's name and both values on a scatter, the cell and its value on a heatmap, and the bin's range and count on a histogram, with the units the axis writes. The figure's `.write_html()` and `.to_html()` carry the chart's fonts, plotly.js and the popup's script, so the file looks and works the same offline; `include_plotlyjs="directory"` writes plotly.js once beside several files, `fonts=` links a stylesheet of the fonts rather than embedding them (`beacon.plot.webfonts.font_faces` and `font_files` write one), and `fit=True` makes the figure fill the frame it is shown in, such as an iframe the page sizes. Where the script cannot run, such as in some notebooks, plotly's own labels show instead. Zooming and the toolbar are off: the figure is the chart as designed.
- The documentation covers the chart API: the Charts page has a section for each chart class with its options and a picture, tables and their presets, grids, interactive charts, and every setting with its default, the series palette and the colour maps. A chart styles page changes every setting on sample charts and shows the code for it, and the gallery shows every chart in every theme beside a live interactive version.
- `beacon.report.blocks.Chart.of(chart, path)` saves a chart, grid or table as an image and returns a report block holding it, so a report can hold a table under the same frame as its charts.
- `GET /data/stores` says why a store cannot be read when it can tell without loading it: a store saved before 0.7.0, with exchange rates in a `RATE` column, is `readable: false` with `problem` `{code: "STORE_FORMAT_OUTDATED", message}`. It was listed as readable and failed only when loaded, with the generic `CONFIGURATION_ERROR`. Activating it is refused at once with a 409 and the same code.

### Changed

- A plain `pip install py-beacon-kit` now does everything but serve the API and download market data: charts and interactive charts, PDF and Excel reports, the optimiser and Postgres stores work without naming an extra, so the core install brings matplotlib, plotly, fonttools, reportlab, openpyxl, scipy, psycopg and platformdirs. Two extras remain: `server` for the local API server, which no longer pulls in the others, and `data` for Yahoo Finance downloads. The `plot`, `plot-interactive`, `pdf`, `excel`, `optimise` and `postgres` extras are still declared and add nothing, so an install command written for an earlier release still works. `import beacon` still loads none of the new packages; each is imported when the feature that needs it is first used.
- **Breaking:** the result charts (`IndexResult.plot`, `BacktestResult.plot`, `AttributionResult.plot`, `OptimisationResult.plot`, `RiskModel.plot`) and `compare()` return a chart rather than a matplotlib `Axes`, built on the chart classes, so they take the same options: `.save()`, `.show()`, `style=`. `chart.ax` and `chart.figure` are the matplotlib axes and figure for code that drew on the returned axes. A chart is drawn only when it is used, so calling a chart method no longer draws on pyplot's current figure: `plt.savefig()` or `plt.show()` after it finds nothing, and `chart.save()` or `chart.show()` takes its place. Touching `result.plot` no longer imports matplotlib.
- **Breaking:** `ax=` is gone from every chart method and from `compare()`. A chart always has a figure of its own and is always framed; drawing into a caller's axes gave a chart with a plain title and none of the frame.
- The result charts as rebuilt: the weights chart's cap label is drawn (it sat just above the top bar and was clipped away), bars and contributions run from the largest at the top, percent axes read 5% rather than 5.0%, the constituents chart has no legend (each line is named at its end), the correlation heatmap writes each pair's value in its cell and labels its colour bar at its two ends with values in place of "less correlated" and "more correlated", and is cropped to the cells. The frontier's minimum-variance and tangency points take the palette's magenta and deep blue.
- Series take a new palette in order: blue, deep blue, magenta, pink, lime, green, olive, red and oxblood, then black (white on a dark canvas) and five greys. Orange is no longer a default: in `compare()` a second line is deep blue rather than purple, and a third magenta rather than orange. Marked periods get their own colour per theme.
- A chart's heading is the same height with or without a subtitle, so the accent bar is one size on every chart. A title with no subtitle is centred on the bar and the mark; with a subtitle the heading is unchanged.
- A chart's frame is laid out before it is drawn, without a renderer, so an interactive version can draw the same layout: where the plot sits, where each axis ends and what its ticks say, and where the heading, legend and notes go. The ticks are the same as before. Text is now measured from the bundled fonts rather than from the drawn image, so a plot's edges move by up to 0.014in. Notes now wrap at their measured width, so a line can hold more: the annual returns chart's notes fit on one line instead of two. A chart's measuring axes keep the ticks they were drawn with: setting new limits on the returned axes no longer picks new ticks.

## [0.7.0] - 2026-10-12

Funds: synthetic and futures-based replication, futures to equitise a cash buffer, withholding tax by fund domicile, share classes that share one pool of assets, including classes in another currency, hedged or not, and more ways to build an active portfolio. Reference data now says what each series is, and exchange rates are read from `CLOSE`: market data with a `RATE` column is refused, so a store saved before this release has to be regenerated.

### Added

- Synthetic replication: `IndexTracking(index, SyntheticReplication(basket, spread_bps=..., exposure_limit=0.10))` holds a substitute basket and a total return swap that pays the fund the index's return for the basket's, less a spread. The basket is required: an index definition, calculated over the run and held at its weights, or fixed weights. The swap is valued every day and included in the NAV, and is reset (settled in cash, its notional set back to net assets) when its value owed to the fund reaches the exposure limit (10% of net assets by default, the UCITS limit), at each of the index's rebalances, and on any day money flows in or out. `result.swap_value`, `result.swap_resets` and `result.counterparty_exposure` record it. A swap's settlement is a `"SWAP"` cash flow.
- Futures-based replication: `IndexTracking(index, FuturesReplication(roll_days=5, roll_cost_bps=..., dividend_yield=None))` holds cash, earning the run's cash rate, and long index futures sized to its net assets. Contracts are priced at fair value from the index level, the cash rate and the index's dividend yield, expire on the third Friday of each quarter and roll a set number of sessions before expiry at a cost in basis points. A price index's dividend yield is estimated at each roll from its constituents' trailing twelve months; a total return index's is zero. Profit and loss is settled in cash every day as a `"FUTURES"` cash flow.
- `Implementation(equitise=IndexFutures(...))` holds the same futures on a physical fund's uninvested cash, so its buffer earns the index's return. The server's backtest request takes `implementation.equitise`.
- `result.futures_notional`, `futures_rolls` and `futures_exposure` record a fund's index futures.
- Withholding tax by fund domicile: `Vehicle(domicile=..., withholding_rates=...)`. Each preset carries its domicile (UK OEIC the United Kingdom, Luxembourg SICAV Luxembourg, Irish ICAV and UCITS ETF Ireland, the US presets the United States). A dividend is withheld at the vehicle's own rate for its source country, then a built-in table for the domicile (on US dividends: Ireland 15%, Luxembourg 30%, the United Kingdom 15%, the United States nothing; nothing on UK dividends; nothing on Irish dividends to an Irish fund), then the run's flat rate. The source country is the company's `COUNTRY_DOMICILE` or `COUNTRY`, otherwise its listing exchange's country.

- A total return swap's value counts the dividends the underlying paid: `dividends` in `TotalReturnSwap`'s market data, and in the server's TRS pricing request, joins the total return leg as `notional * ((spot + dividends) / initial_price - 1)`. `dividend_yield` stays an input to the futures breakeven only, and its description now says why.

- Share classes that share one pool of assets: `Backtest(share_classes=[...], initial_capital={...}, flows={...})` runs every class in one book, so capacity caps, market impact and execution limits see the whole fund. Each day the book's gains, losses and income are split by the classes' net assets at the previous close; each class's own fee, flows (dealt at its own NAV per unit) and distributions reach it alone. `result.share_classes` holds each class's net assets, units, NAV per unit, fees, distributions and flows. `ShareClass` moved to `beacon.backtest`; `beacon.fund` still exports it.

- Share classes in another currency: `ShareClass(name, currency="EUR")` has its capital, flows, fees and distributions in its own currency, and its NAV per unit moves with the book's return and the exchange rate. `hedged=True` hedges it back into its own currency with one-month forwards priced at covered interest parity from the two currencies' short rates, rolled on the first business day of each month and resized when the hedge leaves `hedge_ratio` by more than `hedge_band` (3% by default). The forward's gains and losses are the class's alone; a settlement is a `"HEDGE"` cash flow, and `result.hedge_value` and each class's `hedges` and `hedge_value` record it.
- Reference data says what each series is: `ASSET_CLASS` (`Equity`, `Currency` or `Fixed Income`), `SECURITY_TYPE` (`Spot` for a currency pair, `Short Rate` for a short-term interest rate), a pair's `BASE_CURRENCY` and `QUOTE_CURRENCY`, a short rate's `TENOR` and `DAY_COUNT`, and `TRADABLE`. A row without them is an equity, so equity-only reference data needs no change. Currency pairs and short rates are not tradable unless `TRADABLE` says so, and are never offered as universe members. `pair_reference` and `short_rate_reference` in `beacon.data` build these rows.
- Short rates: `DataFetcher.short_rate_on(currency, date)`, `fetch_short_rates` and `short_rate` read a currency's rate, an annual decimal in `CLOSE`. Where a currency has several, the one month rate is read, else the overnight. A rate above 1 is refused as probably a percent.
- `DataFetcher.tradable_identifiers` and `is_tradable` say what a fund can hold.

- More ways to build an active portfolio: `MinimumVariance`, `MaximumDiversification`, `RiskParity(budgets)`, `MaximumSharpe(base_return, alpha_per_score)`, `BlackLitterman(risk_aversion, tau, alpha_per_score, confidence)`, `FactorTilts(factors, targets)` (each factor a signal) and `TopN(n, weighting)`, all long-only within the strategy's constraints. The first three and factor tilts need no signal, so `ActiveStrategy`'s `signal` is now optional for them. Each rebalance in `result.active` records every holding's share of the risk (`risk_contributions`) and what the construction reports (`exposures`), and the server's backtest results carry both.

### Changed

- **Breaking:** an exchange rate is read from `CLOSE`, like every other series, and a currency pair is the series reference data describes as `Currency`/`Spot`, found by its two currencies whatever its identifier. Market data with a `RATE` column is refused, with a message saying to move the rates to `CLOSE` and add a reference row per pair (`pair_reference("GBPUSD")` builds one). A series named like a pair but with no such row is no longer a pair. The importer's `fx` sheet is gone for the same reason, and the reference sheet requires `EXCHANGE` of equities only. `fetch_fx_rates` no longer takes `column`. Synthetic and sample data are written the new way.
- `fund.backtest()` runs every share class together, with `initial_capital` and `flows` given by class name (one number for a fund of one class), and returns the fund's result with each class in `result.share_classes`. It no longer takes `share_class=`, and `Fund.vehicle_for()` is gone: a class's fee now reaches it inside the pooled run.
- A backtest with a preset vehicle now withholds dividends by its domicile instead of the flat `withholding_tax_rate`, so its dividends, NAV and tracking difference change. Pass `domicile=None` to keep the flat rate.

## [0.6.0] - 2026-10-10

Charts: every chart framed as Beacon desktop frames it, with the beta mark, a heading and legend, and a notes line; six themes, including GitHub's dark page and transparent backgrounds; a constituents chart; axes that end on their ticks; and readable, consistently sized text. `IndexFund` and `ETF`, deprecated in 0.5.0, are removed.

### Added

- Charts that create their own figure are framed as Beacon desktop frames them: the beta mark, then a thin accent bar beside a title in Inter Tight and a subtitle in Source Serif 4 italic; the legend, when there is one, at the right of the heading; and one italic notes line saying where the data came from and the dates it covers. Every chart method and `compare` takes `title=`, `subtitle=` and `notes=` to replace them, or an empty string to leave one out, and `beacon.plot.frame_text(ax, part)` reads them back. The fonts ship with py-beacon under the SIL Open Font License.
- Themes: `beacon.plot.use()` takes `beacon-light` (or `light`), `beacon-dark` (or `dark`), `github-dark`, `white`, `transparent-dark-axes` and `transparent-light-axes`, each also a matplotlib style, and `palette()` takes a theme. A chart keeps the theme it was drawn in, so its own colours follow it on a transparent background.
- `constituents()` on an index or a backtest result: the index drawn over its constituents' closes, all rebased to 100, each named at the end of its line. It draws the eight largest by weight unless given another `limit`, and the notes say how many it left out. The quickstart's first chart now uses it.
- `correlation(shading="blended")` blends the heatmap's colours from cell to cell; `"squares"` is the default.

### Changed

- Charts are designed 6in wide and drawn at 80% of their size, with fixed text sizes: 8pt axis titles, 7.8pt tick labels, 5.5pt legends and labels on lines, and 4.5pt notes, all in the title's colour. The time series are 3:2. Date axes label the year at each January and the month between, and a chart's side margins widen to fit its labels.
- Every axis that measures something ends on its outermost ticks, and where both axes measure, the lowest y label is left blank so it does not sit on the first x label.
- The level charts (`level`, `constituents`, `performance` and `compare`) have no gridlines and draw one line at 100. The bar charts and the frontier have no gridlines either, with a line at zero where a bar can fall below it. Ticks point inwards, and the axes are thinner and in the title's colour.
- The frontier's x axis is labelled, and its tangency point is a deep blue rather than orange. The correlation heatmap has no tick marks, and its colour bar is labelled at each end.
- `compare` no longer prints a metrics table under the chart. Its notes say how many shared observations the comparison rests on.
- A chart's notes, and the `"At ..."` date on a weights chart, moved from the axes to the figure's notes line. Read them with `frame_text(ax, "notes")` rather than `ax.texts`. A chart drawn into axes you pass with `ax=` keeps its notes and legend in the axes.

### Deprecated

- `beacon.plot.use()`, `palette()` and `style_dict()` name their parameter `theme`; `mode=` still works, with a `DeprecationWarning`, and a call by position is unchanged.

### Removed

- `beacon.fund.IndexFund` and `beacon.fund.ETF`, as their deprecation in 0.5.0 said. Use a `Fund` with a vehicle: an open-ended preset such as `uk_oeic()` for an `IndexFund`, or `ucits_etf()` or `us_etf()` for an `ETF`.

### Fixed

- A dark chart saved after switching back to the light style was saved on the light background. A saved chart now keeps its own background.
- The last value marked at the end of a level chart's line was cut off at the right edge, and its dot was cut in half where the axis ends.
- The drawdown axis rounded a 2.5% step to whole percents, labelling -2.5% as -2% and -7.5% as -8%.

## [0.5.0] - 2026-10-08

Funds and strategies: money flowing in and out with units and fund accounting; fund vehicles with structure presets, four pricing methods, UCITS and 1940 Act limits, and ETFs with a modelled market price; a `Fund` product record; index tracking through optimised or sampled replication; active strategies built from a signal; and all of it through the server's backtest API. `IndexFund` and `ETF` are deprecated.

### Added

- Flows in a backtest: `Backtest(flows=...)` takes money in and pays it out, as `DatedFlows`, `PeriodicFlows` (an amount or a share of the assets), `RandomFlows` (seeded) or `PerformanceChasingFlows`, or several added together. Flows create and cancel units at the day's NAV per unit, so `nav_per_unit` measures performance whatever money moved, and with flows every return metric is a unit's and the summary adds `money_weighted_return`. The result records each flow in `flows`, and `aum` and `units_outstanding` each day. Without flows every figure is unchanged.
- `Implementation(cash_buffer=...)` keeps a share of the book in cash, which outflows draw on first, and `invest_flows` says whether an inflow buys the last rebalance's weights (the default) or the holdings pro rata.
- `Vehicle`: what the fund itself charges and its unit price at launch. `Vehicle(management_fee_bps=...)` accrues the fee daily on net assets (ACT/365) as a liability the NAV is net of, paid from cash as it can be.
- Pricing methods for a fund's dealing, so the investors who come or go can pay for the trading they cause: `SinglePricing` (the default), `DualPricing`, `SwingPricing` (full, or partial above a threshold) and `DilutionLevy`. A swing factor, spread or levy left unset is estimated from the run's own costs. Each flow records its `dealing_price` and the `adjustment` paid into the fund.
- Diversification limits for a fund structure, applied with the capacity caps: `UcitsLimits` (5/10/40, or 20/35 for an index tracker) and `Act1940Limits` (the 1940 Act's diversified-fund test).
- `OwnershipCap(free_float=False)` measures against every share outstanding rather than the free float.
- Structure presets: `uk_oeic`, `luxembourg_sicav`, `irish_icav` and `us_mutual_fund`, each a `Vehicle` with its structure's usual pricing and limits, any setting of which can be changed. `preset(name)` and `PRESETS` list them.
- Exchange-traded funds: `EtfVehicle`, and the `ucits_etf` (cash creations) and `us_etf` (in kind) presets. Flows are rounded to whole creation units, with the rest carried to the next day; in-kind creations move the holdings without cost, and cash ones pay a creation fee into the fund. After each close the shares are quoted on an exchange by `EtfMarket`: a premium that persists, follows creations and the NAV's return, and stays inside the arbitrage band, and a spread that follows the basket's trading cost and the NAV's volatility. `result.market` holds each day's quote and the summary adds the exchange investor's return and the average premium and spread.
- `Fund`: a fund product, with a name, a currency, `ShareClass`es and documents, one strategy and one vehicle. `fund.backtest(share_class=...)` runs a `Backtest` of it with that class's fee and distribution policy (accumulating or distributing).
- Strategies: `IndexTracking(index, replication=...)` holds an index in full, through an optimised subset (`OptimisedReplication`: the optimiser minimises tracking error within a holdings limit, using a covariance re-estimated at each rebalance from a year of returns, shrunk toward constant correlation), or through a stratified sample (`SampledReplication`: the largest names in each sector and size cell, each cell at its index weight). The run is measured against the index, and `result.replication` records each rebalance's weights, holdings and ex-ante tracking error. Pass it to `Backtest.run` or as a `Fund`'s strategy.
- Active strategies: `ActiveStrategy(benchmark, signal, construction, constraints)` builds its own portfolio at each rebalance and is measured against its benchmark. Signals: `FieldSignal`, `Momentum` and `FunctionSignal`, standardised into scores. Constructions: `MaxAlpha` (the most exposure to the scores within a tracking-error budget) and `MeanVariance` (expected active return against active variance), behind one `Construction` interface. Constraints relative to the benchmark: `TrackingErrorBudget`, `ActiveShare`, `RelativeSectorBounds`, `RelativePositionBounds`, `HoldingsLimit` and `TurnoverLimit`. `result.active` records each rebalance; the summary adds the information ratio, average active share and ex-ante tracking error; `active_attribution()` splits the active return by name.
- `BacktestResult.information_ratio()` for any run that tracks an index.
- The backtest API carries every part of a backtest: `POST /beacon/{index_id}/backtest` takes optional `strategy` (index tracking with a replication, or an active strategy), `implementation`, `flows`, `vehicle` (a preset with settings laid over it), `modelling_assumptions`, `dividends` and `currency`. A request without them runs as before. `GET /backtest/options` lists every type and field with units and bounds, the presets with their settings and the engine's default assumptions; `POST /beacon/{index_id}/backtest/validate` answers what would run, or findings at each field's path. The result adds the resolved `settings`, NAV per unit, units, flows, an ETF's quotes, replication and active records, the implementation's stages and unfilled orders, and saved-run rows add the strategy, vehicle, preset and currency.
- Field descriptions in `/backtest/options`, `/indices/rule-types` and `/optimise/constraint-types` carry a `unit` (such as `fraction`, `bps` or `money`) and `minimum` and `maximum` bounds, so a form can format a value and refuse one out of range.

### Deprecated

- `IndexFund` and `ETF`, to be removed in 0.6.0. Use a `Fund` with a vehicle: an open-ended preset such as `uk_oeic()` for an `IndexFund`, or `ucits_etf()` or `us_etf()` for an `ETF`. Until then they keep their own fee convention (a fee per trading day, applied to the backtest's NAV afterwards).

### Fixed

- An optimisation with a holdings limit re-solves the kept names from the first answer, with the dropped names at zero, rather than from its original starting point. From there the solver could stop on weights that were not fully invested and refuse a problem it could solve. Optimised indices, optimised replication and active strategies with a holdings limit can solve to slightly different weights.

## [0.4.0] - 2026-10-02

A backtest can now model how a fund carries out its index at its size: screens, capacity caps, market impact and execution limits, dividends, and its modelling assumptions in one object. Results change: a backtest receives dividends, sizes its buys net of costs, and keeps its book in the index's currency. `ExpressionScreen` moved to `beacon.backtest`.

### Added

- `Implementation`: how a backtest carries out its index at its size, without changing the index. Its screens decide which names may be held at each rebalance (`MarketCapScreen`, `LiquidityScreen`, `MinimumPriceScreen`, `ListingAgeScreen`, `ExclusionScreen`, `ExpressionScreen`), with optional exit levels (buffers) so names near a threshold do not flip in and out. Removed weight is spread pro rata across the remaining names or held as cash. Pass it as `Backtest(implementation=...)`.
- Capacity caps in a backtest's `Implementation`: `OwnershipCap` (a share of free-float market cap), `LiquidityCap` (days of traded value at a participation rate) and `WeightCap` limit each position at the book's size, with the excess spread across the names still under their caps or held as cash. `MinimumPosition` drops positions too small to keep. Each rebalance's `RebalanceStep` records what was capped.
- Market impact and execution limits in a backtest's `Implementation`. `MarketImpact` charges each trade for its size against the name's traded value, by the square-root law, so the same weights cost a large fund more. `ExecutionLimit` caps how much of an order trades in a day, by participation in the day's volume or spread over a number of days; the rest is worked on the following sessions. A blank volume is replaced by the last one reported within `volume_backfill_days` (a new modelling assumption, 5 days unless set), and otherwise by the average daily volume.
- `UnfilledOrder.reason`: why an order went unfilled, `"cash"`, `"no price"` or `"execution limit"`. The server's unfilled orders carry it too.
- `BacktestResult.rebalance_steps`: for each rebalance, the target, the names removed and the screen that removed each, and the weights traded to.
- `ModellingAssumptions`: what a backtest takes as given about markets and data in one object. FX policy, stale-price threshold and free-float backfill (shared with the index calculation), and cash rate, risk-free rate and periods per year. Set a process-wide default with `use_modelling_assumptions`, and override it per backtest with `Backtest(modelling_assumptions=...)`. Results and index results record what they assumed. The defaults change no result.
- Dividends in the backtest. A holding is paid each cash distribution on the pay date, for the shares held at the start of the ex-date, converted into the book's currency and net of `withholding_tax_rate` (a new modelling assumption). `Backtest(dividends=...)` reinvests the cash in the current holdings the day it arrives (the default), keeps it as cash until the next rebalance, or distributes it, with returns adding distributions back. Each payment is recorded in `portfolio.cash_flows`.
- Cash can earn interest (`cash_rate`), recorded in `Portfolio.cash_flows`.
- `IndexResult.currency` and `BacktestResult.currency`: the currency an index's levels and a backtest's book are in. A backtest run from the engine reports its `currency` too.
- `IndexFund` and `ETF` take the backtest settings as keyword arguments (`currency`, `modelling_assumptions`, `dividends`, `implementation`, `modifiers`, `benchmark` and `cache`) and pass them to the backtest they run. A fund ran with the defaults whatever was needed.
- `Backtest(cache=False)` turns the index result cache off. `cache=None` already meant the default location, so there was no way to ask for none.
- The index preview returns each constituent's market caps: `market_cap` and `free_float_market_cap` in the index currency, `market_cap_local` and `free_float_market_cap_local` in `local_currency`, with `market_cap_currency`, `priced_from` and `price_is_stale`. Every name has them, excluded ones included, at the preview's `resolved_date`, and they match `GET /data/reference` field for field. Null where a name has no price or the data has no free float.

### Changed

- `GET /data/reference` computes `market_cap` and `free_float_market_cap` about 12 times faster for a large batch: 0.3s for 1,000 names, from 1.8s. Each name's currency, latest row and FX rate were read separately; they are now read once for the batch. The answers are unchanged.
- A rebalance sizes its buys so that they and their costs fit the cash, so it fills in full. Buys were sized before costs, so the last buy of almost every rebalance with costs came up short and was recorded in `unfilled`.
- A target name with no price on a rebalance day is recorded in `unfilled` with the reason `"no price"`. It was skipped without a record. In the server's unfilled orders, `price` and `shortfall_value` can now be null.
- A backtest over data with cash distributions now receives them, so its NAV and every figure from it rise by about the dividend yield. It earned the price return only, and trailed a total-return index by about that much.
- `ExpressionScreen` is a screen, passed in `Implementation(screens=[...])`, and no longer takes a `fetcher`. It was a trade modifier that cancelled buys of a failing name, left its weight in cash and only trimmed a failing holding. A failing name is now removed from the target, sold in full and its weight redistributed. It moved from `beacon.backtest.rules` to `beacon.backtest`, and a screen passed in `modifiers` is refused with a message saying where it goes.
- A backtest keeps its book in the index's currency unless `currency` is passed. It defaulted to USD whatever the index's currency, so a euro or sterling index was valued in dollars and its NAV picked up exchange-rate moves the index does not have. This includes backtests run from the engine and by an `IndexFund`.

### Fixed

- An index over data with no `SHARES_OUTSTANDING` column is refused before the run starts, saying the calculation sizes every index from market cap. An equal-weighted index passed the check, since its scheme reads no columns, and then failed at the divisor as an index worth 0.0.
- The index result cache keys on the data source's FX policy, stale-price threshold and free-float backfill. It keyed on the store alone, so changing one of those settings could reuse an index calculated under the old one.

## [0.3.1] - 2026-10-01

One fix: futures and roll pricing refuse bad input as a 422 rather than a 404.

### Fixed

- Futures and roll pricing answer 422 `INVALID_ARGUMENT`, not 404 `DATA_NOT_FOUND`, for a negative or missing time to expiry, an expiry before the valuation date, and a back expiry that is not after the front.

## [0.3.0] - 2026-09-30

Money in one currency across multi-currency indices, prices carried over gaps, expressions that refuse fields the data does not have, and engine state that survives. Two breaking changes: `data.actions` is gone from expressions, and some requests the engine cannot answer as put answer 422 instead of 404.

### Added

- `DataFetcher.fetch_prices` reads prices for several instruments at once, converted into one currency day by day under the dataset's FX policy.
- `IndexResult.price_gaps` and `beacon.index.PriceGap`: the days a held name had no bar and was valued at its last close. `PriceGap` is still importable from `beacon.backtest`.
- The asset view reports `currency`, which its returns, beta and tracking error are measured in (the index's), and `price_currency`, which its price series is in (the name's own).
- A feature import's response says whether the rows were `saved` into the served store, and gives the new `data_version`.
- `BacktestResult.get_annual_returns()`: calendar-year returns that compound to the whole run's return.
- `TotalReturnSwap` takes `underlying_type` (`INDEX`, the default, `ETF` or `EQUITY`). It was always `INDEX`.
- Each release from `GET /changelog` carries its `summary`, the prose between its heading and its first section.
- A risk model request takes an optional `currency`, and a risk model and an optimisation run report the currency they were measured in.

### Changed

- Requests that cannot be answered as put answer 422 `INVALID_ARGUMENT` instead of 404 `DATA_NOT_FOUND`: an unknown price interval, an adjusted series without `CLOSE`, and an empty or oversized feature batch.
- An expression with a field the data does not have is refused with `ExpressionError` by `universe.where`, `ExpressionScreen` and an index's `ExpressionRule`, naming the field and suggesting close matches. Before, it selected nothing.
- `IndexDefinition` refuses an unknown calendar or rebalancing frequency when it is built, naming the valid values and suggesting close calendar codes. Before, the mistake surfaced only when rebalance dates were first computed.
- The API schema states more of what it accepts: `since` on `/changelog` is a dotted version, and `/beacon/compare` needs at least two `ids`, each a valid identifier. These are now refused as `VALIDATION_ERROR` by the schema check. The boolean fields of a synthetic-data or import request no longer accept `0`, `1` or strings.

### Removed

- `data.actions` in expressions. No expression could read its fields, so every name was missing a value; it now raises `UnknownDatasetError`, and the field catalogue and `GET /data/fields` no longer list an `actions` namespace. Read corporate actions from the fetcher's `corporate_actions`.

### Fixed

- Money is compared in one currency wherever names are compared or added up. `LiquidityRule`'s traded-value floor, the engine's risk models, optimisation runs, the weights pane (drift, risk contributions, active risk), attribution and the asset view all read each name's prices in its own currency, so a multi-currency index mixed yen with dollars and left exchange-rate returns and risk out. They now convert into the index's currency.
- An index values a held name with no bar on a session at its last close, as the backtest engine does, and lists the day in `IndexResult.price_gaps`. It used to value the name at zero, so the level dipped by the name's weight for the day and recovered when the bar returned.
- An expression's `market_cap` inside an index is in the index's currency, as `MarketCapRule`'s bounds are. Outside an index it stays in USD.
- `POST /data/features` now saves the rows into the served store when it is a folder, so they survive a restart, and changes `data_version` and sends a `data.freshness` event for `features`. Before, a client caching on `data_version` missed the change and a restart lost the rows.
- The engine always keeps the saved results its views read: the latest backtest of each index, every optimisation run and the latest estimate of each risk model. The limit of 50 saved jobs now applies only to the rest, so a run of loads, refreshes or renders no longer makes those views answer 404.
- `data.market.free_float` and `free_float_market_cap` in an expression carry the free float forward as far as the data's `free_float_backfill_days`, as every other read does, rather than 10 days.
- Attribution for a capped index no longer fails when its window starts after the index's first rebalance. The cap drag is measured over the same periods as the contributions.
- A cancelled dividend is no longer reinvested by a total-return index, counted in the trailing dividend and yield, or applied to adjusted closes; a cancelled split is no longer applied to adjusted closes either.
- The engine checks for a token before it loads any data, so a missing token is reported at once rather than after a large store has loaded.
- `http://127.0.0.1` and `http://[::1]` on any port are allowed origins, as `http://localhost` already was.
- A wrong or missing token on the event socket closes it with code 1008 and a reason, as intended, instead of refusing the handshake with an HTTP 403 a browser cannot read.
- `BacktestPlots.annual_returns` measures each year from the previous year's close, and the first from the initial capital. It measured each year from its own first close, so every year after the first lost its first day's return.
- A chart drawn before `beacon.plot.use()` gets the light colours on matplotlib's white background, not the dark mode's.
- `OptimisationPlots.frontier` starts the capital market line at the rate the frontier was traced at, unless another is passed, so the line is tangent.
- The Excel reports accept a `pathlib.Path` as well as a string, and the holdings report writes `valuation_date` into its sheet instead of only logging it.
- `futures_roll_return` uses 365-day years, like every other year fraction in the derivatives. It used 365.25, so its rates came out smaller than the rest by a factor of 365/365.25.
- The sample dataset in `beacon.testing.dataset` stores its GBPUSD rate in `RATE`, so `fx_pairs` and the coverage view list it. A pair stored without `RATE` still converts but now logs a warning that it is not listed.
- A report template with an unknown page setting answers 422 instead of 500, and so does a futures or swap price whose inputs overflow.
- A plain `OPTIONS` request answers 405 with an `Allow` header listing every method its path supports. It listed only one route's methods on a path with several, such as `/data/features`. CORS preflights are unchanged.
- The nightly API fuzz run passes again: it leaves out operations that start heavy work, and skips the check that a schema-valid request is accepted only where a rule spans fields or depends on the data.
- Generating or extending synthetic data from an isolated engine (`python -I`, as the Beacon app runs it) keeps the child process isolated too, so it cannot load packages from the user's own site-packages.

## [0.2.0] - 2026-09-26

Choose where the engine's data comes from: named stores, synthetic data you can extend to today, CSV and Excel import, and read-only Postgres. Also documentation at pybeacon.dev, and fixes to index levels and backtests around splits and delistings.

### Added

- Documentation at https://pybeacon.dev: a guide to each part of py-beacon (data, universes, methodology, expressions, backtests, funds, derivatives, the optimiser, risk, attribution, charts, reports and the server), and a complete Python reference. Every example in it runs.
- The engine can start without any data and load a store later. Loading runs as a job with progress, and the event socket announces `data.loaded` when the new data is being served.
- Named data stores. Register a data folder under a name you choose, see every store at `GET /data/stores`, and switch which one the engine serves. The engine remembers the active store and serves it again on the next start.
- Generate synthetic data from the engine: `POST /data/synthetic` creates a new store with the size, dates and seed you choose (end date defaulting to today), and serves it when it is ready. It runs as a job with progress, and gives exactly the same data as `python -m beacon.synthetic` with the same settings.
- `python -m beacon.synthetic --progress` prints a line at each stage, for a program running it.
- Extend a generated store to today without changing its history: `python -m beacon.synthetic --extend PATH`, or `beacon.synthetic.extend`. Prices, exchange rates, listings, dividends, splits and features carry on from where the store stops, and the rows already there are never changed. The same store extended to the same date always gives the same data. Stores generated from this version on keep the settings this needs beside the data.
- Import your own data from CSV files or an Excel workbook: `POST /data/import` checks every row and saves a new store, or refuses with one finding per problem naming its sheet, row and column. `GET /data/import/template` downloads a blank template, and `beacon.data.importing.load_files` does the same from Python. Dates are read as YYYY-MM-DD only, so a date cannot be read as the wrong day.
- The `server` extra now includes `excel`, for Excel import.
- Refresh a store from its own source: `POST /data/stores/{id}/refresh` extends synthetic data to today, reads a folder or database again, and saves what changes. If the store is being served, the engine serves the refreshed data. Each store in `GET /data/stores` says what a refresh would do.
- A folder store can choose to refresh from Yahoo Finance instead (`refresh_from`), which downloads new prices and saves them into the folder. It is never the default.
- A data store can be a Postgres database. Give it tables or views named `market`, `reference`, and optionally `fx`, `corporate_actions` and `features`, with the import template's columns; views over your own tables work. The engine only reads, over a read-only connection, and checks every row as an import does. The password is never stored: name an environment variable that holds it. Needs the new `postgres` extra.

### Changed

- A backtest's return metrics start from its initial capital: `get_returns()` has one return per simulated day, the first from the capital to the first close. The tracking difference, tracking error, volatility and drawdown now include the cost of the opening trades, which they used to leave out, so a run with costs no longer shows a positive tracking difference for that reason.
- In a backtest result from the engine, `returns`, `drawdown` and `annual_returns` start from the initial capital too, so they agree with the metrics. `returns` now has one value per day of `level`, the first from the capital to the first close. `level` is unchanged.
- `POST /data/coverage/{dataset}/sync` is deprecated. It now refreshes the store being served from that store's own source, and no longer downloads from Yahoo Finance unless the store chose it. Its body is ignored, and its job is a `refresh:{store_id}` job. Use `POST /data/stores/{id}/refresh`.
- A request that needs data when none is loaded now answers 409 `NO_DATA_LOADED`, saying what could not be done. It answered 500 `CONFIGURATION_ERROR` before.
- `/health` says which store is being served and whether one is loading.
- `/health` and the data events carry `data_version`, a token that changes whenever the data being served changes. A client compares it to tell whether what it cached is still current.

### Fixed

- `IndexFund.calculate_nav` keeps the fund's trading cost when it extends a run to a later date. It used to re-run at zero cost.
- A split, reverse split or stock dividend no longer moves an index level or a backtest's NAV. On the ex-date the units or shares held change by the ratio, as the price does. Before, a split between rebalances cut the level and NAV by the split, including in synthetic data, which splits every year.
- A constituent delisted the session before a rebalance no longer takes its weight out of the index level. It leaves first, as on any other day.
- `SelectionResult.excluded_by` names the right step when prices go stale. With stale names dropped, it credited each exclusion to the step before it, and a stale name to the last rule.
- A damaged data file no longer stops the engine with a crash. It is refused with a message naming the file, and a damaged store found at startup is skipped with a warning.

## [0.1.1] - 2026-09-25

### Added

- `DataFetcher.fx_route` says how a currency pair is converted: from a stored rate, its inverse, or a cross through USD.

### Fixed

- A currency with no stored rate of its own is now converted using the inverse of the reverse pair, or a cross through USD. Before, an index in pounds holding US shares refused because only GBP to USD was stored.
- The quickstart in the README and on the docs home page runs again.

## [0.1.0] - 2026-09-24

The first release.

### Added

- Build an index from rules and a weighting scheme: equal weight, market cap or free-float market cap, with optional weight caps.
- Calculate the index level on a real exchange calendar, as price, total return or net total return.
- Adjust the index for dividends, special dividends and delistings.
- Write selection rules as expressions, such as `data.market.market_cap > 1e9`, including rules on company fundamentals and other features.
- Backtest a portfolio that trades to the index weights, with transaction costs, drift thresholds and a benchmark.
- Read results with tracking error, attribution, risk, concentration and drift, and draw them as charts.
- Hold names in several currencies. Prices are converted with FX rates.
- Load data from files, generate a realistic synthetic dataset, or download prices with yfinance. Data is kept in a local store that reports its coverage and age.
- Three settings, shown on `/health`: whether a missing FX rate carries forward, when a stale price drops a name, and how long a free float carries forward (90 days by default).
- A local API server for the Beacon desktop app, covering data, indices, universes, backtests, the optimiser, risk, derivatives and PDF reports. Long tasks run as jobs with live progress.
- This changelog, served at `GET /changelog` so the app can show what changed.
- Optional extras keep the core install small: `data`, `excel`, `pdf`, `optimise`, `plot` and `server`.

### Notes

- An index or backtest never uses a price, rate or free float dated after the day it is working on.
- Requires Python 3.11 or later.

[Unreleased]: https://github.com/karanbh01/py-beacon/compare/v0.9.4...HEAD
[0.9.4]: https://github.com/karanbh01/py-beacon/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/karanbh01/py-beacon/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/karanbh01/py-beacon/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/karanbh01/py-beacon/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/karanbh01/py-beacon/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/karanbh01/py-beacon/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/karanbh01/py-beacon/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/karanbh01/py-beacon/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/karanbh01/py-beacon/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/karanbh01/py-beacon/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/karanbh01/py-beacon/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/karanbh01/py-beacon/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/karanbh01/py-beacon/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/karanbh01/py-beacon/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/karanbh01/py-beacon/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/karanbh01/py-beacon/releases/tag/v0.1.0
