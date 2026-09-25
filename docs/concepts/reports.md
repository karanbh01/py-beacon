# Reports

py-beacon writes two kinds of report:

- **PDF documents** (`beacon.report`): a page laid out from blocks, for
  reading as it was designed, such as a factsheet.
- **Excel workbooks** (`beacon.portfolio.reporting`): holdings and performance
  tables, for someone to pick up and work with.

Building a PDF template needs nothing extra. Rendering it needs the `pdf`
extra (reportlab), and Excel output needs the `excel` extra (openpyxl):

```bash
pip install "py-beacon-kit[pdf,excel]"
```

## A template is data

A `ReportTemplate` is a page setup and an ordered list of blocks. The blocks
are plain frozen dataclasses in `beacon.report.blocks` (also importable from
`beacon.report`). They know nothing about drawing, so a template can be built,
saved as JSON and edited without reportlab installed.

| Block | Draws |
| --- | --- |
| `Header(title, subtitle="", as_of="")` | A title band with a rule beneath; the as-of text sits on the right |
| `Text(body, size=9.0, muted=False)` | A paragraph, wrapped to the page width; `\n` starts a new line |
| `StatGrid(stats, columns=4)` | Headline figures, each a `Stat(label, value, change="")`, wrapping onto new rows |
| `Table(columns, rows, title="", align_right=[])` | A header row and data rows; `align_right` lists column positions to right-align |
| `BarChart(categories, values, title="", height=140.0)` | Horizontal bars drawn directly in the PDF; negative values extend left of a centre axis |
| `Chart(image_path=None, title="", height=200.0)` | A rendered image file, scaled to the page width with its aspect ratio kept |

Every value shown on the page is a string you format yourself (`Stat.value`,
table cells, `as_of`), because only the caller knows whether a number is a
percentage, an amount or a ratio. Heights are in points (1/72 inch).

`PageSetup(size="A4", orientation="portrait", margin=48.0)` sets the sheet.
Sizes are `A4`, `LETTER` and `A5`; orientation is `portrait` or `landscape`;
the margin, in points, applies to all four sides.

Blocks check themselves when built and raise `ReportingError`: a table row
with the wrong number of cells, a bar chart with mismatched or no categories,
a stat grid with fewer than one column, or an unknown page size or
orientation.

## Building a template

This example draws a correlation chart to a PNG, then lays out a one-page
factsheet around it. The chart needs the `plot` extra.

```python
import matplotlib.pyplot as plt

import beacon.plot
from beacon.report import (
    BarChart, Chart, Header, PageSetup, ReportTemplate,
    Stat, StatGrid, Table, Text,
)
from beacon.risk import estimate_risk_model
from beacon.testing import dataset

risk = estimate_risk_model(dataset.returns(), intensity=0.1)

beacon.plot.use("light")
ax = risk.plot.correlation()
ax.figure.savefig("correlation.png", dpi=150)
plt.close("all")

weights = dataset.equal_weights()

template = ReportTemplate(
    template_id="factsheet",
    name="Canonical factsheet",
    page=PageSetup(size="A4", orientation="portrait"),
    blocks=[
        Header("Canonical Index", subtitle="Monthly factsheet", as_of="31 Dec 2024"),
        StatGrid([Stat("Level", "1,842.10", "+2.4% MTD"),
                  Stat("1Y return", "18.42%"),
                  Stat("Volatility", "15.84%"),
                  Stat("Constituents", str(len(weights)))]),
        Text("Six synthetic companies, rebalanced quarterly.", muted=True),
        BarChart(list(weights), [0.012, 0.009, 0.004, 0.021, -0.006, 0.003],
                 title="Contribution to return"),
        Table(["Constituent", "Weight"],
              [[name, f"{weight:.1%}"] for name, weight in weights.items()],
              title="Holdings",
              align_right=[1]),
        Chart(image_path="correlation.png", title="Correlation", height=180.0),
    ])
```

`BarChart` suits a few labelled values such as top holdings, sector weights or
contributions, and needs no plotting library. `Chart` holds a path to an image
drawn elsewhere, usually by [`beacon.plot`](charts.md). If `image_path` is
`None` or the file does not exist, the renderer draws a dashed placeholder
labelled with the chart's title, so a layout can be reviewed before its
charts exist.

## Rendering to PDF

`beacon.report.pdf.render(template, output_path)` draws the template and
returns the path it wrote. Parent folders are created as needed.

```python
from beacon.report.pdf import render

path = render(template, "factsheet.pdf")
print(path, path.stat().st_size, "bytes")
```

What to expect from the renderer:

- **One page.** Blocks are laid out top to bottom. If they do not fit, it
  raises `ReportingError` naming the block that overflowed and how much room
  was left. It does not paginate.
- **Deterministic output.** Two renders of the same template produce
  byte-identical files, so a hash confirms that two people got the same
  document.
- **Fixed print colours.** Colours come from the mode-independent paper
  design tokens, so a PDF looks the same whatever theme the application is
  using.
- **Standard fonts.** Helvetica, which every PDF reader has, so nothing is
  embedded.
- The template's `name` becomes the PDF title and its `template_id` the
  subject.

Importing `beacon.report.pdf` without reportlab raises
`MissingDependencyError` naming the `pdf` extra. `beacon.report.blocks` always
imports.

## Saving and loading templates

`to_dict()` gives a JSON-ready form, with each block tagged by a `kind`, and
`ReportTemplate.from_dict()` reads it back:

```python
import json

from beacon.report import ReportTemplate, block_from_dict

stored = json.dumps(template.to_dict(), indent=2)
restored = ReportTemplate.from_dict(json.loads(stored))
assert restored == template

note = block_from_dict({"kind": "text", "body": "Past performance is not a guide."})
print(note)
```

`block_from_dict` rebuilds a single block. The kinds are `header`, `text`,
`stat_grid`, `table`, `bar_chart` and `chart`. An unknown or missing kind
raises `ReportingError` rather than being skipped, since skipping would drop
content from the report without saying so.

The API server stores templates under `/reports/templates` and renders them
as a job through `POST /reports/render`, with the PDF downloaded from
`GET /reports/renders/{render_id}`. It also offers a built-in `FACTSHEET-A4`
template generated from an index's latest run. See the
[Server guide](../server.md).

## Excel reports

`ReportGenerator` in `beacon.portfolio.reporting` writes two workbooks. Both
check for openpyxl when called, append `.xlsx` to a path that lacks it, and
raise `ReportingError` if writing fails.

```python
import pandas as pd

from beacon.portfolio.base import Portfolio
from beacon.portfolio.reporting import ReportGenerator

portfolio = Portfolio("DEMO", initial_cash=100_000.0)
portfolio.execute_buy("AAA", 100, 120.0, cost=1.2, date=pd.Timestamp("2024-12-30"))
portfolio.update_prices({"AAA": 125.0}, date=pd.Timestamp("2024-12-31"))

reports = ReportGenerator()
reports.generate_holdings_report_excel(portfolio,
                                       "holdings.xlsx",
                                       valuation_date=pd.Timestamp("2024-12-31"))

reports.generate_performance_report_excel(portfolio.nav.to_frame("nav"),
                                          "performance.xlsx",
                                          report_title="Demo performance")

print(pd.ExcelFile("holdings.xlsx").sheet_names)   # HoldingsSummary, TransactionHistory
```

**`generate_holdings_report_excel(portfolio, report_path, valuation_date)`**
writes a `HoldingsSummary` sheet from `portfolio.get_holdings_summary()`,
including a cash row, and a `TransactionHistory` sheet when the portfolio has
transactions. It reports the portfolio as it stands, so call
`update_prices(...)` first for current market values and weights.
`valuation_date` is used only in log messages; it is not written to the
workbook.

**`generate_performance_report_excel(performance_data, report_path, report_title="Performance Report")`**
writes a non-empty DataFrame, index included, to one sheet. The sheet is
named after `report_title` with spaces replaced by underscores and cut to 30
characters, or `PerformanceData` when the title is `None`. It raises
`ValueError` for an empty frame or anything that is not a DataFrame. A
backtest's NAV works well here: `result.trading_nav.to_frame("nav")` (see
[Backtest](backtest.md)).

Pass `report_path` as a string: both methods call `str.endswith` on it, so a
`pathlib.Path` fails.

The full API is in the [Reports reference](../reference/report.md) and the
[Portfolio reference](../reference/portfolio.md).
