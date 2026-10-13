# scripts/build_gallery.py
"""
Render every chart and write the documentation gallery.

Run at docs build time, so the gallery shows what the code currently draws
rather than screenshots someone took once. A gallery that can go stale is worse
than no gallery: it becomes a record of what the charts used to look like, and
nobody knows which images are still true.

Every chart (each result's and each chart class's) is drawn from
`beacon.testing.dataset` in every theme, and beside it a live interactive
version in the light and dark themes. The data is reproducible, so the same
run on any machine produces the same pictures, for the same reason the
image-regression baselines can exist at all.

Usage:

    python scripts/build_gallery.py [output_directory]

Defaults to `docs/gallery`. The page is written as `gallery.md` beside the
output directory, linking the files as `gallery/<file>`:

- `{name}.light.png` and `{name}.dark.png`: the Beacon themes. pybeacon.dev
  shows the one matching the reader's light or dark setting.
- `{name}.{theme}.png`: the other four themes, shown in both settings.
- `{name}.light.html` and `{name}.dark.html`: the live versions, which fill
  the frame the page gives them. They share `plotly.min.js`, and `fonts.css`
  with the font files in `fonts/`.
"""
import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from beacon.analysis import attribute, drifted_weights
from beacon.backtest.engine import BacktestEngine
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.optimise import (
    FullInvestment,
    GroupBounds,
    PositionBounds,
    minimise_tracking_error,
)
from beacon.optimise.frontier import efficient_frontier
from beacon.plot import (
    Area,
    Bar,
    Column,
    Combo,
    Grid,
    Heatmap,
    Histogram,
    Line,
    Scatter,
    Table,
    compare,
    settings,
    use,
)
from beacon.plot.interactive.html import figure_fonts
from beacon.plot.webfonts import font_faces, font_files
from beacon.risk import estimate_risk_model
from beacon.testing import dataset

DEFAULT_OUTPUT = Path("docs/gallery")

END = "2024-12-31"
CAP = 0.20
RISK_FREE = 0.02

# Print resolution, so the charts stay sharp on a high-density display and
# when zoomed.
DPI = 300

# The live versions are drawn this much larger than the images' design, the
# width of the site's text column (about 600px, from a 461px design), so their
# text and lines are sized for it; they then fill whatever frame the page gives
# them.
LIVE_SCALE = 1.3

# Each theme and the suffix its images take: the Beacon themes as .light and
# .dark, which the site swaps by the reader's setting, and the others by name,
# shown in both. With what each theme is for.
THEMES = {
    "beacon-light": ("light", "Beacon's light canvas."),
    "beacon-dark": ("dark", "Beacon's dark canvas."),
    "github-dark": ("github-dark", "GitHub's dark page."),
    "white": ("white", "A white page."),
    "transparent-dark-axes": ("transparent-dark-axes",
                              "No background, dark text: for a light page."),
    "transparent-light-axes": ("transparent-light-axes",
                               "No background, light text: for a dark page."),
}
LIVE = ("beacon-light", "beacon-dark")

# Chart id -> its title and what it is for, shown as the caption. Written here
# rather than scraped from docstrings because a caption and an API docstring
# answer different questions.
RESULT_CHARTS = {
    "level": ("Index level", "The index level, rebased to 100, against an optional benchmark."),
    "weights": ("Weights", "Constituent weights at a rebalance, with the weight cap marked."),
    "constituents": ("Constituents", "The index among its constituents, each rebased to 100."),
    "performance": ("Performance", "Growth of 100, with a drawdown panel on the same dates."),
    "annual_returns": ("Annual returns",
                       "Calendar-year returns, green for gains and red for losses."),
    "contributions": ("Contributions", "Each constituent's contribution to return, with the "
                                       "cap and cost drags in the notes."),
    "compare": ("Compare", "Several results on one axis, rebased to 100 on the dates they share."),
    "exposures": ("Exposures", "Active weights against the target, coloured by sign."),
    "frontier": ("Frontier", "An efficient frontier, its minimum-variance and tangency points, "
                             "and the capital market line."),
    "correlation": ("Correlation", "The correlation matrix as a heatmap, shaded from 0.2 to 1.0."),
}
CLASS_CHARTS = {
    "line": ("Line", "One or more series over dates, the first in the accent colour, with a "
                     "reference line, a marked period and the last value labelled."),
    "area": ("Area", "Parts of a whole stacked, here four holdings' shares."),
    "column": ("Column", "Values by category, upright, coloured by sign."),
    "bar": ("Bar", "Values by category, across, with value labels and a cap."),
    "combo": ("Combo", "Columns on the left axis and a line on the right, on one calendar."),
    "scatter": ("Scatter", "One measure against another, with a fitted line and its slope "
                           "and fit."),
    "histogram": ("Histogram", "Counts in equal bins, with the normal curve and a marked VaR."),
    "heatmap": ("Heatmap", "A table of values as colour, each value written in its cell."),
    "grid": ("Grid", "Several charts under one heading: a tall chart beside two."),
    "table": ("Table", "A DataFrame as a table, with column groups, row groups and signed "
                       "returns; live, its columns sort."),
}


def build_results() -> dict[str, object]:
    """Everything the gallery draws from, computed once."""
    fetcher = dataset.data_fetcher()

    definition = IndexDefinition(
        index_id="CANON", index_name="Canonical Index",
        base_date=dataset.START, base_value=1000.0, currency="USD",
        eligibility_rules=[], weighting_scheme=MarketCapWeighted(),
        rebalancing_frequency="QUARTERLY",
        calendar="XNYS",
        universe_identifiers=list(dataset.UNIVERSE),
        max_constituent_weight=CAP)

    index_result = IndexCalculator(definition, fetcher).run(
        start_date=dataset.START, end_date=END)

    backtest = BacktestEngine(
        start_date=dataset.START, end_date=END, initial_capital=10_000_000.0,
        data_provider=fetcher, index_result=index_result,
        transaction_cost_bps=10.0).run()

    prices = dataset.prices().loc[:END]
    weights = drifted_weights(index_result.weight_snapshots, prices)
    asset_returns = prices.pct_change().reindex(weights.index)
    period_returns = (weights.shift(1) * asset_returns).sum(axis=1)
    attribution = attribute(period_returns, weights, asset_returns,
                            cap_drag=-0.003, cost_drag=-0.004)

    risk_model = estimate_risk_model(dataset.returns(), intensity=0.1)

    optimisation = minimise_tracking_error(
        dataset.equal_weights(),
        [FullInvestment(), PositionBounds(0.0, 0.25),
         GroupBounds("Technology", dataset.sectors()["Technology"],
                     maximum=CAP)],
        risk_model)

    expected = {name: 0.04 + 0.02 * position
                for position, name in enumerate(dataset.UNIVERSE)}
    frontier = efficient_frontier(risk_model, expected, points=12,
                                  risk_free_rate=RISK_FREE)

    return {"index": index_result, "backtest": backtest,
            "attribution": attribution, "risk": risk_model,
            "optimisation": optimisation, "frontier": frontier,
            "prices": prices, "benchmark": prices["CCC"]}


def result_calls(results: dict[str, object]) -> dict[str, Callable[[], object]]:
    """Each result chart's id to a callable that builds it."""
    index = results["index"]
    backtest = results["backtest"]

    return {
        "level": lambda: index.plot.level(benchmark=results["benchmark"]),
        "weights": lambda: index.plot.weights(),
        "constituents": lambda: index.plot.constituents(),
        "performance": lambda: backtest.plot.performance(),
        "annual_returns": lambda: backtest.plot.annual_returns(),
        "contributions": lambda: results["attribution"].plot.contributions(),
        "compare": lambda: compare(index, backtest),
        "exposures": lambda: results["optimisation"].plot.exposures(),
        "frontier": lambda: results["optimisation"].plot.frontier(
            results["frontier"], risk_free_rate=RISK_FREE),
        "correlation": lambda: results["risk"].plot.correlation(),
    }


def class_calls(prices: pd.DataFrame) -> dict[str, Callable[[], object]]:
    """Each chart class's id to a callable that builds an example of it."""
    rebased = prices / prices.iloc[0] * 100
    held = prices[["AAA", "BBB", "CCC", "DDD"]]
    monthly = prices["CCC"].resample("ME").last().pct_change().dropna().loc["2024"]
    monthly.index = monthly.index.strftime("%b")
    weights = pd.Series([0.0792, 0.0741, 0.0688, 0.0617, 0.0583, 0.0526],
                        index=["Northwind Systems", "Halcyon Health", "Arden Industrials",
                               "Meridian Software", "Corvid Semiconductors", "Larkspur Pharma"],
                        name="Weight")
    daily = prices["AAA"].pct_change().dropna().rename("Daily return")
    drawdown = (prices["AAA"] / prices["AAA"].cummax() - 1).rename("Drawdown")
    returns_2024 = prices.loc["2024"].iloc[[0, -1]].pct_change().iloc[-1].sort_values(
        ascending=False).rename("Return")

    def grid() -> Grid:
        growth = Line(rebased[["AAA", "BBB"]], title="Growth of 100", reference=100)
        ranks = Bar(returns_2024, title="Returns in 2024", percent=True, value_labels=True)
        falls = Area(drawdown, title="AAA's drawdown", percent=True)
        return Grid([[growth, ranks], [falls, ranks]], heights=(3, 1), title="Two holdings",
                    subtitle="A tall chart beside two", source="Synthetic data")

    def table() -> Table:
        performance = pd.DataFrame(
            {"1Y": [8.42, 7.91, 6.10, -2.35, 11.2, 4.4], "3Y": [5.12, 4.80, 3.95, 0.40, 6.8, 2.2],
             "Volatility": [14.2, 13.8, 15.1, 18.4, 21.0, 9.5],
             "Sharpe": [0.61, 0.58, 0.44, -0.12, 0.52, 0.31]},
            index=["Beacon Global", "MSCI World", "Peer median", "Peer low", "Growth fund",
                   "Income fund"])
        return Table(performance, title="Performance and risk",
                     subtitle="To 31 December 2024, against the index and peers",
                     groups={"Return (%)": ["1Y", "3Y"], "Risk": ["Volatility", "Sharpe"]},
                     formats={"Return (%)": "{:.1f}", "Risk": "{:.2f}"},
                     signed=["Return (%)"], highlight=["Beacon Global"],
                     row_groups={"Ours": ["Beacon Global", "MSCI World"],
                                 "Peers": ["Peer median", "Peer low", "Growth fund",
                                           "Income fund"]},
                     row_group_headings=True, source="Synthetic data")

    return {
        "line": lambda: Line(rebased[["AAA", "BBB", "CCC"]], title="Three holdings",
                             subtitle="Each rebased to 100", ylabel="Level", reference=100,
                             label_last=True,
                             periods={"First half": ("2023-03-01", "2023-06-30")},
                             source="Synthetic data"),
        "area": lambda: Area(held.div(held.sum(axis=1), axis=0), title="Shares",
                             subtitle="Each holding's share of the four", ylabel="Share",
                             percent=True, source="Synthetic data"),
        "column": lambda: Column(monthly.rename("CCC"), title="Monthly returns",
                                 subtitle="CCC in 2024", ylabel="Return", percent=True,
                                 source="Synthetic data"),
        "bar": lambda: Bar(weights, title="Largest holdings", subtitle="Capped at 8%",
                           percent=True, value_labels=True, decimals=2, cap=0.08,
                           source="Synthetic data"),
        "combo": lambda: Combo(prices["AAA"].diff().loc["2024"].rename("Net flows"),
                               prices["BBB"].loc["2024"].rename("Assets"), gap=0,
                               title="Daily flows and assets",
                               subtitle="Each column runs to the next trading day",
                               ylabel="Net flows", ylabel_secondary="Assets",
                               source="Synthetic data"),
        "scatter": lambda: Scatter(prices[["AAA", "DDD"]].pct_change().dropna().loc["2024"],
                                   x="DDD", y="AAA", fit="linear", percent=True,
                                   title="Sensitivity",
                                   subtitle="AAA's daily returns against DDD's, 2024",
                                   source="Synthetic data"),
        "histogram": lambda: Histogram(daily, bins=40, normal=True, percent=True,
                                       mark={"95% VaR": float(daily.quantile(0.05))},
                                       title="Daily returns",
                                       subtitle="How often AAA moved by each amount",
                                       source="Synthetic data"),
        "heatmap": lambda: Heatmap(prices.pct_change().dropna().corr(), title="Correlation",
                                   subtitle="Between daily returns", colormap="beacon_corr",
                                   vmin=0.2, vmax=1.0, source="Synthetic data"),
        "grid": grid,
        "table": table,
    }


def render_images(directory: Path,
                  calls: dict[str, Callable[[], object]]) -> None:
    """Every chart in every theme, as a PNG."""
    for theme, (suffix, _) in THEMES.items():
        use(theme)
        for name, call in calls.items():
            call().save(directory / f"{name}.{suffix}.png", dpi=DPI)

    use("beacon-light")


def render_live(directory: Path,
                calls: dict[str, Callable[[], object]]) -> dict[str, tuple[float, float]]:
    """Every chart's live version in the light and dark themes: a plotly
    figure that fills its frame, or a table's HTML.

    Returns:
        dict: Each chart's width and height in pixels, for its frame's shape.
    """
    shapes: dict[str, tuple[float, float]] = {}
    fonts = set()

    for theme in LIVE:
        use(theme)
        suffix = THEMES[theme][0]
        for name, call in calls.items():
            # The heading keeps its own scale, so it is zoomed with the rest.
            with settings(scale=LIVE_SCALE, heading_scale=LIVE_SCALE):
                chart = call()
            target = directory / f"{name}.{suffix}.html"
            if isinstance(chart, Table):
                target.write_text(_page(chart.to_html()), encoding="utf-8")
                shapes[name] = (chart.layout.width * 96.0, chart.layout.height * 96.0)
                continue
            figure = chart.interactive()
            fonts |= figure_fonts(figure)
            figure.write_html(target, include_plotlyjs="directory", fonts="fonts.css", fit=True)
            shapes[name] = (figure.layout.width, figure.layout.height)

    use("beacon-light")
    font_files(fonts, directory / "fonts")
    (directory / "fonts.css").write_text(font_faces(fonts, base="fonts/"), encoding="utf-8")

    return shapes


def _page(fragment: str) -> str:
    """A table's HTML as a page of its own, for a frame."""
    return ('<!doctype html><html><head><meta charset="utf-8">'
            "<style>html,body{margin:0}</style></head><body>" + fragment + "</body></html>")


def write_page(directory: Path,
               shapes: dict[str, tuple[float, float]]) -> Path:
    """Write the gallery's markdown page.

    Each chart's image and live version are linked in both modes:
    pybeacon.dev shows the ones matching the reader's light or dark setting,
    and a local preview shows both. The other themes follow in a row.
    """
    lines = [
        "---",
        "title: Gallery",
        'description: "Every chart py-beacon draws, in every theme, beside a live '
        'interactive version."',
        "---",
        "",
        "# Chart gallery",
        "",
        "Every chart py-beacon draws, rendered from the sample dataset",
        "(`beacon.testing.dataset`). The images are generated when the docs",
        "are built, so they show what the current code draws. Each is drawn",
        "in the light and the dark theme, and the one shown follows your",
        "light or dark setting. Beside it is the live version: move the",
        "pointer over it for the popup. The other four themes follow, for",
        "pages that are white, GitHub's dark, or let the page show through.",
        "[Charts](concepts/charts.md) explains how to draw them.",
        "",
    ]
    others = [theme for theme in THEMES if theme not in LIVE]

    for section, charts in (("Results", RESULT_CHARTS), ("Chart classes", CLASS_CHARTS)):
        lines += [f"## {section}", ""]
        for name, (title, caption) in charts.items():
            width, height = shapes[name]
            ratio = f"{width:.0f} / {height:.0f}"
            lines += [f"### {title}", "", caption, ""]
            lines += [f"![{title}](gallery/{name}.{THEMES[theme][0]}.png){{ loading=lazy }}"
                      for theme in LIVE]
            lines += [""]
            # A bare name: raw HTML is not rewritten, and the published page
            # sits in the same folder as the gallery's files.
            lines += [f'<iframe src="{name}.{THEMES[theme][0]}.html" title="{title}" '
                      f'loading="lazy" style="aspect-ratio: {ratio}"></iframe>'
                      for theme in LIVE]
            lines += ["", "| " + " | ".join(f"`{theme}`" for theme in others) + " |",
                      "| " + " | ".join("---" for _ in others) + " |",
                      "| " + " | ".join(THEMES[theme][1] for theme in others) + " |",
                      "| " + " | ".join(f"![{title}, {theme}](gallery/{name}.{theme}.png)"
                                        "{ loading=lazy }" for theme in others) + " |", ""]

    page = directory.parent / "gallery.md"
    page.write_text("\n".join(lines), encoding="utf-8")

    return page


def main() -> int:
    """Render the gallery and its page."""
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    directory.mkdir(parents=True, exist_ok=True)

    results = build_results()
    calls = {**result_calls(results), **class_calls(results["prices"])}

    render_images(directory, calls)
    shapes = render_live(directory, calls)
    page = write_page(directory, shapes)

    print(f"Rendered {len(calls)} charts in {len(THEMES)} themes, and {len(calls) * len(LIVE)} "
          f"live versions, to {directory}; wrote {page}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
