# tests/test_plot_interactive.py
"""BN-301: interactive charts. Every chart class's plotly figure is drawn
from the same layout as its image: its size, each axis's domain, range,
ticks and labels, and the heading and legend are the layout's."""
import json
import subprocess
import sys

import pandas as pd
import pytest

pytest.importorskip("plotly")

import matplotlib

matplotlib.use("Agg")

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
    reset,
)
from beacon.plot.interactive.readout import date_text, value_text
from beacon.plot.interactive.units import DAY_MS, PX
from beacon.testing import dataset

END = "2024-12-31"


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return dataset.prices().loc[:END]


@pytest.fixture(autouse=True)
def fresh():
    reset()
    yield
    reset()


def rebased(prices, *names):
    chosen = prices[list(names)]
    return chosen / chosen.iloc[0] * 100.0


CHARTS = {
    "line": lambda p: Line(rebased(p, "AAA", "BBB"), title="Two", subtitle="Rebased",
                           ylabel="Level", reference=100, label_last=True, notes="Daily."),
    "secondary": lambda p: Line(p["AAA"].rename("Price"), secondary=p["DDD"].rename("Other"),
                                title="Two scales", ylabel="Price", ylabel_secondary="Other"),
    "area": lambda p: Area(p[["AAA", "BBB"]].div(p[["AAA", "BBB"]].sum(axis=1), axis=0),
                           title="Shares", percent=True),
    "column": lambda p: Column(p["CCC"].resample("ME").last().pct_change().dropna()
                               .loc["2024"].rename("CCC"), title="Monthly", percent=True),
    "bar": lambda p: Bar(pd.Series([0.08, 0.06, 0.05], index=["One", "Two", "Three"],
                                   name="Weight"), title="Weights", percent=True),
    "combo": lambda p: Combo(p["AAA"].diff().loc["2024"].rename("Flows"),
                             p["BBB"].loc["2024"].rename("Assets"), gap=0, title="Combo"),
    "scatter": lambda p: Scatter(p[["AAA", "DDD"]].pct_change().dropna(), x="DDD", y="AAA",
                                 fit="linear", percent=True, title="Sensitivity"),
    "origin": lambda p: Scatter(pd.DataFrame({"x": [-0.002, 0.003, 0.001],
                                              "y": [0.05, -0.04, 0.02]},
                                             index=["A", "B", "C"]),
                                x="x", y="y", labels=pd.Series(["A", "B", "C"]),
                                axes="origin", title="Origin"),
    "histogram": lambda p: Histogram(p["AAA"].pct_change().dropna(), bins=20, normal=True,
                                     title="Returns"),
    "heatmap": lambda p: Heatmap(p.pct_change().dropna().corr(), title="Correlation",
                                 colorbar=True),
    "grid": lambda p: Grid([[Line(rebased(p, "AAA"), title="Growth"),
                             Line(rebased(p, "BBB"), title="Other")]], title="Grid"),
}


def axis_names(index: int) -> tuple[str, str]:
    """The plotly layout keys of the index-th axis pair."""
    suffix = "" if index == 1 else str(index)
    return f"xaxis{suffix}", f"yaxis{suffix}"


@pytest.mark.parametrize("name", list(CHARTS))
class TestParity:

    def test_size(self, prices, name):
        chart = CHARTS[name](prices)
        figure, layout = chart.interactive(), chart.layout

        assert figure.layout.width == pytest.approx(layout.width * PX)
        assert figure.layout.height == pytest.approx(layout.height * PX)
        assert figure.layout.margin.l == 0 and figure.layout.margin.t == 0

    def test_axes_take_the_layout(self, prices, name):
        chart = CHARTS[name](prices)
        figure, layout = chart.interactive(), chart.layout
        count = 0

        for panel in layout.panels:
            count += 1
            x_key, y_key = axis_names(count)
            for laid, key, span in ((panel.x, x_key, (panel.box.left, panel.box.right,
                                                      layout.width)),
                                    (panel.y, y_key, (panel.box.bottom, panel.box.top,
                                                      layout.height))):
                axis = figure.layout[key]
                scale = DAY_MS if laid.kind == "date" else 1.0
                low, high = laid.low * scale, laid.high * scale
                expected = [high, low] if laid.inverted else [low, high]
                assert list(axis.range) == pytest.approx(expected)
                assert list(axis.domain) == pytest.approx([span[0] / span[2],
                                                           span[1] / span[2]])
                assert list(axis.tickvals) == pytest.approx(
                    [tick.value * scale for tick in laid.ticks])
                assert list(axis.ticktext) == [tick.label for tick in laid.ticks]
            if panel.y2 is not None:
                count += 1
                second = figure.layout[f"yaxis{count}"]
                assert list(second.range) == pytest.approx([panel.y2.low, panel.y2.high])
                assert second.overlaying == ("y" if count == 2 else f"y{count - 1}")
            if panel.colour_bar is not None:
                count += 1

    def test_tick_labels_are_the_layouts(self, prices, name):
        chart = CHARTS[name](prices)
        figure, layout = chart.interactive(), chart.layout
        written = [annotation.text for annotation in figure.layout.annotations]

        for panel in layout.panels:
            for laid in (panel.x, panel.y):
                if laid.show_labels:
                    for tick in laid.ticks:
                        if tick.label:
                            assert tick.label in written

    def test_heading_where_the_layout_puts_it(self, prices, name):
        chart = CHARTS[name](prices)
        figure, layout = chart.interactive(), chart.layout
        title = next(annotation for annotation in figure.layout.annotations
                     if annotation.name == "beacon-title")

        assert title.text == layout.title.text
        assert title.x == pytest.approx(layout.title.x / layout.width)
        assert title.y == pytest.approx(layout.title.y / layout.height)

    def test_legend_where_the_layout_puts_it(self, prices, name):
        chart = CHARTS[name](prices)
        figure, layout = chart.interactive(), chart.layout
        if layout.legend is None:
            pytest.skip("no legend")

        for entry in layout.legend.entries:
            placed = [annotation for annotation in figure.layout.annotations
                      if annotation.text == entry.label]
            assert any(annotation.x == pytest.approx(entry.text.x / layout.width)
                       and annotation.y == pytest.approx(entry.text.y / layout.height)
                       for annotation in placed)


class TestFrame:

    def test_tick_marks_include_the_ends(self, prices):
        chart = CHARTS["line"](prices)
        figure, laid = chart.interactive(), chart.layout.panels[0]
        ticks = [shape for shape in figure.layout.shapes
                 if shape.type == "line" and shape.xref == "x" and shape.x0 == shape.x1]

        assert sorted(shape.x0 for shape in ticks) == pytest.approx(
            sorted(tick.value * DAY_MS for tick in laid.x.ticks))

    def test_mark_and_accent_bar(self, prices):
        figure = CHARTS["line"](prices).interactive()
        kinds = [(shape.type, shape.name) for shape in figure.layout.shapes]

        assert ("path", "beacon-mark") in kinds
        assert any(kind == "rect" for kind, _ in kinds)

    def test_second_axis_marks_drawn_on_the_main_axis(self, prices):
        figure = CHARTS["secondary"](prices).interactive()

        assert {trace.yaxis for trace in figure.data} == {"y"}

    def test_heatmap_has_no_axis_lines(self, prices):
        figure = CHARTS["heatmap"](prices).interactive()
        lines = [shape for shape in figure.layout.shapes
                 if shape.type == "line" and "domain" in str(shape.xref)
                 and "domain" in str(shape.yref)]

        assert lines == []


class TestPopup:

    def test_line_readout(self, prices):
        figure = CHARTS["line"](prices).interactive()
        readout = next(trace for trace in figure.data
                       if trace.meta and trace.meta["beacon"]["kind"] == "line"
                       and trace.name == "AAA")
        heading, plain, rows = readout.customdata[0]

        assert heading == date_text(prices.index[0].timestamp() / 86400.0)
        assert json.loads(rows)[0][1] == "AAA"
        assert plain.startswith("AAA: ")

    def test_rows_follow_the_legend(self, prices):
        figure = CHARTS["line"](prices).interactive()
        orders = {trace.name: trace.meta["beacon"]["order"] for trace in figure.data
                  if trace.meta}

        assert orders["AAA"] < orders["BBB"]

    def test_hover_modes(self, prices):
        assert CHARTS["line"](prices).interactive().layout.hovermode == "x unified"
        assert CHARTS["scatter"](prices).interactive().layout.hovermode == "closest"
        assert CHARTS["bar"](prices).interactive().layout.hovermode == "y unified"

    def test_glass_by_theme(self, prices):
        light = Line(rebased(prices, "AAA"), style={"theme": "light"}).interactive()
        white = Line(rebased(prices, "AAA"), style={"theme": "white"}).interactive()

        assert light.layout.meta["beacon"]["glass"].startswith("rgba(233,217,198")
        assert white.layout.meta["beacon"]["glass"] == "rgba(255,255,255,0.65)"

    def test_drawn_traces_take_no_part_in_hovering(self, prices):
        figure = CHARTS["line"](prices).interactive()

        assert all(trace.hoverinfo == "skip" for trace in figure.data if not trace.meta)


class TestReadout:

    def test_date(self):
        assert date_text(0.0) == "1 Jan 1970"

    def test_values(self):
        assert value_text(0.1234, "share") == "12.3%"
        assert value_text(-0.1234, "percent") == "\N{MINUS SIGN}12.34%"
        assert value_text(24.0, "number", whole=True) == "24"
        assert value_text(1234.5, "number") == "1,234"
        assert value_text(1.5, lambda value: f"\N{POUND SIGN}{value:.1f}m") == "\N{POUND SIGN}1.5m"
        assert value_text(float("nan"), "number") == "\N{EN DASH}"


class TestHtml:

    def test_fonts_plotly_and_popup(self, prices):
        html = CHARTS["line"](prices).interactive().to_html()

        assert "@font-face{font-family:'Inter Tight'" in html
        assert "@font-face{font-family:'Source Serif 4'" in html
        assert "beacon-popup" in html
        assert "Plotly.newPlot" in html and len(html) > 3_000_000

    def test_write_html_with_plotly_beside(self, prices, tmp_path):
        figure = CHARTS["line"](prices).interactive()
        figure.write_html(tmp_path / "chart.html", include_plotlyjs="directory")

        assert (tmp_path / "plotly.min.js").exists()
        assert "beacon-popup" in (tmp_path / "chart.html").read_text(encoding="utf-8")

    def test_table_has_no_figure(self, prices):
        with pytest.raises(TypeError, match="to_html"):
            Table(prices.iloc[:3, :2]).interactive()


def test_needs_the_extra():
    """Without plotly a chart is still built and drawn; asking for its
    interactive figure names the extra to install."""
    script = """
import sys

class Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] == "plotly":
            raise ImportError("blocked")
        return None

sys.meta_path.insert(0, Blocker())
import pandas as pd
from beacon.exceptions import MissingDependencyError
from beacon.plot import Line
chart = Line(pd.Series([1.0, 2.0, 3.0], name="A"), title="T")
chart.layout
try:
    chart.interactive()
except MissingDependencyError as error:
    print("plot-interactive" in str(error))
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                               check=False)

    assert completed.stdout.strip() == "True", completed.stderr
