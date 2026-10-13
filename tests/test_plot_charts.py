# tests/test_plot_charts.py
"""BN-299: the chart classes, read off their plans and layouts.

What a chart draws (its marks, axes and legend) is checked on the values a
renderer draws from, and drawing itself only where it matters: that a chart
is drawn lazily, saved in each format, and shown in a notebook.
"""
import json
import subprocess
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from beacon.plot import Area, Chart, Line, configure, reset
from beacon.plot.charts.data import limited
from beacon.plot.marks import AreaMark, LineMark, PointMark, RuleMark, SpanMark, TextMark

DAYS = pd.bdate_range("2024-01-02", "2024-12-31")
RNG = np.random.default_rng(299)
LEVELS = pd.DataFrame(100 * np.cumprod(1 + RNG.normal(0.0004, 0.01, (len(DAYS), 2)), axis=0),
                      index=DAYS, columns=["Fund", "Benchmark"])


@pytest.fixture(autouse=True)
def fresh():
    """Each test starts in Beacon's style with no figures open."""
    reset()
    yield
    reset()
    plt.close("all")


def marks_of(chart: Chart, kind: type) -> list:
    return [mark for mark in chart.plan.marks if isinstance(mark, kind)]


class TestData:

    @pytest.mark.parametrize("given, expected", [
        (None, (90.0, 160.0)),
        ((0.0, 200.0), (0.0, 200.0)),
        ((200.0, 0.0), (0.0, 200.0)),
        ((0.0, None), (0.0, 160.0)),
        ((None, 300.0), (90.0, 300.0)),
        ((200.0, None), (200.0, 270.0)),
        ((None, 50.0), (-20.0, 50.0)),
    ])
    def test_limits(self, given, expected):
        """Swapped limits are put right; a single limit beyond the data keeps
        the data's span on the automatic side."""
        assert limited(90.0, 160.0, given) == expected

    def test_empty_data_is_refused(self):
        with pytest.raises(ValueError, match="nothing to draw"):
            Line(pd.Series([], dtype=float))

    def test_other_data_is_refused(self):
        with pytest.raises(TypeError, match="Series or DataFrame"):
            Line([1, 2, 3])


class TestLine:

    def test_the_first_series_leads(self):
        chart = Line(LEVELS)
        lines = marks_of(chart, LineMark)

        assert [line.colour for line in lines] == ["palette-2", "palette-1"]
        assert lines[-1].width > lines[0].width

    def test_one_series_has_no_legend(self):
        assert Line(LEVELS["Fund"]).layout.legend is None
        assert [entry.label for entry in Line(LEVELS).layout.legend.entries] == [
            "Fund", "Benchmark"]

    def test_a_reference_line(self):
        rules = marks_of(Line(LEVELS, reference=100), RuleMark)

        assert [(rule.value, rule.colour) for rule in rules] == [(100.0, "border")]
        assert not marks_of(Line(LEVELS, reference=100, style={"reference_lines": False}),
                            RuleMark)

    def test_the_last_value_is_dotted_and_labelled(self):
        chart = Line(LEVELS, label_last=True)
        last = float(LEVELS["Fund"].iloc[-1])

        assert [point.y for point in marks_of(chart, PointMark)] == [(last,)]
        assert [text.text for text in marks_of(chart, TextMark)] == [f"{last:,.1f}"]

    def test_marked_periods(self):
        chart = Line(LEVELS, periods={"Drawdown": ("2024-03-01", "2024-05-01")},
                     period_legend="Drawdowns")
        spans = marks_of(chart, SpanMark)

        assert len(spans) == 1 and spans[0].colour == "period"
        assert [text.text for text in marks_of(chart, TextMark)] == ["Drawdown"]
        assert chart.layout.legend.entries[-1].key == "period"

    def test_period_labels_can_be_left_off(self):
        chart = Line(LEVELS, periods=[("2024-03-01", "2024-05-01", "Drawdown")],
                     period_labels=False)

        assert not marks_of(chart, TextMark)

    def test_a_secondary_axis(self):
        chart = Line(LEVELS["Fund"], secondary=LEVELS["Benchmark"] * 10,
                     ylabel_secondary="Assets")
        panel = chart.layout.panels[0]

        assert panel.y2 is not None and panel.y2.title.text == "Assets"
        assert panel.y2.low <= 10 * LEVELS["Benchmark"].min()
        assert [line.side for line in marks_of(chart, LineMark)] == ["left", "right"]
        assert chart.layout.legend.entries[-1].label == "Benchmark (right)"

    def test_percentages(self):
        chart = Line(LEVELS / 100 - 1, percent=True)

        assert all(tick.label.endswith("%") for tick in chart.layout.panels[0].y.ticks
                   if tick.label)

    def test_limits_move_out_to_ticks(self):
        panel = Line(LEVELS, ylim=(0, 175)).layout.panels[0]

        assert panel.y.low == 0.0
        assert panel.y.high >= 175.0

    def test_an_inverted_axis(self):
        assert Line(LEVELS, invert=True).layout.panels[0].y.inverted


class TestArea:

    def test_stacked_parts(self):
        shares = pd.DataFrame({"A": [0.5, 0.4], "B": [0.5, 0.6]},
                              index=pd.to_datetime(["2024-01-01", "2024-02-01"]))
        areas = marks_of(Area(shares, percent=True), AreaMark)

        assert [area.upper for area in areas] == [(0.5, 0.4), (1.0, 1.0)]

    def test_one_filled_series(self):
        drawdown = LEVELS["Fund"] / LEVELS["Fund"].cummax() - 1
        chart = Area(drawdown.rename("Drawdown"), colors=["danger"])

        assert [mark.colour for mark in chart.plan.marks] == ["danger", "danger"]
        assert chart.layout.legend is None

    def test_the_axis_runs_from_the_first_value_to_the_last(self):
        panel = Area(LEVELS).layout.panels[0]

        assert not panel.x.ended
        assert panel.x.low == pytest.approx(Area(LEVELS).plan.description.panels[0].x.low)


class TestDrawing:

    def test_building_a_chart_needs_no_matplotlib(self):
        script = textwrap.dedent("""
            import importlib.abc, json, sys

            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == "matplotlib" or name.startswith("matplotlib."):
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Block())

            import pandas as pd
            from beacon.plot import Line

            chart = Line(pd.Series([1.0, 2.0, 3.0], index=pd.bdate_range("2024-01-01", periods=3)),
                         title="Level", reference=1.0, label_last=True)
            laid = chart.layout
            print(json.dumps({"matplotlib": "matplotlib" in sys.modules,
                              "ticks": len(laid.panels[0].y.ticks)}))
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, check=True)
        found = json.loads(result.stdout)

        assert found["matplotlib"] is False
        assert found["ticks"] > 1

    def test_a_chart_draws_only_when_used(self):
        chart = Line(LEVELS, title="Level")

        assert chart._drawn is None
        assert chart.ax.figure is chart.figure
        assert chart._drawn is not None

    def test_the_style_is_fixed_when_it_is_built(self):
        chart = Line(LEVELS)
        configure(mark=False)

        assert chart.layout.mark is not None

    @pytest.mark.parametrize("suffix", [".png", ".svg", ".pdf"])
    def test_it_saves_in_each_format(self, tmp_path, suffix):
        target = Line(LEVELS, title="Level").save(tmp_path / f"chart{suffix}")

        assert target.exists() and target.stat().st_size > 1000

    def test_a_notebook_shows_it(self):
        data, metadata = Line(LEVELS, title="Level")._repr_png_()

        assert data.startswith(b"\x89PNG")
        assert metadata["width"] > 400

    def test_the_drawn_ticks_are_the_layouts(self):
        chart = Line(LEVELS, title="Level")
        drawn = [label.get_text() for label in chart.ax.get_yticklabels()]

        assert drawn == [tick.label for tick in chart.layout.panels[0].y.ticks]
