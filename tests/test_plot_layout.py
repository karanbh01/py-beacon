# tests/test_plot_layout.py
"""BN-298: the chart layout, worked out once without a renderer.

The tick rules are ports of matplotlib's, so they are checked against
matplotlib itself on many ranges; text is measured from the font files and
checked against matplotlib's text paths; the layout is built with matplotlib
blocked; and a drawn chart is checked to be the layout it was drawn from.
"""
import datetime
import json
import random
import subprocess
import sys
import textwrap

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator, PercentFormatter, ScalarFormatter

import test_plot
from beacon.plot import dateticks, design, metrics, style, ticks, use
from beacon.plot.arrange import MARK_ASPECT, lay_out, wrap
from beacon.plot.frame import frame_text
from beacon.plot.layout import Axis, Description, LegendItem, Panel
from beacon.plot.mark import beta_outline
from beacon.plot.render import describe
from beacon.tokens import LIGHT

index_result = test_plot.index_result
backtest = test_plot.backtest
risk_model = test_plot.risk_model

FIGURE = (4.8, 3.6)


@pytest.fixture(autouse=True)
def light_after():
    """Each test starts and ends in the light theme with no figures open."""
    use(LIGHT)
    yield
    use(LIGHT)
    plt.close("all")


def chart(**changes):
    """A two-series level chart over 2024, described by hand."""
    start = dateticks.to_number(datetime.datetime(2024, 1, 2))
    end = dateticks.to_number(datetime.datetime(2024, 12, 31))
    settings = {
        "size": FIGURE,
        "title": "Index level",
        "subtitle": "CANON, rebased to 100",
        "notes": "Index dated from 02/01/2024 to 31/12/2024. Rebased to 100 at the "
                 "first observation, and drawn against its benchmark rebased the same.",
        "panels": (Panel(x=Axis("date", start, end),
                         y=Axis("number", 96.3, 143.1, title="Level")),),
        "legend": (LegendItem("Index", "accent"), LegendItem("Benchmark", "text-secondary")),
    }

    return Description(**{**settings, **changes})


class TestNumberTicks:
    """The number ticks and labels are matplotlib's, on ranges of every
    scale."""

    def test_they_match_matplotlib(self):
        rng = random.Random(298)
        figure = plt.figure(figsize=FIGURE)

        for _ in range(400):
            tall = rng.uniform(0.5, 3.0)
            ax = figure.add_axes((0.1, 0.1, 0.8, tall / FIGURE[1]))
            ax.tick_params(labelsize=6.24)
            spacing, locator = ticks.Spacing(), ax.yaxis.get_major_locator()
            if rng.random() < 0.3:
                spacing = ticks.Spacing(bins=4)
                locator = MaxNLocator(nbins=4, steps=list(ticks.STEPS))
                ax.yaxis.set_major_locator(locator)
            scale = 10 ** rng.uniform(-4, 7)
            low = rng.uniform(-1, 1) * scale
            high = low + rng.uniform(0.001, 2) * scale
            room = ticks.room_for(tall, 6.24, across=False)

            with matplotlib.rc_context({"axes.autolimit_mode": "round_numbers"}):
                expected_ends = tuple(float(v) for v in locator.view_limits(low, high))
            assert ticks.ends(low, high, spacing, room) == expected_ends

            ax.set_ylim(*expected_ends)
            expected = [float(v) for v in locator()]
            assert ticks.tick_values(*expected_ends, spacing, room) == expected

            plain = ScalarFormatter()
            ax.yaxis.set_major_formatter(plain)
            written = ticks.number_labels(expected, *expected_ends)
            assert written.labels == tuple(plain.format_ticks(expected))
            assert written.offset == plain.get_offset()

            percent = PercentFormatter(1.0)
            ax.yaxis.set_major_formatter(percent)
            assert (ticks.percent_labels(expected, *expected_ends).labels
                    == tuple(percent.format_ticks(expected)))
            ax.remove()


class TestDateTicks:
    """The date ticks and labels are matplotlib's concise dates."""

    def test_they_match_matplotlib(self):
        rng = random.Random(298)

        for _ in range(400):
            start = datetime.datetime(1990, 1, 1) + datetime.timedelta(days=rng.uniform(0, 12000))
            stop = start + datetime.timedelta(days=10 ** rng.uniform(0.7, 4.3))
            low, high = mdates.date2num(start), mdates.date2num(stop)
            locator = mdates.AutoDateLocator()
            expected = [float(v) for v in locator.tick_values(start, stop)]
            found = dateticks.tick_values(low, high)

            assert found == pytest.approx(expected, abs=1e-9)

            formatter = mdates.ConciseDateFormatter(locator)
            written = dateticks.labels(found)
            assert written.labels == tuple(formatter.format_ticks(expected))
            assert written.offset == formatter.get_offset()

    def test_a_year_ends_on_its_january_ticks(self):
        low, high, ended = dateticks.ends(dateticks.to_number(datetime.datetime(2023, 1, 3)),
                                          dateticks.to_number(datetime.datetime(2024, 12, 31)))

        assert ended
        assert dateticks.to_date(low) == datetime.datetime(2023, 1, 1)
        assert dateticks.to_date(high) == datetime.datetime(2025, 1, 1)


class TestMetrics:
    """Text is measured from the font files as matplotlib sets it."""

    @pytest.mark.parametrize("text, size, family, weight, italic", [
        ("Index", 4.4, "Inter Tight", "light", False),
        ("Benchmark \N{MINUS SIGN}12.5%", 6.24, "Inter Tight", "normal", False),
        ("CANON, rebased to 100", 6.3, "Source Serif 4", 350, True),
        ("Notes: Source: Synthetic data.", 3.6, "Inter Tight", "light", True),
        ("1,234.5", 4.4, "Inter Tight", "bold", False),
    ])
    def test_widths_match_matplotlibs_text_paths(self,
                                                 text,
                                                 size,
                                                 family,
                                                 weight,
                                                 italic):
        style.register()
        font = FontProperties(family=family, weight=weight,
                              style="italic" if italic else "normal")
        expected = TextPath((0, 0), text, size=size, prop=font).get_extents().width / 72.0
        found = metrics.width(text, size, metrics.Font(family, metrics.weight_of(weight), italic))

        assert found == pytest.approx(expected, abs=1e-4)

    def test_a_font_with_no_file_is_measured_as_inter_tight(self, caplog):
        found = metrics.width("Level", 8.0, metrics.Font("No Such Sans"))

        assert found == metrics.width("Level", 8.0, metrics.Font("Inter Tight"))
        assert "not bundled" in caplog.text

    def test_the_marks_proportions(self):
        extent = beta_outline().get_extents()

        assert pytest.approx(extent.width / extent.height) == MARK_ASPECT


class TestLayout:
    """The rules a layout follows, read off the value itself."""

    def test_the_corner_label_is_blank_and_the_axes_end_on_ticks(self):
        laid = lay_out(chart(), design.Style())
        panel = laid.panels[0]

        assert panel.y.ticks[0].label == ""
        assert all(tick.label for tick in panel.y.ticks[1:])
        for axis in (panel.x, panel.y):
            assert axis.ended
            assert axis.ticks[0].value == pytest.approx(axis.low)
            assert axis.ticks[-1].value == pytest.approx(axis.high)

    def test_the_corner_label_stays_when_asked(self):
        laid = lay_out(chart(), design.Style().updated(blank_lowest_label=False))

        assert all(tick.label for tick in laid.panels[0].y.ticks)

    def test_the_notes_fit_the_plots_width(self):
        laid = lay_out(chart(), design.Style())
        box, notes = laid.panels[0].box, laid.notes

        assert notes is not None
        assert notes.text.startswith("Notes: ")
        for line in notes.text.split("\n"):
            assert metrics.width(line, notes.size, notes.font) <= box.width

    def test_the_legend_ends_where_the_x_axis_does(self):
        laid = lay_out(chart(), design.Style())

        assert laid.legend is not None
        assert laid.legend.corner == "lower right"
        assert laid.legend.anchor[0] == pytest.approx(laid.panels[0].box.right)
        assert laid.legend.box.right == pytest.approx(laid.panels[0].box.right)

    def test_colours_are_names(self):
        laid = lay_out(chart(), design.Style())

        assert laid.title.colour == "text-primary"
        assert [entry.colour for entry in laid.legend.entries] == ["accent", "text-secondary"]

    def test_the_heading_sits_beside_the_plot(self):
        laid = lay_out(chart(), design.Style())
        left = laid.panels[0].box.left

        assert laid.accent_bar.left == pytest.approx(left)
        assert laid.mark.right < left
        assert laid.title.x > laid.accent_bar.right

    def test_nothing_is_left_outside_the_figure(self):
        laid = lay_out(chart(notes=""), design.Style())
        title = laid.panels[0].y.title

        assert title is not None
        assert title.x - metrics.box(title.text, title.size, title.font).height >= 0.2 - 1e-9

    def test_no_legend_when_the_style_has_none(self):
        assert lay_out(chart(), design.Style().updated(legend="none")).legend is None

    def test_wrapping_keeps_a_long_word_whole(self):
        lines = wrap("a " + "x" * 80 + " b", 0.5, 4.5, metrics.Font("Inter Tight", 300))

        assert lines == ["a", "x" * 80, "b"]


class TestWithoutMatplotlib:
    """The plotly renderer lays charts out without matplotlib."""

    def test_a_layout_needs_no_matplotlib(self):
        script = textwrap.dedent("""
            import importlib.abc, json, sys

            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name == "matplotlib" or name.startswith("matplotlib."):
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Block())

            from beacon.plot.arrange import lay_out
            from beacon.plot.design import Style
            from beacon.plot.layout import Axis, Description, LegendItem, Panel

            laid = lay_out(Description(size=(4.8, 3.6), title="Level", notes="A note.",
                                 panels=(Panel(Axis("date", 19724.0, 20088.0),
                                               Axis("number", 96.3, 143.1)),),
                                 legend=(LegendItem("Index", "accent"),)),
                           Style())
            print(json.dumps({"matplotlib": "matplotlib" in sys.modules,
                              "labels": [t.label for t in laid.panels[0].y.ticks]}))
        """)
        result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                                text=True, check=True)
        found = json.loads(result.stdout)

        assert found["matplotlib"] is False
        assert found["labels"][0] == ""
        assert found["labels"][1:]


class TestRenderer:
    """A drawn chart is the layout it was drawn from."""

    @pytest.mark.parametrize("chart_name", ["level", "performance", "correlation"])
    def test_the_figure_is_where_the_layout_puts_it(self,
                                                    chart_name,
                                                    index_result,
                                                    backtest,
                                                    risk_model):
        draw = {"level": index_result.plot.level,
                "performance": backtest.plot.performance,
                "correlation": risk_model.plot.correlation}[chart_name]
        ax = draw()
        figure = ax.figure
        figure.canvas.draw()
        described, panels = describe(ax, frame_text(ax, "title"), frame_text(ax, "subtitle"),
                                     frame_text(ax, "notes"),
                                     0.5 * style.SCALE if chart_name == "correlation" else None)
        laid = lay_out(described, design.current())
        width, height = figure.get_size_inches()

        for axes, panel in zip(panels, laid.panels, strict=True):
            box = axes.get_position()
            assert (box.x0 * width, box.y0 * height) == pytest.approx(
                (panel.box.left, panel.box.bottom), abs=1e-9)
            drawn = [label.get_text() for label in axes.get_yticklabels()]
            assert drawn == [tick.label for tick in panel.y.ticks]
