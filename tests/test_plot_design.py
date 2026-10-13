# tests/test_plot_design.py
"""BN-303: every part of the chart design as a setting.

`Style` holds the settings, `configure()` sets them for every chart drawn
afterwards, `settings()` for one `with` block, and a chart's `style=` for that
chart alone, the most specific winning. Beacon's design is the default.
"""
import subprocess
import sys

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.plot import Style, compare, configure, current, reset, settings, style, use
from beacon.plot.frame import MARK_LABEL, REFERENCE_ID
from beacon.plot.themes import SERIES
from beacon.testing import dataset
from beacon.tokens import LIGHT

END = "2024-06-28"


@pytest.fixture(autouse=True)
def _fresh():
    """Every test starts in Beacon's design and the light theme, with no figures."""
    plt.close("all")
    reset()
    use(LIGHT)
    yield
    plt.close("all")
    reset()
    use(LIGHT)


@pytest.fixture(scope="module")
def index_result():
    definition = IndexDefinition(
        index_id="CANON", index_name="Canonical", base_date=dataset.START,
        base_value=1000.0, currency="USD", eligibility_rules=[],
        weighting_scheme=MarketCapWeighted(), rebalancing_frequency="QUARTERLY",
        calendar="XNYS", universe_identifiers=list(dataset.UNIVERSE))

    return IndexCalculator(definition, dataset.data_fetcher()).run(start_date=dataset.START,
                                                                   end_date=END)


def mark(figure) -> list:
    return [ax for ax in figure.get_axes() if ax.get_label() == MARK_LABEL]


def by_gid(figure, gid: str) -> list:
    return [artist for artist in figure.findobj()
            if getattr(artist, "get_gid", lambda: None)() == gid]


class TestTheStyleValue:

    def test_the_default_is_todays_design(self):
        default = Style()

        assert default == Style.beacon()
        assert default.theme == "beacon-light"
        assert (default.heading.title_size, default.heading.subtitle_size) == (9.0, 6.3)
        assert default.axes.tick_label_size == style.TICK_LABEL_SIZE
        assert default.axes.title_size == style.AXIS_TITLE_SIZE
        assert default.text.label_size == style.LINE_LABEL_SIZE
        assert default.lines.series_width * style.SCALE == style.SERIES_WIDTH

    def test_plain_drops_the_mark_and_keeps_the_rest(self):
        plain = Style.plain()

        assert plain.theme == "white"
        assert plain.heading.mark is False
        assert plain.heading.accent_bar is True
        assert plain.heading.title_font == "Inter Tight"

    def test_replace_changes_settings_by_group(self):
        changed = Style().replace(heading={"mark": False}, lines={"series_width": 2.0})

        assert changed.heading.mark is False
        assert changed.lines.series_width == 2.0
        assert Style().heading.mark is True

    def test_updated_takes_short_names(self):
        changed = Style().updated(mark=False, heading_scale=1.3, legend="bottom", theme="dark")

        assert (changed.heading.mark, changed.heading.scale) == (False, 1.3)
        assert changed.legend.position == "bottom"
        assert changed.theme == "beacon-dark"

    @pytest.mark.parametrize(("change", "message"), [
        ({"marks": False}, "'marks' is not a chart setting"),
        ({"legend": "top"}, "legend.position must be one of"),
        ({"aspect": "wide"}, "size.aspect must be width:height"),
        ({"up_color": "green"}, "colors.up must be a hex colour"),
        ({"scale": 0}, "size.scale must be a number above zero"),
        ({"mark": "no"}, "heading.mark must be True or False"),
        ({"theme": "neon"}, "theme must be one of"),
    ])
    def test_a_bad_setting_says_what_is_allowed(self, change, message):
        with pytest.raises(ValueError, match=message):
            Style().updated(**change)

    def test_a_style_saves_and_loads(self, tmp_path):
        house = Style.plain().updated(series_colors=["#112233", "#445566"], aspect="16:9",
                                      notes_size=5.0, grid="y")

        assert Style.load(house.save(tmp_path / "house.toml")) == house

    def test_a_broken_file_is_refused(self, tmp_path):
        (tmp_path / "bad.toml").write_text("theme = \n", encoding="utf-8")

        with pytest.raises(ValueError, match="not valid TOML"):
            Style.load(tmp_path / "bad.toml")

    def test_a_file_with_an_unknown_group_is_refused(self, tmp_path):
        (tmp_path / "bad.toml").write_text("[sizes]\nscale = 1.2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="no 'sizes' group"):
            Style.load(tmp_path / "bad.toml")


class TestChoosingAStyle:

    def test_configure_builds_on_the_current_style(self):
        configure(mark=False)
        configure(scale=1.2)

        assert current().heading.mark is False
        assert current().size.scale == 1.2

    def test_configure_takes_a_style_a_preset_or_a_file(self, tmp_path):
        assert configure(Style.plain()) == Style.plain()
        assert configure(preset="beacon") == Style()
        assert configure(Style.plain().save(tmp_path / "p.toml")) == Style.plain()

        with pytest.raises(TypeError, match="style or a preset"):
            configure(Style(), preset="plain")

    def test_settings_lasts_for_its_block(self):
        configure(mark=False)

        with settings(theme="white", legend="none") as inside:
            assert inside.theme == "white"
            assert inside.heading.mark is False

        assert current().theme == "beacon-light"
        assert current().legend.position == "heading"
        assert current().heading.mark is False

    def test_use_changes_the_theme_and_nothing_else(self):
        configure(mark=False, scale=1.1)
        use("github-dark")

        assert current().theme == "github-dark"
        assert (current().heading.mark, current().size.scale) == (False, 1.1)

    def test_reset_returns_to_beacons_design(self):
        configure(mark=False, theme="white")

        assert reset() == Style()

    def test_importing_beacon_plot_does_not_import_matplotlib(self):
        code = ("import sys\nimport beacon.plot\nfrom beacon.plot import Style, configure\n"
                "configure(mark=False)\nprint('matplotlib' in sys.modules)")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True)

        assert out.stdout.strip() == "False"


class TestTheFrame:

    def test_the_mark_and_accent_bar_can_be_turned_off(self, index_result):
        on = index_result.plot.level().get_figure()
        off = index_result.plot.level(style={"mark": False, "accent_bar": False}).get_figure()

        assert mark(on) and by_gid(on, "beacon-accent-bar")
        assert not mark(off) and not by_gid(off, "beacon-accent-bar")

    def test_the_heading_scales_on_its_own(self, index_result):
        figure = index_result.plot.level(style={"heading_scale": 1.5}).get_figure()
        title = by_gid(figure, "beacon-title")[0]

        assert title.get_fontsize() == pytest.approx(13.5)
        assert tuple(figure.get_size_inches()) == pytest.approx(style.figure_size("level"))

    def test_scale_zooms_the_figure(self, index_result):
        figure = index_result.plot.level(style={"scale": 1.25}).get_figure()

        assert figure.get_size_inches()[0] == pytest.approx(6.0 * 0.8 * 1.25)

    def test_width_and_aspect_set_the_size(self, index_result):
        figure = index_result.plot.level(style={"width": 8.0, "aspect": "16:9"}).get_figure()
        width, height = figure.get_size_inches()

        assert width == pytest.approx(8.0 * 0.8)
        assert width / height == pytest.approx(16 / 9)

    def test_the_notes_can_be_turned_off_or_restyled(self, index_result):
        hidden = index_result.plot.level(style={"notes": False}).get_figure()
        upright = index_result.plot.level(style={"notes_italic": False,
                                                 "notes_size": 6.0}).get_figure()
        note = by_gid(upright, "beacon-notes")[0]

        assert not by_gid(hidden, "beacon-notes")
        assert note.get_fontstyle() == "normal"
        assert note.get_fontsize() == pytest.approx(6.0)

    def test_a_legend_can_go_under_the_plot_or_nowhere(self, compared):
        ax = compared(style={"legend": "bottom"})
        legend_box = ax.get_legend().get_window_extent()
        plot_box = ax.get_window_extent()

        assert legend_box.y1 < plot_box.y0
        assert compared(style={"legend": "none"}).get_legend() is None

    def test_a_legend_can_go_beside_the_plot(self, compared):
        ax = compared(style={"legend": "right"})

        assert ax.get_legend().get_window_extent().x0 > ax.get_window_extent().x1


class TestTheAxes:

    def test_axis_lines_and_ticks_can_be_turned_off(self, index_result):
        ax = index_result.plot.level(style={"x_axis_line": False, "y_ticks": False})

        assert not ax.spines["bottom"].get_visible()
        assert ax.spines["left"].get_visible()
        assert all(tick.tick1line.get_markersize() == 0 for tick in ax.yaxis.get_major_ticks())

    def test_gridlines_follow_the_style(self, index_result):
        ax = index_result.plot.level(style={"grid": "y"})

        assert any(line.get_visible() for line in ax.get_ygridlines())
        assert not any(line.get_visible() for line in ax.get_xgridlines())

    def test_reference_lines_can_be_left_out(self, index_result):
        ax = index_result.plot.level(style={"reference_lines": False})

        assert not [line for line in ax.get_lines() if line.get_gid() == REFERENCE_ID]

    def test_the_lowest_label_is_blank_unless_asked(self, compared):
        blank = compared()
        kept = compared(style={"blank_lowest_label": False})

        def lowest(ax):
            ax.get_figure().canvas.draw()
            return ax.yaxis.get_major_formatter()(min(ax.get_ylim()), 0)

        assert lowest(blank) == ""
        assert lowest(kept) != ""


class TestColours:

    def test_series_take_the_new_palette(self, compared):
        ax = compared()
        colours = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()
                   if line.get_gid() != REFERENCE_ID]

        assert colours[:2] == list(SERIES[LIGHT][:2])

    def test_series_colours_can_be_replaced(self, compared):
        ax = compared(style={"series_colors": ["#112233", "#445566"]})
        colours = [matplotlib.colors.to_hex(line.get_color()) for line in ax.get_lines()
                   if line.get_gid() != REFERENCE_ID]

        assert colours[:2] == ["#112233", "#445566"]

    def test_the_accent_bar_colour_can_be_set(self, index_result):
        figure = index_result.plot.level(style={"accent_bar_color": "#ff0000"}).get_figure()

        bar = by_gid(figure, "beacon-accent-bar")[0]

        assert matplotlib.colors.to_hex(bar.get_facecolor()) == "#ff0000"


class TestPrecedence:

    def test_a_charts_own_style_is_its_alone(self, index_result):
        configure(mark=False)
        before = dict(matplotlib.rcParams)

        drawn = index_result.plot.level(style={"mark": True, "scale": 1.3}).get_figure()

        assert mark(drawn)
        assert current().heading.mark is False
        assert current().size.scale == 1.0
        assert matplotlib.rcParams["lines.linewidth"] == before["lines.linewidth"]

    def test_settings_beats_configure_and_style_beats_settings(self, index_result):
        configure(mark=False)

        with settings(mark=True):
            assert mark(index_result.plot.level().get_figure())
            assert not mark(index_result.plot.level(style={"mark": False}).get_figure())

        assert not mark(index_result.plot.level().get_figure())

    def test_configure_takes_effect_at_the_next_chart(self, index_result):
        configure(series_width=2.0)
        ax = index_result.plot.level()
        index_line = next(line for line in ax.get_lines() if line.get_gid() != REFERENCE_ID)

        assert index_line.get_linewidth() == pytest.approx(2.0 * 0.8)


@pytest.fixture
def compared(index_result):
    """A compare chart of the index against a copy of itself, drawn on demand."""
    def draw(style=None):
        return compare(index_result, index_result, labels=["One", "Two"], style=style)

    return draw


def test_levels_are_unchanged_by_a_style(index_result):
    """A style changes the drawing, never the numbers drawn."""
    plain = index_result.plot.level()
    zoomed = index_result.plot.level(style={"scale": 1.4, "theme": "white"})

    def first(ax):
        return next(line for line in ax.get_lines() if line.get_gid() != REFERENCE_ID)

    assert list(first(plain).get_ydata()) == list(first(zoomed).get_ydata())
