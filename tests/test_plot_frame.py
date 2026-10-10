# tests/test_plot_frame.py
"""BN-289 and BN-291 to BN-294: the chart heading and legend, axes that end
on their ticks, chart sizes, themes and the correlation chart's options.

Reuses test_plot's results; these tests look at where things are drawn and in
what colour, which the image regression tests check only as pictures.
"""
import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, to_rgba
from matplotlib.patches import Rectangle

import test_plot
from beacon.plot import compare, style, use
from beacon.plot.frame import MARK_LABEL, NOTES_SIZE, ink
from beacon.plot.themes import THEMES
from beacon.testing import dataset
from beacon.tokens import DARK, LIGHT, colour

END = test_plot.END

# test_plot's results, as fixtures here.
index_result = test_plot.index_result
backtest = test_plot.backtest
attribution = test_plot.attribution
optimisation = test_plot.optimisation
risk_model = test_plot.risk_model
frontier = test_plot.frontier


@pytest.fixture(autouse=True)
def light_after():
    """Each test starts and ends in the light theme with no figures open."""
    use(LIGHT)
    yield
    use(LIGHT)
    plt.close("all")


def bar_of(figure) -> Rectangle:
    return next(artist for artist in figure.artists if isinstance(artist, Rectangle))


def mark_of(figure):
    return next(axes for axes in figure.get_axes() if axes.get_label() == MARK_LABEL)


def text_with(figure, gid: str):
    return next(text for text in figure.texts if text.get_gid() == gid)


def window(artist, figure):
    figure.canvas.draw()
    return artist.get_window_extent(figure.canvas.get_renderer())


class TestHeading:
    """BN-291."""

    def test_the_bar_starts_at_the_y_axis(self,
                                         index_result):
        ax = index_result.plot.level()

        assert bar_of(ax.figure).get_x() == pytest.approx(ax.get_position().x0)

    def test_the_mark_sits_left_of_the_bar_and_beside_the_heading(self,
                                                                  index_result):
        ax = index_result.plot.level()
        bar, mark = bar_of(ax.figure), mark_of(ax.figure).get_position()
        middle = (mark.y0 + mark.y1) / 2.0

        assert mark.x1 < bar.get_x()
        assert bar.get_y() < middle < bar.get_y() + bar.get_height()

    def test_the_subtitle_is_source_serif_at_350(self,
                                                  index_result):
        subtitle = text_with(index_result.plot.level().figure, "beacon-subtitle")

        assert subtitle.get_fontname() == "Source Serif 4"
        assert subtitle.get_fontweight() == 350
        assert subtitle.get_fontstyle() == "italic"

    def test_the_notes_are_small_and_italic(self,
                                            index_result):
        notes = text_with(index_result.plot.level().figure, "beacon-notes")

        assert notes.get_fontsize() == NOTES_SIZE
        assert notes.get_fontstyle() == "italic"


class TestLegend:
    """BN-291: the legend in the heading band."""

    def test_it_ends_where_the_x_axis_does_above_the_plot(self,
                                                          index_result):
        ax = index_result.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
        legend = window(ax.get_legend(), ax.figure)
        plot = window(ax, ax.figure)

        assert legend.x1 == pytest.approx(plot.x1, abs=2)
        assert legend.y0 > plot.y1

    def test_one_row_sits_level_with_the_subtitle(self,
                                                  index_result):
        ax = index_result.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
        legend = window(ax.get_legend(), ax.figure)
        bar_bottom = bar_of(ax.figure).get_y() * ax.figure.bbox.height

        assert legend.y0 == pytest.approx(bar_bottom, abs=2)

    def test_one_that_cannot_fit_beside_the_heading_goes_under_it(self,
                                                                  index_result,
                                                                  backtest):
        ax = compare(index_result, backtest,
                     labels=["An index with a very long name indeed",
                             "A backtest with an even longer name than that"],
                     title="A title long enough to fill the heading on its own")
        legend = window(ax.get_legend(), ax.figure)
        bar_bottom = bar_of(ax.figure).get_y() * ax.figure.bbox.height

        assert legend.y1 < bar_bottom


class TestAxes:
    """BN-292."""

    @pytest.mark.parametrize("chart, axis", [("level", "x"), ("level", "y"),
                                             ("contributions", "x"),
                                             ("frontier", "y"),
                                             ("exposures", "x")])
    def test_a_measuring_axis_ends_on_its_ticks(self,
                                                chart,
                                                axis,
                                                index_result,
                                                attribution,
                                                optimisation,
                                                frontier):
        draw = {"level": index_result.plot.level,
                "contributions": attribution.plot.contributions,
                "frontier": lambda: optimisation.plot.frontier(frontier),
                "exposures": optimisation.plot.exposures}[chart]
        ax = draw()
        scale = ax.xaxis if axis == "x" else ax.yaxis
        low, high = sorted(ax.get_xlim() if axis == "x" else ax.get_ylim())
        ticks = list(scale.get_majorticklocs())

        assert min(ticks) == pytest.approx(low)
        assert max(ticks) == pytest.approx(high)

    def test_the_corner_label_is_blank_where_both_axes_measure(self,
                                                               index_result):
        ax = index_result.plot.level()
        ax.figure.canvas.draw()
        labels = [label.get_text() for label in ax.get_yticklabels()]

        assert labels[0] == ""
        assert all(labels[1:])

    def test_an_axis_of_names_keeps_every_label(self,
                                                attribution):
        ax = attribution.plot.contributions()
        ax.figure.canvas.draw()

        assert all(label.get_text() for label in ax.get_yticklabels())

    def test_the_axes_are_in_the_titles_colour(self,
                                               index_result):
        ax = index_result.plot.level()

        assert to_hex(ax.spines["left"].get_edgecolor()) == colour("text-primary", LIGHT)

    def test_the_bar_charts_have_no_gridlines(self,
                                              index_result,
                                              attribution):
        for ax in (index_result.plot.weights(), attribution.plot.contributions()):
            assert not any(line.get_visible() for line in ax.get_xgridlines())


class TestSizes:
    """BN-293."""

    def test_a_chart_is_drawn_at_the_scale(self,
                                           index_result):
        width, height = index_result.plot.level().figure.get_size_inches()

        assert (width, height) == pytest.approx((6.0 * style.SCALE, 4.5 * style.SCALE))

    def test_the_performance_panel_matches_the_level_plot(self,
                                                          index_result,
                                                          backtest):
        level = index_result.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
        performance = backtest.plot.performance()

        def tall(ax):
            return ax.get_position().height * ax.figure.get_size_inches()[1]

        assert tall(performance) == pytest.approx(tall(level), abs=0.03)


class TestThemes:
    """BN-289."""

    def test_every_theme_is_a_registered_style(self):
        style.register()

        assert {theme.style for theme in THEMES.values()} <= set(plt.style.available)

    @pytest.mark.parametrize("name", [*THEMES, LIGHT, DARK])
    def test_use_takes_every_theme(self,
                                   name,
                                   index_result):
        use(name)

        assert index_result.plot.level() is not None

    def test_an_unknown_theme_is_refused(self):
        with pytest.raises(ValueError, match="theme must be one of"):
            use("sepia")

    def test_a_transparent_theme_draws_no_background(self,
                                                     index_result):
        use("transparent-light-axes")
        ax = index_result.plot.level()
        title = text_with(ax.figure, "beacon-title")

        assert to_rgba(ax.figure.get_facecolor())[3] == 0.0
        assert to_hex(title.get_color()) == colour("text-primary", DARK)

    def test_github_dark_uses_githubs_page_and_greys(self,
                                                     index_result):
        use("github-dark")
        ax = index_result.plot.level()

        assert to_hex(ax.figure.get_facecolor()) == "#0d1117"
        assert to_hex(text_with(ax.figure, "beacon-title").get_color()) == "#e6edf3"

    def test_a_chart_keeps_the_theme_it_was_drawn_in(self,
                                                     index_result):
        use("transparent-light-axes")
        ax = index_result.plot.level()
        use(LIGHT)

        assert ink(ax, "text-primary") == colour("text-primary", DARK)

    def test_palette_takes_a_theme(self):
        assert style.palette("white")["canvas"] == "#ffffff"
        assert style.palette("transparent-dark-axes")["canvas"] == "none"


class TestCorrelationShading:
    """BN-294."""

    @pytest.mark.parametrize("shading, interpolation", [("squares", "nearest"),
                                                        ("blended", "bicubic")])
    def test_each_shading(self,
                          shading,
                          interpolation,
                          risk_model):
        ax = risk_model.plot.correlation(shading=shading)

        assert ax.get_images()[0].get_interpolation() == interpolation

    def test_an_unknown_shading_is_refused(self,
                                           risk_model):
        with pytest.raises(ValueError, match="shading must be one of"):
            risk_model.plot.correlation(shading="dotted")

    def test_the_heatmap_has_no_tick_marks(self,
                                           risk_model):
        ax = risk_model.plot.correlation()

        assert all(tick.tick1line.get_markersize() == 0
                   for tick in ax.xaxis.get_major_ticks())


class TestReadableText:
    """BN-295."""

    def test_dates_are_the_year_and_the_month(self,
                                              index_result):
        ax = index_result.plot.level()
        ax.figure.canvas.draw()
        labels = [label.get_text() for label in ax.get_xticklabels()]

        assert labels[0] == "2023"
        assert "Apr" in labels

    @pytest.mark.parametrize("chart", ["performance", "correlation", "level"])
    def test_nothing_runs_past_the_figure(self,
                                          chart,
                                          index_result,
                                          backtest,
                                          risk_model):
        draw = {"performance": backtest.plot.performance,
                "correlation": risk_model.plot.correlation,
                "level": index_result.plot.level}[chart]
        figure = draw().figure
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()

        for axes in figure.get_axes():
            box = axes.get_tightbbox(renderer)
            assert box.x0 >= 0
            assert box.x1 <= figure.bbox.width

    def test_the_notes_are_in_the_labels_colour(self,
                                                index_result):
        notes = text_with(index_result.plot.level().figure, "beacon-notes")

        assert to_hex(notes.get_color()) == colour("text-primary", LIGHT)

    def test_the_legend_and_line_labels_share_a_size(self,
                                                     index_result):
        ax = index_result.plot.level(benchmark=dataset.prices()["CCC"].loc[:END])
        sizes = {text.get_fontsize() for text in ax.get_legend().get_texts()}
        value = next(text for text in ax.texts if text.get_text().replace(".", "").isdigit())

        assert sizes == {style.LINE_LABEL_SIZE}
        assert value.get_fontsize() == style.LINE_LABEL_SIZE
