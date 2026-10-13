# tests/test_plot_grid.py
"""BN-300: several charts in one figure, read off the grid's layout."""
import numpy as np
import pandas as pd
import pytest

from beacon.plot import Bar, Column, Grid, Line, reset
from beacon.plot.charts.stack import Stacked

DAYS = pd.bdate_range("2024-01-02", "2024-12-31")
RNG = np.random.default_rng(300)
LEVELS = pd.DataFrame(100 * np.cumprod(1 + RNG.normal(0.0004, 0.01, (len(DAYS), 2)), axis=0),
                      index=DAYS, columns=["Fund", "Benchmark"])


@pytest.fixture(autouse=True)
def fresh():
    reset()
    yield
    reset()


def two_by_two(**options) -> Grid:
    late = LEVELS.loc["2024-07-01":]

    return Grid([[Line(LEVELS["Fund"], title="Fund"), Line(LEVELS["Benchmark"], title="Index")],
                 [Line(late["Fund"], title="Since July"), Column(pd.Series([0.1, -0.05],
                                                                           index=["A", "B"]),
                                                                 title="Returns")]],
                title="Four panels", **options)


class TestPlacement:

    def test_every_plot_is_the_same_size(self):
        boxes = [panel.box for panel in two_by_two().layout.panels]

        assert len({round(box.width, 6) for box in boxes}) == 1
        assert len({round(box.height, 6) for box in boxes}) == 1

    def test_rows_and_columns_line_up(self):
        boxes = [panel.box for panel in two_by_two().layout.panels]

        assert boxes[0].top == pytest.approx(boxes[1].top)
        assert boxes[0].left == pytest.approx(boxes[2].left)
        assert boxes[0].right < boxes[1].left
        assert boxes[0].bottom > boxes[2].top

    def test_each_panel_is_titled_above_it(self):
        panels = two_by_two().layout.panels

        assert [panel.title.text for panel in panels] == ["Fund", "Index", "Since July",
                                                          "Returns"]
        assert all(panel.title.y >= panel.box.top for panel in panels)

    def test_the_titles_clear_the_heading(self):
        laid = two_by_two().layout

        assert max(panel.title.y for panel in laid.panels) < laid.accent_bar.bottom

    def test_the_frame_lines_up_with_the_outer_panels(self):
        laid = two_by_two().layout
        left = min(panel.box.left for panel in laid.panels)

        assert laid.accent_bar.left == pytest.approx(left)
        assert laid.notes is None or laid.notes.x == pytest.approx(left)

    def test_an_empty_cell(self):
        grid = Grid([[Line(LEVELS["Fund"], title="Fund"), None],
                     [None, Line(LEVELS["Benchmark"], title="Index")]], title="Diagonal")
        boxes = [panel.box for panel in grid.layout.panels]

        assert len(boxes) == 2
        assert boxes[1].left > boxes[0].right and boxes[1].top < boxes[0].bottom

    def test_the_axes_end_on_their_ticks(self):
        for panel in two_by_two().layout.panels[:3]:
            assert panel.x.ended and panel.y.ended


class TestSharing:

    def test_one_calendar_when_asked(self):
        panels = two_by_two(sharex=True).layout.panels

        assert panels[2].x.low == pytest.approx(panels[0].x.low)
        assert panels[2].x.high == pytest.approx(panels[0].x.high)

    def test_each_its_own_dates_otherwise(self):
        panels = two_by_two().layout.panels

        assert panels[2].x.low > panels[0].x.low

    def test_one_legend_when_the_panels_share_series(self):
        grid = Grid([[Line(LEVELS, title="All of 2024"),
                      Line(LEVELS.loc["2024-07-01":], title="Since July")]], title="Shared")

        assert [entry.label for entry in grid.layout.legend.entries] == ["Fund", "Benchmark"]

    def test_no_legend_when_they_do_not(self):
        other = LEVELS.rename(columns={"Benchmark": "Peer"})
        grid = Grid([[Line(LEVELS, title="One"), Line(other, title="Two")]], title="Unshared")

        assert grid.layout.legend is None


class TestRefusals:

    def test_an_empty_grid(self):
        with pytest.raises(ValueError, match="at least one chart"):
            Grid([[None]])

    def test_a_chart_of_several_panels(self):
        stack = Stacked([Line(LEVELS["Fund"]), Bar(pd.Series([1.0], index=["A"]))], [3, 1])

        with pytest.raises(ValueError, match="one panel each"):
            _ = Grid([[stack]]).plan


class TestNames:

    MONTHS = pd.Series(np.linspace(-0.02, 0.03, 12),
                       index=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                              "Oct", "Nov", "Dec"])

    def test_names_that_fit_stay_level(self):
        assert Column(self.MONTHS, title="One").layout.panels[0].x.rotation == 0.0

    def test_names_that_do_not_fit_turn_until_they_do(self):
        grid = Grid([[Column(self.MONTHS, title=name) for name in "ABC"]], title="Three")

        assert {panel.x.rotation for panel in grid.layout.panels} == {90.0}
