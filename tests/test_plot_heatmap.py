# tests/test_plot_heatmap.py
"""BN-299: the heatmap and its colour maps, read off its plan and layout."""
import numpy as np
import pandas as pd
import pytest

from beacon.plot import Heatmap, reset
from beacon.plot.colormaps import colour_at, ink_on, stops
from beacon.plot.marks import CellsMark, TextMark
from beacon.testing import dataset

CORRELATION = dataset.prices().pct_change().dropna().corr()

MONTHLY = pd.DataFrame(np.random.default_rng(299).normal(0.006, 0.03, (3, 12)),
                       index=[2023, 2024, 2025],
                       columns=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
                                "Oct", "Nov", "Dec"])


@pytest.fixture(autouse=True)
def fresh():
    reset()
    yield
    reset()


def cells_of(chart: Heatmap) -> CellsMark:
    return next(mark for mark in chart.plan.marks if isinstance(mark, CellsMark))


class TestColourMaps:

    def test_a_map_and_its_reverse(self):
        assert stops("blues")[0] == "#eaf1fa"
        assert stops("blues_r") == tuple(reversed(stops("blues")))

    def test_colours_of_your_own(self):
        assert stops(["#000000", "#ffffff"]) == ("#000000", "#ffffff")

    def test_an_unknown_map_is_refused(self):
        with pytest.raises(ValueError, match="colormap must be one of"):
            stops("rainbow")

    def test_blending_along_a_map(self):
        assert colour_at(("#000000", "#ffffff"), 0.5) == "#808080"
        assert colour_at(("#000000", "#ffffff"), 2.0) == "#ffffff"

    def test_text_reads_on_any_cell(self):
        assert ink_on("#ffffff") != ink_on("#000000")


class TestHeatmap:

    def test_values_either_side_of_zero_diverge_about_it(self):
        cells = cells_of(Heatmap(MONTHLY))

        assert cells.low == -cells.high
        assert cells.stops == stops("blue-red")

    def test_values_one_way_take_a_one_way_map(self):
        cells = cells_of(Heatmap(MONTHLY.abs()))

        assert cells.stops == stops("blues")
        assert cells.low == pytest.approx(MONTHLY.abs().to_numpy().min())

    def test_the_lower_triangle(self):
        cells = cells_of(Heatmap(CORRELATION, lower=True)).values

        assert cells[0][1] is None and cells[1][0] is not None

    def test_values_are_written_unless_asked_not_to(self):
        written = [mark for mark in Heatmap(CORRELATION, decimals=1).plan.marks
                   if isinstance(mark, TextMark)]

        assert len(written) == CORRELATION.size
        assert written[0].text == "1.0"
        assert not [mark for mark in Heatmap(CORRELATION, values=False).plan.marks
                    if isinstance(mark, TextMark)]

    def test_the_colour_bar_is_labelled_at_its_ends(self):
        panel = Heatmap(CORRELATION, colormap="beacon_corr", vmin=0.2, vmax=1.0).layout.panels[0]

        assert [tick.label for tick in panel.colour_bar.axis.ticks] == ["0.20", "1.00"]
        assert Heatmap(CORRELATION, colorbar=False).layout.panels[0].colour_bar is None

    def test_rows_run_down_and_columns_slant(self):
        panel = Heatmap(CORRELATION).layout.panels[0]

        assert panel.y.inverted and panel.y.ticks[0].label == "AAA"
        assert panel.x.rotation == 45.0

    def test_the_frame_is_cropped_to_the_cells(self):
        chart = Heatmap(CORRELATION, title="Correlation")
        laid = chart.layout
        full = chart.plan.description.size

        assert laid.width < full[0]
        assert laid.panels[0].box.width == pytest.approx(laid.panels[0].box.height)

    def test_an_unknown_shading_is_refused(self):
        with pytest.raises(ValueError, match="shading must be"):
            Heatmap(CORRELATION, shading="dotted")
