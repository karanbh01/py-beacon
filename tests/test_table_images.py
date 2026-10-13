# tests/test_table_images.py
"""BN-302: image regression for Table, its three presets in the default theme
and a dark one, and its groups, boxes and heatmaps.

The baselines are drawn on CI by the baselines workflow, as every chart
image is: a pixel of drift fails (tests/test_plot_images.py says why).
"""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from beacon.plot import Table, reset
from beacon.tokens import DARK, LIGHT

TOLERANCE = 0.0

THEMES = [LIGHT, DARK]

DATA = pd.DataFrame(
    {"1Y": [8.42, 7.91, 6.10, -2.35, 11.2, 4.4],
     "3Y": [5.12, 4.80, 3.95, 0.40, 6.8, 2.2],
     "5Y": [7.30, 6.95, 5.50, 1.10, None, 3.9],
     "Volatility": [14.2, 13.8, 15.1, 18.4, 21.0, 9.5],
     "Sharpe": [0.61, 0.58, 0.44, -0.12, 0.52, 0.31],
     "AUM": [1250, 640, 480, 95, 2310, 760]},
    index=["Beacon Global", "MSCI World", "Peer median", "Peer low", "Growth fund",
           "Income fund"])

GROUPS = {"Return (%)": ["1Y", "3Y", "5Y"], "Risk": ["Volatility", "Sharpe"], "Size": ["AUM"]}

ROW_GROUPS = {"Ours": ["Beacon Global", "MSCI World"],
              "Peers": ["Peer median", "Peer low", "Growth fund", "Income fund"]}

FRAME = {"title": "Performance and risk",
         "subtitle": "To 31 December 2025, against the index and peers",
         "source": "Synthetic data", "notes": "Returns in GBP, net of fees.",
         "groups": GROUPS, "formats": {"Return (%)": "{:.1f}", "Risk": "{:.2f}"},
         "highlight": ["Beacon Global"]}


@pytest.fixture(autouse=True)
def fresh():
    """Each image starts from Beacon's style with no figures open."""
    reset()
    plt.close("all")
    yield
    reset()


@pytest.mark.parametrize("theme", THEMES)
class TestPresetImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_lines(self, theme):
        return Table(DATA, signed=["Return (%)"], style={"theme": theme}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_plain(self, theme):
        return Table(DATA, preset="plain", style={"theme": theme}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_boxed(self, theme):
        return Table(DATA, preset="boxed", style={"theme": theme}, **FRAME).figure


@pytest.mark.parametrize("theme", THEMES)
class TestGroupImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_row_group_headings(self, theme):
        return Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True,
                     style={"theme": theme}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_group_and_row_boxes(self, theme):
        return Table(DATA, row_groups=ROW_GROUPS, preset="boxed", negatives="brackets",
                     style={"theme": theme, "table_box": False, "table_group_boxes": True,
                            "table_row_boxes": True}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_stripes(self, theme):
        return Table(DATA, title="Striped", style={"theme": theme, "table_stripes": True,
                                                   "table_row_rules": False}).figure


@pytest.mark.parametrize("theme", THEMES)
class TestHeatmapImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_squares_faded(self, theme):
        return Table(DATA, heatmap={"group": "Return (%)", "center": 0, "fade": True},
                     style={"theme": theme}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_blended_both(self, theme):
        return Table(DATA, row_groups=ROW_GROUPS, row_group_headings=True,
                     heatmap={"group": "Return (%)", "shading": "blended", "blend": "both"},
                     style={"theme": theme}, **FRAME).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_one_column_without_values(self, theme):
        return Table(DATA, heatmap={"column": "Sharpe", "colormap": "blues", "values": False},
                     style={"theme": theme}, **FRAME).figure
