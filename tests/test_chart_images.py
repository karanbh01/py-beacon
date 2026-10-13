# tests/test_chart_images.py
"""BN-299: image regression for the chart classes, in the default theme and
a dark one.

The baselines are drawn on CI by the baselines workflow, as every chart
image is: a pixel of drift fails (tests/test_plot_images.py says why).
"""
import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from beacon.plot import Area, Bar, Column, Combo, Heatmap, Histogram, Line, Scatter, reset
from beacon.testing import dataset
from beacon.tokens import DARK, LIGHT

TOLERANCE = 0.0

THEMES = [LIGHT, DARK]

END = "2024-12-31"


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """The canonical dataset's closes, two years of them."""
    return dataset.prices().loc[:END]


@pytest.fixture(autouse=True)
def fresh():
    """Each image starts from Beacon's style with no figures open."""
    reset()
    plt.close("all")
    yield
    reset()


def rebased(prices: pd.DataFrame,
            *names: str) -> pd.DataFrame:
    chosen = prices[list(names)]

    return chosen / chosen.iloc[0] * 100.0


@pytest.mark.parametrize("theme", THEMES)
class TestLineImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_line(self, prices, theme):
        return Line(rebased(prices, "AAA", "BBB", "CCC"),
                    title="Three holdings", subtitle="Each rebased to 100",
                    ylabel="Level", reference=100, label_last=True,
                    source="Synthetic data", notes="Daily closes.",
                    style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_line_periods(self, prices, theme):
        return Line(rebased(prices, "AAA", "BBB"),
                    title="Marked periods", subtitle="Two periods, each labelled at its top",
                    reference=100, periods={"First half": ("2023-03-01", "2023-06-30"),
                                            "Autumn": ("2024-09-01", "2024-11-29")},
                    style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_line_secondary(self, prices, theme):
        return Line(prices["AAA"].rename("Price"), secondary=prices["DDD"].rename("Other"),
                    title="Two scales", subtitle="A second series on the right",
                    ylabel="Price", ylabel_secondary="Other", label_last=["Price", "Other"],
                    style={"theme": theme}).figure


@pytest.mark.parametrize("theme", THEMES)
class TestAreaImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_area_filled(self, prices, theme):
        level = prices["AAA"]
        drawdown = (level / level.cummax() - 1).rename("Drawdown")

        return Area(drawdown, title="Drawdown", subtitle="AAA against its running peak",
                    ylabel="Drawdown", percent=True, colors=["danger"],
                    style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_area_stacked(self, prices, theme):
        monthly = prices[["AAA", "BBB", "CCC", "DDD"]].resample("MS").first()
        shares = monthly.div(monthly.sum(axis=1), axis=0)

        return Area(shares, title="Shares", subtitle="Each holding's share of the four",
                    ylabel="Share", percent=True, ylim=(0, 1),
                    style={"theme": theme}).figure


@pytest.mark.parametrize("theme", THEMES)
class TestBarImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_column_grouped(self, prices, theme):
        yearly = prices[["AAA", "BBB"]].resample("YE").last()
        returns = yearly.pct_change().dropna()
        returns.index = returns.index.year

        return Column(returns, title="Calendar-year returns", subtitle="Two holdings by year",
                      ylabel="Return", percent=True, value_labels=True,
                      style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_column_signed(self, prices, theme):
        monthly = prices["CCC"].resample("ME").last().pct_change().dropna().loc["2024"]
        monthly.index = monthly.index.strftime("%b")

        return Column(monthly, title="Monthly returns", subtitle="CCC in 2024",
                      ylabel="Return", percent=True, style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_bar_capped(self, theme):
        weights = pd.Series([0.0792, 0.0741, 0.0688, 0.0617, 0.0583, 0.0526],
                            index=["Northwind Systems", "Halcyon Health", "Arden Industrials",
                                   "Meridian Software", "Corvid Semiconductors",
                                   "Larkspur Pharma"])

        return Bar(weights, title="Largest holdings", subtitle="Capped at 8%",
                   xlabel="Weight", percent=True, decimals=2, cap=0.08, value_labels=True,
                   signed=False, color="accent", style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_combo_daily(self, prices, theme):
        returns = prices["AAA"].pct_change().dropna().loc["2024"]
        flows = (returns * 40).rename("Net flows")
        assets = (300 + flows.cumsum()).rename("Assets")

        return Combo(flows, assets, gap=0, title="Daily flows and assets",
                     subtitle="Each column runs to the next trading day",
                     ylabel="Net flows", ylabel_secondary="Assets",
                     style={"theme": theme}).figure


@pytest.mark.parametrize("theme", THEMES)
class TestPointImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_scatter_fitted(self, prices, theme):
        returns = prices[["AAA", "DDD"]].pct_change().dropna().loc["2024"]

        return Scatter(returns, x="DDD", y="AAA", fit="linear", percent=True,
                       title="Sensitivity", subtitle="AAA's daily returns against DDD's, 2024",
                       style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_scatter_origin(self, prices, theme):
        returns = prices.pct_change().dropna().loc["2024"]
        active = returns.std() - returns.std().mean()
        excess = returns.mean() * 252 - (returns.mean() * 252).mean()
        bets = pd.DataFrame({"Active weight": active, "Excess return": excess})

        return Scatter(bets, x="Active weight", y="Excess return", labels=bets.index.to_series(),
                       axes="origin", signed=True, percent=True,
                       quadrants=("Over, ahead", "Under, ahead", "Under, behind",
                                  "Over, behind"),
                       title="Did the bets pay off?", subtitle="Each holding, 2024",
                       style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_histogram(self, prices, theme):
        returns = prices["AAA"].pct_change().dropna().rename("Daily return")

        return Histogram(returns, bins=40, normal=True,
                         mark={"95% VaR": float(np.quantile(returns, 0.05))}, percent=True,
                         title="Daily returns", subtitle="How often AAA moved by each amount",
                         style={"theme": theme}).figure


@pytest.mark.parametrize("theme", THEMES)
class TestHeatmapImages:

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_heatmap_correlation(self, prices, theme):
        correlation = prices.pct_change().dropna().corr()

        return Heatmap(correlation, title="Correlation", subtitle="Between daily returns",
                       colormap="beacon_corr", vmin=0.2, vmax=1.0,
                       style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_heatmap_blended(self, prices, theme):
        correlation = prices.pct_change().dropna().corr()

        return Heatmap(correlation, title="Correlation", subtitle="Blended, lower triangle",
                       colormap="beacon_corr", vmin=0.2, vmax=1.0, values=False, lower=True,
                       shading="blended", style={"theme": theme}).figure

    @pytest.mark.mpl_image_compare(tolerance=TOLERANCE)
    def test_heatmap_monthly(self, prices, theme):
        monthly = prices["AAA"].resample("ME").last().pct_change().dropna()
        table = pd.DataFrame({"return": monthly, "year": monthly.index.year,
                              "month": monthly.index.strftime("%b")})
        grid = table.pivot(index="year", columns="month", values="return")
        grid = grid[[month for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                                         "Aug", "Sep", "Oct", "Nov", "Dec"] if month in grid]]

        return Heatmap(grid, title="Monthly returns", subtitle="AAA, by month",
                       percent=True, decimals=1, style={"theme": theme}).figure
