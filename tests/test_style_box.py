# tests/test_style_box.py
"""BN-427: the style box, on a hand-made set of names.

Six index names of equal cap, two in each size band, and a fund holding
three of them and one name the index does not hold. Value and growth are set
so the style scores are known.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance.style import (
    BLEND,
    THREE_YEARS,
    StyleSettings,
    coverage,
    size_positions,
    style_box,
    style_classification,
    style_positions,
    style_scores,
)
from beacon.analysis.reports.catalogue import CATALOGUE

SETTINGS = StyleSettings()
PORTFOLIOS = ("fund", "index", "benchmark")


def caps(*values: float) -> pd.Series:
    return pd.Series(values, index=[f"N{position}" for position in range(len(values))])


class TestSize:

    def test_the_anchors(self):
        found = size_positions(caps(1e12, 10e9, 2e9, 1e8), SETTINGS)

        assert found.tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_within_a_band_by_log_cap(self):
        found = size_positions(caps(1e11, np.sqrt(10e9 * 2e9)), SETTINGS)

        assert found.tolist() == pytest.approx([0.5, 1.5])

    def test_it_is_not_clipped(self):
        """A name above $1tn sits above the top, one below $100m below the
        bottom."""
        found = size_positions(caps(4e12, 5e7), SETTINGS)

        assert found.iloc[0] < 0.0
        assert found.iloc[1] > 3.0

    def test_a_missing_or_empty_cap_has_no_position(self):
        found = size_positions(caps(np.nan, 0.0, -5.0), SETTINGS)

        assert found.isna().all()

    def test_the_bands_are_settings(self):
        found = size_positions(caps(5e9), StyleSettings(large_from=5e9, mid_from=1e9))

        assert found.iloc[0] == pytest.approx(1.0)


def snapshot() -> pd.DataFrame:
    """Two names a band, $500bn, $5bn and $500m, each pair a cheap
    slow-growing name and a dear fast-growing one; OUT is held by the fund
    only, and HOLE has no cap."""
    names = ["LV", "LG", "MV", "MG", "SV", "SG", "OUT", "HOLE"]
    cap = [5e11, 5e11, 5e9, 5e9, 5e8, 5e8, 5e9, np.nan]
    cheap = {"PE_RATIO": 8.0, "PB_RATIO": 1.0, "DIVIDEND_YIELD": 0.05,
             "EPS_GROWTH": 0.01, "SALES_GROWTH": 0.01,
             "EPS_GROWTH_3Y": 0.30, "SALES_GROWTH_3Y": 0.30}
    dear = {"PE_RATIO": 40.0, "PB_RATIO": 8.0, "DIVIDEND_YIELD": 0.0,
            "EPS_GROWTH": 0.30, "SALES_GROWTH": 0.25,
            "EPS_GROWTH_3Y": 0.01, "SALES_GROWTH_3Y": 0.01}
    rows = [cheap, dear] * 3 + [{}, cheap]
    frame = pd.DataFrame(rows, index=names)
    frame.insert(0, "MARKET_CAP", cap)

    return frame


INDEX = pd.Series(1 / 6, index=["LV", "LG", "MV", "MG", "SV", "SG"])
FUND = pd.Series({"LV": 0.3, "MG": 0.3, "SG": 0.2, "OUT": 0.1, "HOLE": 0.05})
WEIGHTS = {"fund": FUND, "index": INDEX, "benchmark": None}


@pytest.fixture(scope="module")
def positions() -> pd.DataFrame:
    return style_positions(snapshot(), WEIGHTS, SETTINGS).set_index("name")


class TestStyle:

    def test_growth_less_value(self):
        scores = style_scores(snapshot(), INDEX.index)

        assert scores["LG"] > 0 > scores["LV"]
        assert np.isnan(scores["OUT"])

    def test_the_percentile_is_within_the_size_band(self,
                                                    positions):
        """Two equal caps a band: the value name holds the band's lower half,
        so sits at a quarter of the way, the growth name at three quarters."""
        assert positions.loc["LV", "style"] == pytest.approx(0.75)
        assert positions.loc["LG", "style"] == pytest.approx(2.25)
        assert positions.loc["SV", "style"] == pytest.approx(0.75)

    def test_a_name_without_fields_is_blend(self,
                                            positions):
        assert positions.loc["OUT", "style"] == BLEND

    def test_the_growth_measure_is_a_setting(self):
        """Over three years the cheap names grew as much faster as they are
        cheaper, so the two cancel; over a year the dear names grew faster."""
        year = style_scores(snapshot(), INDEX.index)
        three = style_scores(snapshot(), INDEX.index, THREE_YEARS)

        assert year["LG"] - year["LV"] > 3.0
        assert three["LG"] - three["LV"] == pytest.approx(0.0, abs=1e-9)


class TestPlacing:

    def test_a_name_with_no_cap_is_left_out(self,
                                            positions):
        assert "HOLE" not in positions.index
        assert coverage(positions.reset_index(), WEIGHTS)["fund"] == pytest.approx(0.9 / 0.95)
        assert coverage(positions.reset_index(), WEIGHTS)["index"] == pytest.approx(1.0)

    def test_each_book_s_weights_with_none_for_a_book_without(self,
                                                              positions):
        assert positions.loc["LV", "fund"] == 0.3
        assert positions.loc["LG", "fund"] == 0.0
        assert positions["benchmark"].isna().all()

    def test_caps_are_put_in_dollars(self):
        """In a currency worth half a dollar, $5bn is 10bn: mid stays mid."""
        frame = snapshot()
        frame["MARKET_CAP"] *= 2.0
        found = style_positions(frame, WEIGHTS, SETTINGS, usd_rate=2.0).set_index("name")

        assert found.loc["MV", "size"] == pytest.approx(
            size_positions(caps(5e9), SETTINGS).iloc[0])


class TestTheBoxAndTheCentre:

    def test_each_book_sums_to_one(self,
                                   positions):
        box = style_box(positions.reset_index(), PORTFOLIOS)

        assert box["fund"].sum() == pytest.approx(1.0)
        assert box["index"].sum() == pytest.approx(1.0)
        assert box["benchmark"].isna().all()

    def test_the_index_holds_a_sixth_in_each_corner_and_middle_cell(self,
                                                                    positions):
        box = style_box(positions.reset_index(), PORTFOLIOS).set_index(["size", "style"])

        assert box.loc[("large", "value"), "index"] == pytest.approx(1 / 6)
        assert box.loc[("mid", "growth"), "index"] == pytest.approx(1 / 6)
        assert box.loc[("large", "blend"), "index"] == 0.0

    def test_the_fund_s_weights_are_over_its_placed_names(self,
                                                          positions):
        box = style_box(positions.reset_index(), PORTFOLIOS).set_index(["size", "style"])

        assert box.loc[("large", "value"), "fund"] == pytest.approx(0.3 / 0.9)
        assert box.loc[("mid", "blend"), "fund"] == pytest.approx(0.1 / 0.9)

    def test_the_centre_is_the_weighted_mean(self,
                                             positions):
        found = style_classification(positions.reset_index(), PORTFOLIOS).set_index("book")
        shares = positions["fund"] / positions["fund"].sum()

        assert found.loc["fund", "size_centre"] == pytest.approx(
            float((shares * positions["size"]).sum()))
        assert found.loc["index", "style_centre"] == pytest.approx(1.5)
        assert found.loc["index", "style"] == "blend"
        assert "benchmark" not in found.index

    def test_the_cell_is_the_centre_s(self,
                                      positions):
        found = style_classification(positions.reset_index(), PORTFOLIOS).set_index("book")

        assert found.loc["index", "size"] == ("large", "mid", "small")[
            int(np.floor(found.loc["index", "size_centre"]))]


class TestTheCatalogue:

    def test_three_tables_with_labelled_cells(self):
        for statistic_id in ("style_positions", "style_box", "style_classification"):
            assert CATALOGUE[statistic_id].kind == "table"

        box = {column.name: column for column in CATALOGUE["style_box"].columns}
        assert box["size"].labels["large"] == "Large"
        assert box["style"].labels["growth"] == "Growth"


class TestARunInAnotherCurrency:
    """The bands are in dollars: a pound run's caps are put in dollars at the
    rate it kept, and without one there is no box."""

    DAYS = pd.bdate_range("2024-06-03", periods=3)

    def run(self,
            **tables):
        from beacon.analysis.data import RunData

        returns = pd.Series(0.0, index=self.DAYS)
        frame = snapshot().rename_axis("IDENTIFIER").reset_index()
        frame.index = pd.DatetimeIndex([self.DAYS[0]] * len(frame))
        held = {"fund_weights": pd.DataFrame([FUND] * 3, index=self.DAYS),
                "index_weights": pd.DataFrame([INDEX] * 3, index=self.DAYS),
                "characteristics": frame, **tables}

        return RunData(currency="GBP", returns=returns, nav=returns + 1, units=returns + 1,
                       initial_capital=1.0, periods_per_year=252, risk_free_rate=0.0,
                       tables=held)

    def test_at_the_rate_the_run_kept(self):
        from beacon.analysis.reports.context import Context

        # A pound is 1.25 dollars: the mid names' 5bn pounds are 6.25bn dollars.
        rates = pd.DataFrame({"USD": [0.8] * 3}, index=self.DAYS)
        positions = CATALOGUE["style_positions"].compute(Context(self.run(rates=rates)))

        assert positions.set_index("name").loc["MV", "size"] == pytest.approx(
            size_positions(caps(5e9 * 1.25), SETTINGS).iloc[0])

    def test_without_a_rate_there_is_none(self):
        from beacon.analysis.reports.context import Context
        from beacon.analysis.reports.statistics import NO_DOLLAR_RATE, Unavailable

        with pytest.raises(Unavailable) as raised:
            CATALOGUE["style_box"].compute(Context(self.run()))

        assert raised.value.code == NO_DOLLAR_RATE
