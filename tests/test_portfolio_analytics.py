# tests/test_portfolio_analytics.py
"""BN-382: what a run held and in which groups, on a small hand-made book.

Three names over six days, rebalanced on the first and fourth: BBB moves
from Tech to Energy at the second rebalance, CCC has no sector recorded, and
the fund holds some cash, so its group weights add up to its invested weight.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.portfolio import UNKNOWN, PortfolioAnalytics

DAYS = pd.bdate_range("2024-03-04", periods=6)
FIRST, SECOND = DAYS[0], DAYS[3]

FUND = pd.DataFrame({"AAA": [0.5, 0.5, 0.4, 0.3, 0.3, 0.3],
                     "BBB": [0.3, 0.3, 0.4, 0.5, 0.5, 0.5],
                     "CCC": [0.1, 0.1, 0.1, np.nan, np.nan, np.nan]}, index=DAYS)
INDEX = pd.DataFrame({"AAA": [0.6] * 6, "BBB": [0.4] * 6}, index=DAYS)
REFERENCE = pd.DataFrame({"IDENTIFIER": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
                          "SECTOR": ["Tech", "Tech", None, "Tech", "Energy", None],
                          "REGION": ["US"] * 6},
                         index=pd.DatetimeIndex([FIRST] * 3 + [SECOND] * 3, name="DATE"))


def run(**tables) -> RunData:
    returns = pd.Series(0.0, index=DAYS)

    return RunData(currency="USD", returns=returns, nav=returns + 1, units=returns + 1,
                   initial_capital=1.0, periods_per_year=252, risk_free_rate=0.0,
                   tables={"fund_weights": FUND, "index_weights": INDEX,
                           "reference": REFERENCE, **tables})


@pytest.fixture
def portfolio():
    return PortfolioAnalytics(run())


class TestHoldings:

    def test_the_number_of_names_held_each_day(self,
                                               portfolio):
        assert portfolio.holdings_count().tolist() == [3, 3, 3, 2, 2, 2]
        assert portfolio.holdings_count("index").tolist() == [2] * 6

    def test_a_book_without_weights_has_none(self,
                                             portfolio):
        assert portfolio.holdings_count("benchmark") is None
        assert portfolio.group_weights("SECTOR", "benchmark") is None


class TestGroupWeights:

    def test_each_day_s_groups_are_its_latest_rebalance_s(self,
                                                          portfolio):
        sectors = portfolio.group_weights("SECTOR")

        assert list(sectors.columns) == sorted(["Energy", "Tech", UNKNOWN])
        assert sectors.loc[DAYS[2], "Tech"] == pytest.approx(0.8)
        assert sectors.loc[DAYS[2], "Energy"] == 0.0
        assert sectors.loc[DAYS[4], "Energy"] == pytest.approx(0.5)
        assert sectors.loc[DAYS[4], "Tech"] == pytest.approx(0.3)

    def test_a_name_with_no_group_is_unknown(self,
                                             portfolio):
        assert portfolio.group_weights("SECTOR").loc[FIRST, UNKNOWN] == pytest.approx(0.1)

    def test_group_weights_add_up_to_the_invested_weight(self,
                                                         portfolio):
        sectors = portfolio.group_weights("SECTOR")
        invested = FUND.fillna(0.0).sum(axis=1)

        np.testing.assert_allclose(sectors.sum(axis=1), invested, rtol=1e-12)
        assert invested.iloc[0] == pytest.approx(0.9)

    def test_any_grouping_field(self,
                                portfolio):
        regions = portfolio.group_weights("REGION")

        assert list(regions.columns) == ["US"]

    def test_a_field_the_run_did_not_keep_is_refused(self,
                                                     portfolio):
        with pytest.raises(KeyError, match="INDUSTRY"):
            portfolio.group_weights("INDUSTRY")


class TestActiveGroupWeights:

    def test_the_fund_s_less_the_index_s(self,
                                         portfolio):
        active = portfolio.active_group_weights("SECTOR")

        assert active.loc[DAYS[2], "Tech"] == pytest.approx(0.8 - 1.0)
        assert active.loc[DAYS[4], "Energy"] == pytest.approx(0.5 - 0.4)
        assert active.loc[DAYS[4], "Tech"] == pytest.approx(0.3 - 0.6)
        assert active.loc[FIRST, UNKNOWN] == pytest.approx(0.1)

    def test_against_a_benchmark_without_names_there_is_none(self,
                                                              portfolio):
        assert portfolio.active_group_weights("SECTOR", "benchmark") is None

    def test_against_a_benchmark_with_names(self):
        benchmark = pd.DataFrame({"BBB": [1.0] * 6}, index=DAYS)

        active = PortfolioAnalytics(run(benchmark_weights=benchmark)).active_group_weights(
            "SECTOR", "benchmark")

        assert active.loc[FIRST, "Tech"] == pytest.approx(0.8 - 1.0)
