# tests/test_tables.py
"""BN-379: the periodic and trailing tables.

A generated fund and comparator over twenty-six months from mid-November
2023: the grid's months compound to its years and its years to the whole;
the differences are cell by cell; the tables lay out the figures they come
from.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import formulas, tables
from beacon.analysis.performance.analytics import PerformanceAnalytics

DAYS = pd.bdate_range("2023-11-15", "2026-01-30")
RANDOM = np.random.default_rng(41)
THEIRS = pd.Series(np.r_[0.0, RANDOM.normal(0.0004, 0.01, len(DAYS) - 1)], index=DAYS)
OURS = THEIRS * 0.95 + pd.Series(np.r_[0.0, RANDOM.normal(0.0001, 0.003, len(DAYS) - 1)],
                                 index=DAYS)


@pytest.fixture(scope="module")
def analytics():
    levels = 100.0 * (1 + THEIRS).cumprod()

    return PerformanceAnalytics(RunData.of_returns(OURS, 252, 0.0, {"index": levels}))


class TestTheGrid:

    def test_the_months_compound_to_the_year(self):
        grid = tables.monthly_grid(OURS)

        for year, row in grid.iterrows():
            months = row[tables.MONTHS].dropna()
            assert (1 + months).prod() - 1 == pytest.approx(row[tables.YEAR], rel=1e-12), year

    def test_the_years_compound_to_the_whole(self):
        grid = tables.monthly_grid(OURS)

        assert (1 + grid[tables.YEAR]).prod() - 1 == pytest.approx((1 + OURS).prod() - 1,
                                                                   rel=1e-12)

    def test_months_the_run_did_not_cover_are_blank(self):
        grid = tables.monthly_grid(OURS)

        assert list(grid.index) == ["2023", "2024", "2025", "2026"]
        assert grid.loc["2023", "Oct"] != grid.loc["2023", "Oct"]
        assert grid.loc["2026", "Feb"] != grid.loc["2026", "Feb"]
        assert grid.loc["2024", "Mar"] == pytest.approx((1 + OURS.loc["2024-03"]).prod() - 1)

    def test_the_difference_grid_is_cell_by_cell(self,
                                                 analytics):
        difference = analytics.monthly_grid("difference index")
        ours, theirs = analytics.monthly_grid(), analytics.monthly_grid("index")

        np.testing.assert_allclose(difference.to_numpy(dtype=float),
                                   (ours - theirs).to_numpy(dtype=float), rtol=1e-12)

    def test_a_comparator_the_run_lacks_is_refused(self,
                                                   analytics):
        with pytest.raises(KeyError):
            analytics.monthly_grid("benchmark")


class TestThePeriodTable:

    def test_each_year_s_figures_and_differences(self,
                                                 analytics):
        table = analytics.period_table("annual")
        year = OURS.loc["2024"]
        path = formulas.path_of(year)

        assert list(table.index) == ["2023", "2024", "2025", "2026"]
        assert table.loc["2024", ("fund", "total_return")] == pytest.approx(
            float(path.iloc[-1] - 1), rel=1e-12)
        assert table.loc["2024", ("fund", "volatility")] == pytest.approx(
            formulas.volatility(year, 252))
        assert table.loc["2024", ("difference index", "total_return")] == pytest.approx(
            table.loc["2024", ("fund", "total_return")]
            - table.loc["2024", ("index", "total_return")])

    def test_quarters(self,
                      analytics):
        table = analytics.period_table("quarterly")

        assert table.index[0] == "2023 Q4" and table.index[-1] == "2026 Q1"
        assert (1 + table[("fund", "total_return")]).prod() - 1 == pytest.approx(
            (1 + OURS).prod() - 1, rel=1e-12)


class TestTheTrailingTable:

    def test_it_lays_out_the_trailing_returns(self,
                                              analytics):
        table = analytics.trailing_table()
        trailing = analytics.trailing_returns()

        assert table.loc["1Y", ("fund", "total")] == trailing["1Y"].total
        assert table.loc["3Y", ("fund", "total")] != table.loc["3Y", ("fund", "total")]
        assert table.loc["1Y", ("difference index", "total")] == pytest.approx(
            table.loc["1Y", ("fund", "total")] - table.loc["1Y", ("index", "total")])

    def test_the_comparator_s_year_is_its_own_returns_over_it(self,
                                                              analytics):
        table = analytics.trailing_table()
        start = analytics.trailing_returns()["1Y"].start

        assert table.loc["1Y", ("index", "total")] == pytest.approx(
            (1 + THEIRS.loc[start:]).prod() - 1, rel=1e-12)


class TestMonthOfYear:

    def test_the_mean_of_each_calendar_month(self,
                                             analytics):
        averages = analytics.month_of_year()
        grid = analytics.monthly_grid()

        assert averages["Jan"] == pytest.approx(grid["Jan"].astype(float).mean())
        assert list(averages.index) == tables.MONTHS

    def test_no_difference_over_different_days(self):
        """A comparator that starts after the run has a shorter inception, so
        the two are not differenced."""
        later = 100.0 * (1 + THEIRS.loc["2024-06-03":]).cumprod()
        analytics = PerformanceAnalytics(RunData.of_returns(OURS, 252, 0.0, {"index": later}))

        table = analytics.trailing_table()

        assert table.loc["inception", ("index", "total")] == table.loc[
            "inception", ("index", "total")]
        assert table.loc["inception", ("difference index", "total")] != table.loc[
            "inception", ("difference index", "total")]
        assert table.loc["1Y", ("difference index", "total")] == table.loc[
            "1Y", ("difference index", "total")]
