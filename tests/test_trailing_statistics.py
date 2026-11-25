# tests/test_trailing_statistics.py
"""BN-422: the main figures over each trailing window.

A generated fund and index over a little more than five years to the end of
January 2026: each window's figures are the whole-run formulas on its days,
its returns agree with the trailing returns table, the short windows give no
Sharpe ratio or annualised return (BN-465) and the windows the run is too
short for give nothing.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import formulas
from beacon.analysis.performance.analytics import PerformanceAnalytics
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.performance.trailing_statistics import COLUMNS
from beacon.analysis.reports.catalogue import CATALOGUE

DAYS = pd.bdate_range("2020-11-16", "2026-01-30")
RANDOM = np.random.default_rng(41)
THEIRS = pd.Series(np.r_[0.0, RANDOM.normal(0.0004, 0.01, len(DAYS) - 1)], index=DAYS)
OURS = THEIRS * 0.95 + pd.Series(np.r_[0.0, RANDOM.normal(0.0001, 0.003, len(DAYS) - 1)],
                                 index=DAYS)
RISK_FREE = 0.02

# The windows shorter than six months: the run ends in January.
SHORT = ("1M", "3M", "MTD", "QTD", "YTD")


def analytics_of(levels: pd.Series,
                 settings: AnalysisSettings | None = None) -> PerformanceAnalytics:
    return PerformanceAnalytics(RunData.of_returns(OURS, 252, RISK_FREE, {"index": levels}),
                                settings)


@pytest.fixture(scope="module")
def table():
    return analytics_of(100.0 * (1 + THEIRS).cumprod()).trailing_statistics()


def cell(table: pd.DataFrame,
         period: str,
         statistic: str,
         book: str):
    found = table[(table["period"] == period) & (table["statistic"] == statistic)]

    assert len(found) == 1
    return found.iloc[0][book]


def blank(value) -> bool:
    return value is None or value != value


class TestTheShape:

    def test_a_row_a_window_and_figure(self,
                                       table):
        assert tuple(table.columns) == COLUMNS
        assert list(table["period"].unique()) == ["1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y",
                                                  "MTD", "QTD", "YTD", "inception"]
        assert len(table) == 11 * 7

    def test_each_row_says_its_unit(self,
                                    table):
        units = dict(zip(table["statistic"], table["unit"], strict=False))

        assert units["sharpe_ratio"] == "ratio"
        assert units["volatility"] == "fraction"

    def test_a_benchmark_the_run_lacks_is_blank(self,
                                                table):
        assert table["benchmark"].map(blank).all()

    def test_the_fund_has_no_figures_against_itself(self,
                                                    table):
        against = table[table["statistic"].isin(["tracking_difference", "tracking_error"])]

        assert against["fund"].map(blank).all()


class TestTheFigures:

    def test_the_returns_are_the_trailing_returns(self,
                                                  table):
        analytics = analytics_of(100.0 * (1 + THEIRS).cumprod())
        trailing = analytics.trailing_table()

        for window in ("1M", "1Y", "3Y", "YTD", "inception"):
            assert cell(table, window, "total_return", "fund") == pytest.approx(
                trailing.loc[window, ("fund", "total")], rel=1e-12), window
            assert cell(table, window, "total_return", "index") == pytest.approx(
                trailing.loc[window, ("index", "total")], rel=1e-12), window

    def test_a_window_s_figures_are_the_formulas_on_its_days(self,
                                                             table):
        start = analytics_of(100.0 * (1 + THEIRS).cumprod()).trailing_returns()["1Y"].start
        days = OURS.loc[start:]

        assert cell(table, "1Y", "volatility", "fund") == pytest.approx(
            formulas.volatility(days, 252), rel=1e-12)
        assert cell(table, "1Y", "sharpe_ratio", "fund") == pytest.approx(
            formulas.sharpe_ratio(days, 252, RISK_FREE), rel=1e-12)
        assert cell(table, "1Y", "max_drawdown", "fund") == pytest.approx(
            formulas.max_drawdown(formulas.path_of(days)), rel=1e-12)

    def test_tracking_is_against_the_comparator_over_the_same_days(self,
                                                                    table):
        start = analytics_of(100.0 * (1 + THEIRS).cumprod()).trailing_returns()["3Y"].start
        active = OURS.loc[start:] - THEIRS.loc[start:]

        assert cell(table, "3Y", "tracking_error", "index") == pytest.approx(
            formulas.tracking_error(active, 252), rel=1e-9)
        assert cell(table, "3Y", "tracking_difference", "index") == pytest.approx(
            cell(table, "3Y", "total_return", "fund")
            - cell(table, "3Y", "total_return", "index"), rel=1e-12)

    def test_inception_is_the_headline(self,
                                       table):
        headline = analytics_of(100.0 * (1 + THEIRS).cumprod()).headline()

        for name in ("total_return", "annualised_return", "volatility", "max_drawdown",
                     "sharpe_ratio"):
            assert cell(table, "inception", name, "fund") == headline[name], name


class TestWhatIsLeftBlank:

    def test_no_sharpe_ratio_over_a_short_window(self,
                                                 table):
        for window in SHORT:
            assert blank(cell(table, window, "sharpe_ratio", "fund")), window
            assert blank(cell(table, window, "sharpe_ratio", "index")), window

        assert not blank(cell(table, "6M", "sharpe_ratio", "fund"))
        assert not blank(cell(table, "1M", "volatility", "fund"))

    def test_annualised_only_over_a_year_or_more(self,
                                                 table):
        for window in ("1M", "6M", "YTD"):
            assert blank(cell(table, window, "annualised_return", "fund")), window

        for window in ("1Y", "3Y", "5Y", "inception"):
            assert not blank(cell(table, window, "annualised_return", "fund")), window

    def test_an_analysis_can_set_the_shortest_windows(self):
        settings = AnalysisSettings(min_annualised_rate_window="0M",
                                    min_annualised_ratio_window="3M")
        table = analytics_of(100.0 * (1 + THEIRS).cumprod(), settings).trailing_statistics()

        assert not blank(cell(table, "1M", "annualised_return", "fund"))
        assert not blank(cell(table, "3M", "sharpe_ratio", "fund"))
        assert blank(cell(table, "1M", "sharpe_ratio", "fund"))

    def test_a_window_longer_than_the_run_is_blank(self,
                                                   table):
        rows = table[table["period"] == "10Y"]

        assert rows[["fund", "index"]].map(blank).all().all()

    def test_no_tracking_over_different_days(self):
        """A comparator that starts after the run has a shorter inception, so
        the two are not compared over it."""
        later = 100.0 * (1 + THEIRS.loc["2024-06-03":]).cumprod()
        table = analytics_of(later).trailing_statistics()

        assert not blank(cell(table, "inception", "total_return", "index"))
        assert blank(cell(table, "inception", "tracking_difference", "index"))
        assert blank(cell(table, "inception", "tracking_error", "index"))
        assert not blank(cell(table, "1Y", "tracking_error", "index"))
        assert blank(cell(table, "3Y", "total_return", "index"))


class TestTheCatalogue:

    def test_it_is_a_table_with_labelled_columns(self):
        statistic = CATALOGUE["trailing_statistics"]

        assert statistic.kind == "table"
        assert [column.name for column in statistic.columns] == list(COLUMNS)
        assert statistic.columns[0].labels["inception"] == "Since inception"
        assert statistic.columns[1].labels["tracking_difference"] == "Tracking difference"
