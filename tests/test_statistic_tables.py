# tests/test_statistic_tables.py
"""BN-467: named tables of single-figure statistics side by side in a run's
analysis, `analytics.table()` for the API, with a ``summary`` by default.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import pandas as pd
import pytest

from beacon.analysis.reports.rerun import rerun
from beacon.analysis.request import AnalysisRequest, analyse
from beacon.analysis.statistic_tables import SUMMARY, WINDOWS, StatisticTable
from beacon.backtest import Backtest
from beacon.server.analysis import build_request, catalogue
from beacon.server.analysis_schemas import AnalysisRequestSpec


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def block(result):
    return analyse(result, AnalysisRequest(reports={}))


def frame(found: dict) -> pd.DataFrame:
    return pd.DataFrame(found["data"], columns=found["columns"])


class TestTheSummary:
    def test_every_analysis_holds_it(self,
                                     block):
        summary = frame(block["statistic_tables"]["summary"])

        assert list(summary.columns[:3]) == ["statistic", "window", "unit"]
        assert {"fund", "index", "benchmark"} <= set(summary.columns)
        assert len(summary) == len(SUMMARY.statistics) * len(WINDOWS)
        assert list(summary["window"].unique()) == list(WINDOWS)

    def test_its_figures_are_the_analytics(self,
                                           result,
                                           block):
        summary = frame(block["statistic_tables"]["summary"]).set_index(["statistic", "window"])
        performance = result.analytics.portfolio.performance

        assert summary.loc[("sharpe_ratio", "6M"), "fund"] == pytest.approx(
            performance.sharpe_ratio(trailing=["6M"])["6M"])
        assert summary.loc[("total_return", "YTD"), "index"] == pytest.approx(
            performance.total_return(of="index"))

    def test_against_figures_under_the_comparator(self,
                                                  result,
                                                  block):
        summary = frame(block["statistic_tables"]["summary"]).set_index(["statistic", "window"])
        relative = result.analytics.portfolio.relative

        assert summary.loc[("tracking_error", "YTD"), "benchmark"] == pytest.approx(
            relative.tracking_error(against="benchmark"))
        assert pd.isna(summary.loc[("tracking_error", "YTD"), "fund"])

    def test_each_row_says_its_unit(self,
                                    block):
        summary = frame(block["statistic_tables"]["summary"])
        units = dict(zip(summary["statistic"], summary["unit"], strict=False))

        assert units["sharpe_ratio"] == "ratio"
        assert units["volatility"] == "fraction"

    def test_short_windows_follow_the_annualising_rule(self,
                                                       block):
        summary = frame(block["statistic_tables"]["summary"]).set_index(["statistic", "window"])

        assert pd.isna(summary.loc[("sharpe_ratio", "3M"), "fund"])
        assert not pd.isna(summary.loc[("volatility", "3M"), "fund"])


class TestOtherTables:
    def test_by_period_with_one_portfolio(self,
                                          result):
        block = analyse(result, AnalysisRequest(reports={}, statistic_tables={
            "quarters": {"statistics": ["total_return", "sharpe_ratio"], "by": "quarter",
                         "annualise": True}}))
        quarters = frame(block["statistic_tables"]["quarters"])

        assert list(quarters.columns) == ["statistic", "period", "unit", "value"]
        assert quarters["value"].notna().all()
        assert "summary" not in block["statistic_tables"]

    def test_over_the_window(self,
                             result):
        block = analyse(result, AnalysisRequest(reports={}, statistic_tables={
            "run": StatisticTable(statistics=["total_return", "tracking_error"], of="all")}))
        run = frame(block["statistic_tables"]["run"]).set_index("statistic")

        assert list(run.columns[:1]) == ["unit"]
        assert run.loc["total_return", "fund"] == pytest.approx(
            result.analytics.portfolio.performance.total_return())

    def test_none_asked_for(self,
                            result):
        block = analyse(result, AnalysisRequest(reports={}, statistic_tables={}))

        assert block["statistic_tables"] == {}

    def test_one_that_fails_is_unavailable(self,
                                           result):
        block = analyse(result, AnalysisRequest(reports={}, statistic_tables={
            "wide": {"statistics": ["value_at_risk"]}}))

        assert block["unavailable"]["statistic_tables/wide"]["code"] == "failed"


class TestWhatIsRefused:
    def test_an_unknown_statistic(self):
        with pytest.raises(ValueError, match="Unknown statistic sharp_ratio"):
            AnalysisRequest(statistic_tables={"t": {"statistics": ["sharp_ratio"]}})

    def test_windows_and_periods_at_once(self):
        with pytest.raises(ValueError, match="trailing windows or periods, not both"):
            StatisticTable(statistics=["total_return"], trailing=["1Y"], by="year")


class TestThroughTheServer:
    def test_the_request_takes_them(self):
        spec = AnalysisRequestSpec(statistic_tables={
            "years": {"statistics": ["total_return"], "by": "year"}})
        request = build_request(spec)

        assert request.as_data()["statistic_tables"]["years"]["by"] == "year"
        assert build_request(AnalysisRequestSpec()).as_data()["statistic_tables"] == {
            "summary": SUMMARY.as_data()}

    def test_the_catalogue_labels_the_windows(self):
        assert catalogue()["names"]["window"]["inception"] == "Since inception"


def test_a_rerun_keeps_the_tables(result,
                                  block):
    request = AnalysisRequest(reports={}, statistics=["volatility"], statistic_tables={})
    again = rerun(result.run_data, request, None, previous=block)

    assert "summary" in again["statistic_tables"]
