# tests/test_analytics_reports.py
"""BN-456: `result.analytics.reports`, a report's statistics worked out
together, each what its analytic gives called bare.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import pandas as pd
import pytest

from beacon.analysis import Statistic
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.definitions import REPORTS
from beacon.analysis.reports.options import AnalysisOptions
from beacon.backtest import Backtest


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def factsheet(result):
    return result.analytics.reports.factsheet()


class TestAReport:
    def test_its_statistics_in_the_report_s_order(self,
                                                  factsheet):
        assert factsheet.name == "factsheet" and factsheet.label == "Factsheet"
        assert factsheet.level == "core"
        assert factsheet.statistics == REPORTS["factsheet"].level("core").included

    def test_each_value_is_its_analytic_called_bare(self,
                                                    result,
                                                    factsheet):
        portfolio = result.analytics.portfolio

        assert factsheet["sharpe_ratio"] == portfolio.performance.sharpe_ratio()
        assert factsheet[Statistic.VOLATILITY] == portfolio.risk.volatility()
        assert factsheet["total_return"] == portfolio.performance.total_return()

    def test_a_folded_statistic_keeps_its_id(self,
                                             factsheet):
        assert "trailing_returns" in factsheet

    def test_what_the_run_cannot_give_says_why(self,
                                               factsheet):
        assert factsheet.unavailable["net_flows"].startswith("This needs a run with money")

        with pytest.raises(KeyError, match="net_flows could not be given"):
            factsheet["net_flows"]

    def test_what_it_could_add(self,
                               factsheet):
        assert factsheet.optional == REPORTS["factsheet"].level("core").optional

        with pytest.raises(KeyError, match="optional in the factsheet core"):
            factsheet["calmar_ratio"]

    def test_single_figures_side_by_side(self,
                                         result,
                                         factsheet):
        frame = factsheet.to_frame()

        assert list(frame.columns[:3]) == ["fund", "index", "benchmark"]
        assert frame.loc["sharpe_ratio", "fund"] == result.analytics.portfolio.performance.\
            sharpe_ratio()

    def test_as_the_api_gives_it(self,
                                 factsheet):
        block = factsheet.to_dict()

        assert list(block["reports"]) == ["factsheet"]
        assert "sharpe_ratio" in block["statistics"]


class TestAskingForOne:
    def test_with_statistics_added_and_taken_out(self,
                                                 result):
        found = result.analytics.reports.factsheet(add=["calmar_ratio"],
                                                   remove=["volatility"])

        assert "calmar_ratio" in found and "volatility" not in found
        assert found["calmar_ratio"] == result.analytics.portfolio.performance.calmar_ratio()

    def test_its_extended_level(self,
                                result):
        found = result.analytics.reports.risk(level="extended")

        assert found.statistics == REPORTS["risk"].level("extended").included

    def test_under_other_options(self,
                                 result):
        options = AnalysisOptions(window=AnalysisSettings(risk_free_rate=0.05))
        found = result.analytics.reports.performance(options=options)

        assert found["sharpe_ratio"] < result.analytics.portfolio.performance.sharpe_ratio()

    def test_a_statistic_it_cannot_add(self,
                                       result):
        with pytest.raises(ValueError, match="optional in the factsheet core"):
            result.analytics.reports.factsheet(add=["total_return"])

    def test_every_report_has_its_method(self,
                                         result):
        reports = result.analytics.reports

        assert all(callable(getattr(reports, name)) for name in REPORTS)


def test_the_reports_the_run_asked_for(result):
    asked = result.analytics.reports.asked

    assert list(asked) == ["factsheet"]
    assert isinstance(asked["factsheet"].to_frame(), pd.DataFrame)
