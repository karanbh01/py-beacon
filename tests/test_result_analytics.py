# tests/test_result_analytics.py
"""BN-433: `result.analytics`, the one way in to every calculation on a run.

Runs reuse test_strategy_futures' eight-name market-cap index held
physically through 2024.
"""
import pytest

from beacon.analysis.analytics import Analytics, Category, PortfolioLevel
from beacon.analysis.data import RunData
from beacon.analysis.request import AnalysisRequest
from beacon.backtest import Backtest
from test_strategy_futures import DATA, END, START, index

CATEGORIES = ["performance", "risk", "relative", "holdings", "trading", "flows", "income",
              "vehicle", "attribution"]
SHARPE = AnalysisRequest(reports={}, statistics=["sharpe_ratio"])


def run(analysis: AnalysisRequest | None = None):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
        index(), start=START, end=END, analysis=analysis)


@pytest.fixture(scope="module")
def plain():
    return run()


@pytest.fixture(scope="module")
def analysed():
    return run(SHARPE)


class TestTheEntryPoint:
    def test_it_is_built_once_and_kept(self,
                                       plain):
        assert isinstance(plain.analytics, Analytics)
        assert plain.analytics is plain.analytics

    def test_it_reads_the_run_the_result_holds(self,
                                               plain):
        assert plain.analytics.run_data is plain.run_data

    def test_a_portfolio_s_analytics_come_in_nine_categories(self,
                                                             plain):
        portfolio = plain.analytics.portfolio

        assert isinstance(portfolio, PortfolioLevel)
        assert [category.name for category in portfolio.categories()] == CATEGORIES
        assert all(isinstance(getattr(portfolio, name), Category) for name in CATEGORIES)
        assert portfolio.performance is portfolio.performance


class TestTheAnalysisAskedFor:
    def test_a_run_asked_for_none_has_no_request_or_timings(self,
                                                            plain):
        assert plain.analytics.request is None
        assert plain.analytics.timings == {}

    def test_the_request_timings_and_block_are_the_run_s(self,
                                                         analysed):
        analytics = analysed.analytics

        assert analytics.request is SHARPE
        assert analytics.timings == analysed.analysis_timings
        assert analytics.to_dict() is analysed.analysis_block

    def test_it_shares_the_work_of_the_run_s_analysis(self,
                                                      analysed):
        assert analysed.analytics._context is analysed._analysis_context

    def test_a_run_asked_for_none_works_out_the_factsheet_when_asked(self):
        result = run()
        block = result.analytics.to_dict()

        assert list(block["reports"]) == ["factsheet"]
        assert result.analytics.to_dict() is block
        assert result.analytics.timings == block["timings"]


class TestOfAStoredRun:
    def test_it_gives_what_the_finished_run_gave(self,
                                                 analysed):
        stored = Analytics(RunData.from_result(analysed), SHARPE)
        point = stored.to_dict()["statistics"]["sharpe_ratio"]["point"]

        assert point == analysed.analysis_block["statistics"]["sharpe_ratio"]["point"]
