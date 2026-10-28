# tests/test_analysis_request.py
"""BN-397: a backtest's analysis, asked for up front or built on first read.

Runs reuse test_strategy_futures' eight-name market-cap index held
physically through 2024.
"""
import pandas as pd
import pytest

from beacon.analysis.attribution import Attribution
from beacon.analysis.data import RunData
from beacon.analysis.performance.analytics import PerformanceAnalytics
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.request import (
    ATTRIBUTION,
    PERFORMANCE,
    PORTFOLIO,
    RISK,
    TABLES,
    AnalysisRequest,
)
from beacon.analysis.risk.analytics import RiskAnalytics
from beacon.backtest import Backtest
from beacon.strategy import Factor
from beacon.strategy.signals import FunctionSignal
from test_strategy_futures import DATA, END, START, index

EVERYTHING = AnalysisRequest(portfolio=True, risk=True, attribution=True)


def run(strategy,
        analysis: AnalysisRequest | None = None):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False).run(
        strategy, start=START, end=END, analysis=analysis)


@pytest.fixture(scope="module")
def plain():
    return run(index())


@pytest.fixture(scope="module")
def analysed():
    return run(index(), analysis=EVERYTHING)


class TestBuiltOnFirstRead:

    def test_each_part_is_built_once_and_kept(self,
                                              plain):
        assert plain.analytics is plain.analytics
        assert plain.attribution is plain.attribution
        assert isinstance(plain.risk, RiskAnalytics)
        assert plain.analysis_request is None and plain.analysis_timings == {}

    def test_with_default_settings_its_figures_are_the_library_s(self,
                                                                 plain):
        own = PerformanceAnalytics(RunData.from_result(plain)).summary()

        assert plain.analytics.summary() == own
        assert plain.attribution.contribution().to_frame().equals(
            Attribution(RunData.from_result(plain)).contribution().to_frame())


class TestAskedForUpFront:

    def test_the_request_and_each_part_s_time_are_kept(self,
                                                       analysed):
        assert analysed.analysis_request is EVERYTHING
        assert list(analysed.analysis_timings) == [TABLES, PERFORMANCE, PORTFOLIO, RISK,
                                                   ATTRIBUTION]
        assert all(seconds >= 0.0 for seconds in analysed.analysis_timings.values())

    def test_the_run_is_unchanged(self,
                                  plain,
                                  analysed):
        pd.testing.assert_series_equal(analysed.portfolio.nav, plain.portfolio.nav)
        assert analysed.analytics.summary() == plain.analytics.summary()

    def test_performance_alone_by_default(self):
        assert AnalysisRequest().parts() == [PERFORMANCE]

    def test_its_settings_reach_the_analytics(self):
        settings = AnalysisSettings(start=pd.Timestamp("2024-07-01"))
        result = run(index(), analysis=AnalysisRequest(settings=settings))

        assert result.analytics.returns.index[0] >= pd.Timestamp("2024-07-01")

    def test_its_factors_reach_the_risk(self):
        factor = Factor("constant", FunctionSignal(lambda name, date, fetched: 1.0))
        result = run(index(), analysis=AnalysisRequest(factors=[factor]))

        assert result.risk.factors.names == ["constant"]


@pytest.mark.parametrize("module", ["beacon.analysis.request", "beacon.analysis.risk.analytics",
                                    "beacon.backtest.analytics_access"])
def test_each_imports_first_in_a_fresh_process(module):
    """The request is imported by the backtest, whose result its analysis
    reads: imported first, neither may find the other half-built."""
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", f"import {module}"], capture_output=True,
                          text=True, check=False)

    assert done.returncode == 0, done.stderr

