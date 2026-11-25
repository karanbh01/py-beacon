# tests/test_analytics_risk_checks.py
"""BN-448 and BN-449: the risk model's checks as analytics,
`forecast_check()` and `bias_test()`, each what the run's risk over time
gives, and the concentration in crises, what the run's concentration gives.

The risk model's run is test_risk_contributions_over_time's thirty names in
three sectors, the model estimated on the 100 sessions before each
rebalance; the concentration's, test_strategy_futures' eight-name index held
through 2024, its benchmark the market crises are read from.
"""
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.performance.crises import THRESHOLD
from beacon.analysis.performance.settings import AnalysisSettings
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions
from beacon.analysis.reports.statistics import Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.analysis.risk.concentration import ConcentrationAnalytics
from beacon.analysis.risk.contributions import RiskOverTime
from beacon.analysis.risk.factors import FactorAnalytics
from test_factor_analytics import SOURCE, factors
from test_risk_contributions_over_time import noisy


@pytest.fixture(scope="module")
def risk():
    request = AnalysisRequest(options=AnalysisOptions(risk_model=RiskModelOptions(window=100)),
                              factors=factors())

    return Analytics(noisy(), request, source=SOURCE).portfolio.risk


@pytest.fixture(scope="module")
def over_time():
    return RiskOverTime(FactorAnalytics(noisy(), SOURCE, factors=factors()), window=100)


class TestTheForecastCheck:
    def test_it_is_the_run_s(self,
                             risk,
                             over_time):
        pd.testing.assert_frame_equal(risk.forecast_check(),
                                      over_time.forecast_check(against="index"))

    def test_on_a_rebalance(self,
                            risk):
        table = risk.forecast_check()
        day = table.index[1]

        pd.testing.assert_series_equal(risk.forecast_check(as_of=day), table.loc[day])


class TestTheBiasTest:
    def test_it_is_the_run_s(self,
                             risk,
                             over_time):
        found = risk.bias_test()
        today = over_time.bias_statistics(against="index")

        assert found.to_dict() == pytest.approx(today)
        assert list(found.index) == ["volatility", "tracking_error"]

    def test_without_the_data_store(self):
        with pytest.raises(Unavailable):
            Analytics(noisy()).portfolio.risk.bias_test()


class TestConcentrationInCrises:
    @pytest.fixture(scope="class")
    def result(self):
        from beacon.backtest import Backtest
        from beacon.index.calculation import IndexCalculator
        from test_strategy_futures import DATA, END, START, index

        benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START,
                                                                     end_date=END)

        return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                        transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                           end=END)

    def test_it_is_the_run_s(self,
                             result):
        found = result.analytics.portfolio.risk.crisis_concentration()
        today = ConcentrationAnalytics(result.run_data, None, AnalysisSettings()).in_crises(
            market="benchmark", threshold=THRESHOLD)

        pd.testing.assert_frame_equal(found, today)
        assert found.index[-1] == "calm"

    def test_within_a_window(self,
                             result):
        found = result.analytics.portfolio.risk.crisis_concentration(start="2024-07-01")
        settings = AnalysisSettings(start=pd.Timestamp("2024-07-01"))
        today = ConcentrationAnalytics(result.run_data, None, settings).in_crises(
            market="benchmark", threshold=THRESHOLD)

        pd.testing.assert_frame_equal(found, today)
