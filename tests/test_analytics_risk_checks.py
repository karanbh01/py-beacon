# tests/test_analytics_risk_checks.py
"""BN-448: the risk model's checks as analytics, `forecast_check()` and
`bias_test()`, each what the run's risk over time gives.

The run is test_risk_contributions_over_time's thirty names in three sectors,
the model estimated on the 100 sessions before each rebalance.
"""
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions
from beacon.analysis.reports.statistics import Unavailable
from beacon.analysis.request import AnalysisRequest
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
