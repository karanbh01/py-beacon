# tests/test_instrument_profile_factors.py
"""BN-471: a name's factors at each rebalance, `instrument.profile(name)`'s
`factor_exposures` (what the risk model says it is) and `factor_betas` (how
its returns behaved).

The run is test_risk_contributions_over_time's thirty names over 300
sessions on two style factors, the model estimated on the 100 sessions
before each rebalance, which the last three have, and the betas fitted over
100 days; the countries, sectors and currencies are test_global_risk_model's
sixty names in four countries.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import columns_of
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions, RollingOptions
from beacon.analysis.reports.statistics import Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.analysis.risk import risk_store
from beacon.analysis.risk.factors import regressed
from beacon.strategy.factors import default_factors
from test_factor_analytics import NAMES, REBALANCES, SOURCE, factors
from test_risk_contributions_over_time import noisy

NAME = NAMES[5]
WINDOW = 100
OPTIONS = AnalysisOptions(risk_model=RiskModelOptions(window=100),
                          rolling=RollingOptions(window=WINDOW))


@pytest.fixture(scope="module")
def analytics():
    return Analytics(noisy(), AnalysisRequest(options=OPTIONS, factors=factors()), source=SOURCE)


@pytest.fixture(scope="module")
def profile(analytics):
    return analytics.instrument.profile(NAME)


class TestItsExposures:

    def test_the_model_s_each_rebalance(self,
                                        analytics,
                                        profile):
        found = profile.factor_exposures
        models = analytics._context.risk_over_time.models

        assert list(found.index) == REBALANCES[2:]
        assert list(found.columns) == ["exposure/market", "exposure/first", "exposure/second"]

        for day in found.index:
            assert found.loc[day].tolist() == pytest.approx(
                models.at(day).exposures.loc[NAME].tolist(), rel=1e-12)

    def test_their_labels(self,
                          profile):
        columns = columns_of(profile.factor_exposures)

        assert (columns[1].label, columns[1].unit) == ("First", "ratio")

    def test_a_stored_run_s_without_the_data_store(self,
                                                   analytics,
                                                   profile,
                                                   tmp_path):
        assert risk_store.keep(analytics._context.rebalance_risk, default_factors(),
                               OPTIONS.risk_model.settings(), tmp_path)
        stored = Analytics(noisy(), AnalysisRequest(options=OPTIONS), None, tmp_path)

        pd.testing.assert_frame_equal(stored.instrument.profile(NAME).factor_exposures,
                                      profile.factor_exposures, check_freq=False)


class TestTheGroupsOfAGlobalModel:

    @pytest.fixture(scope="class")
    def found(self):
        from test_global_risk_model import REBALANCES as GLOBAL
        from test_global_risk_model import models

        return risk_store.grouped_exposures(models().at(GLOBAL[1]).exposures)

    def test_each_name_s_country_sector_and_currency(self,
                                                     found):
        assert found.loc["R00", ["country", "sector", "currency"]].tolist() == ["US", "A", None]
        assert found.loc["R30", ["country", "currency"]].tolist() == ["DE", "EUR"]
        # Four names are too few for a country of their own.
        assert found.loc["R50", ["country", "currency"]].tolist() == ["Asia Pacific (other)",
                                                                      "NZD"]

    def test_the_styles_stay_numbers(self,
                                     found):
        assert {"exposure/market", "exposure/size"} <= set(found.columns)
        assert not any(column.startswith(("exposure/country", "exposure/sector"))
                       for column in found.columns)


class TestItsBetas:

    def test_each_rebalance_s_regression(self,
                                         analytics,
                                         profile):
        found = profile.factor_betas
        factor_returns = analytics._context.factor_analytics.factor_returns()
        mine = analytics._context.run.tables["name_returns"][NAME].rename("target")
        paired = factor_returns.join(mine, how="inner").dropna()
        day = found.index[-1]
        before = paired.loc[paired.index <= day].iloc[-WINDOW:]
        fitted = regressed(before["target"], before.drop(columns="target"), 252)

        assert found.attrs == {"window": WINDOW}
        assert found.loc[day, "beta/first"] == pytest.approx(
            fitted.coefficients.loc["first", "coefficient"], rel=1e-10)
        assert found.loc[day, "t_stat/market"] == pytest.approx(
            fitted.coefficients.loc["market", "t_stat"], rel=1e-10)
        assert found.loc[day, "r_squared"] == pytest.approx(fitted.r_squared, rel=1e-10)
        assert found.loc[day, "observations"] == WINDOW

    def test_only_rebalances_with_the_window_before_them(self,
                                                         profile):
        found = profile.factor_betas

        assert set(found.index) <= set(REBALANCES)
        assert REBALANCES[0] not in found.index
        assert np.isfinite(found.to_numpy(dtype=float)).all()

    def test_their_labels(self,
                          profile):
        labels = {column.name: column.label for column in columns_of(profile.factor_betas)}

        assert labels["beta/first"] == "First beta"
        assert labels["t_stat/first"] == "First t-statistic"

    def test_without_the_data_store(self,
                                    tmp_path):
        stored = Analytics(noisy(), AnalysisRequest(options=OPTIONS), None, tmp_path)

        with pytest.raises(Unavailable):
            _ = stored.instrument.profile(NAME).factor_betas
