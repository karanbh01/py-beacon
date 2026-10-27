# tests/test_analysis_risk_model.py
"""BN-386: the risk model an analysis estimates at each rebalance.

The run is test_strategy_futures' eight-name market-cap index held
physically through 2024, rebalanced quarterly; its first rebalance has no
returns before it, so the models here are estimated over 40 sessions.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from beacon.analysis.risk.model import FACTOR, RebalanceRiskModels, RiskModelSettings
from beacon.exceptions import CalculationError
from beacon.risk import FactorRiskModel, RiskModel, estimate_risk_model, fit_factor_model
from test_strategy_futures import index, run

SHORT = RiskModelSettings(window=40)


@pytest.fixture(scope="module")
def data():
    return RunData.from_result(run(index()))


@pytest.fixture(scope="module")
def second(data):
    return RebalanceRiskModels(data, SHORT).dates[1]


def with_returns(data: RunData,
                 returns: pd.DataFrame) -> RunData:
    """*data* with its name returns replaced."""
    tables = {name: data.tables[name] for name in (REFERENCE, FUND_WEIGHTS, INDEX_WEIGHTS)}

    return RunData(currency=data.currency, returns=data.returns, nav=data.nav, units=data.units,
                   initial_capital=data.initial_capital,
                   periods_per_year=data.periods_per_year,
                   risk_free_rate=data.risk_free_rate,
                   tables={**tables, NAME_RETURNS: returns})


class TestEachRebalance:

    def test_the_model_is_estimated_on_the_sessions_before_it(self,
                                                              data,
                                                              second):
        models = RebalanceRiskModels(data, SHORT)
        returns = data.tables[NAME_RETURNS].sort_index()
        before = returns[returns.index < second].iloc[-40:]
        names = models.held(second)

        model = models.at(second)

        assert isinstance(model, RiskModel)
        assert sorted(model.asset_ids) == names
        expected = estimate_risk_model(before[names])
        pd.testing.assert_frame_equal(model.covariance, expected.covariance)

    def test_each_is_estimated_once(self,
                                    data,
                                    second):
        models = RebalanceRiskModels(data, SHORT)

        assert models.at(second) is models.at(str(second.date()))

    def test_the_names_held_are_the_fund_s_and_the_index_s(self,
                                                         data,
                                                         second):
        held = RebalanceRiskModels(data, SHORT).held(second)

        assert held == sorted(f"N{i}" for i in range(8))

    def test_a_name_without_the_whole_window_is_left_out(self,
                                                         data,
                                                         second):
        returns = data.tables[NAME_RETURNS].copy()
        returns.loc[returns.index < second - pd.Timedelta(days=10), "N3"] = np.nan
        models = RebalanceRiskModels(with_returns(data, returns), SHORT)

        assert "N3" not in models.at(second).asset_ids
        assert models.uncovered(second) == ["N3"]


class TestAFactorModel:

    def test_it_is_fitted_on_the_exposures_given(self,
                                                 data,
                                                 second):
        def exposures(day, names):
            return pd.DataFrame({"size": np.linspace(-1.0, 1.0, len(names))}, index=names)

        models = RebalanceRiskModels(data, RiskModelSettings(kind=FACTOR, window=40),
                                     exposures=exposures)
        returns = data.tables[NAME_RETURNS].sort_index()
        names = models.held(second)
        before = returns[returns.index < second].iloc[-40:][names]

        model = models.at(second)

        assert isinstance(model, FactorRiskModel)
        expected = fit_factor_model(before, exposures(second, names))
        pd.testing.assert_frame_equal(model.factor_covariance, expected.factor_covariance)

    def test_it_needs_exposures(self,
                                data):
        with pytest.raises(ValueError, match="exposures"):
            RebalanceRiskModels(data, RiskModelSettings(kind=FACTOR))


class TestRefused:

    def test_a_day_that_is_not_a_rebalance(self,
                                           data):
        with pytest.raises(KeyError, match="rebalances"):
            RebalanceRiskModels(data, SHORT).at("2024-02-14")

    def test_too_few_sessions_before_it(self,
                                        data):
        models = RebalanceRiskModels(data)

        with pytest.raises(CalculationError, match="fewer than the 252"):
            models.at(models.dates[1])

    @pytest.mark.parametrize(("settings", "message"), [
        ({"kind": "pca"}, "pca"), ({"window": 5}, "window"), ({"target": "none"}, "none"),
        ({"intensity": 1.5}, "intensity")])
    def test_settings_out_of_range(self,
                                   settings,
                                   message):
        with pytest.raises(ValueError, match=message):
            RiskModelSettings(**settings)
