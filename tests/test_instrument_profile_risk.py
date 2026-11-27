# tests/test_instrument_profile_risk.py
"""BN-470: a name's risk and liquidity at each rebalance,
`instrument.profile(name).risk` and `.liquidity`, and each name's expected
risk kept with the run.

The run is test_risk_contributions_over_time's thirty names over 300
sessions, its risk model estimated on the 100 sessions before each
rebalance, which the last three have; the fund is held against an equally
weighted index.
"""
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions
from beacon.analysis.reports.statistics import NO_DATA_STORE, Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.analysis.risk import factor_store, risk_store
from beacon.strategy.factors import default_factors
from test_factor_analytics import NAMES, REBALANCES, SOURCE, factors
from test_risk_contributions_over_time import noisy

NAME = NAMES[3]
OPTIONS = AnalysisOptions(risk_model=RiskModelOptions(window=100))


def request(options: AnalysisOptions = OPTIONS,
            chosen=None) -> AnalysisRequest:
    return AnalysisRequest(options=options, factors=chosen)


@pytest.fixture(scope="module")
def analytics():
    return Analytics(noisy(), request(chosen=factors()), source=SOURCE)


@pytest.fixture(scope="module")
def found(analytics):
    return analytics.instrument.profile(NAME).risk


@pytest.fixture(scope="module")
def over_time(analytics):
    return analytics._context.risk_over_time


class TestEachRebalance:

    def test_a_row_each_rebalance_its_model_has(self,
                                                found):
        assert list(found.index) == REBALANCES[2:]
        assert list(found.columns) == ["volatility_contribution", "volatility_share",
                                       "tracking_error_contribution", "tracking_error_share"]
        assert found.attrs == {"against": "index"}

    def test_each_is_the_model_s_that_day(self,
                                          found,
                                          over_time):
        for day in found.index:
            own = over_time.contributions(day).loc[NAME]
            active = over_time.contributions(day, against="index").loc[NAME]

            assert found.loc[day, "volatility_contribution"] == pytest.approx(
                own["contribution"], rel=1e-12)
            assert found.loc[day, "volatility_share"] == pytest.approx(own["share"], rel=1e-12)
            assert found.loc[day, "tracking_error_contribution"] == pytest.approx(
                active["contribution"], rel=1e-12)
            assert found.loc[day, "tracking_error_share"] == pytest.approx(active["share"],
                                                                           rel=1e-12)

    def test_the_last_is_the_per_name_table_s(self,
                                              analytics,
                                              found):
        table = analytics.instrument.table(columns=["risk_contribution"])

        assert found["tracking_error_contribution"].iloc[-1] == pytest.approx(
            table.loc[NAME, "risk_contribution"], rel=1e-12)

    def test_the_shares_add_up_to_1(self,
                                    analytics):
        table = analytics._context.name_risk

        assert table.groupby(level=0)["share/fund"].sum().tolist() == pytest.approx([1.0] * 3)

    def test_over_a_window(self,
                           analytics):
        later = analytics.instrument.profile(NAME, start=REBALANCES[3]).risk

        assert list(later.index) == REBALANCES[3:]


class TestARebalanceItCannotFit:

    def test_is_left_out_rather_than_failing_the_rest(self,
                                                      over_time,
                                                      monkeypatch):
        # BN-482: one rebalance failed the whole analysis after its work.
        from beacon.exceptions import CalculationError

        fitted = over_time.dates()
        original = over_time.contributions

        def refused(day,
                    **kwargs):
            if day == fitted[0]:
                raise CalculationError("Risk model", "no name held has returns")

            return original(day, **kwargs)

        monkeypatch.setattr(over_time, "contributions", refused)
        kept = risk_store.rebalance_risk(over_time)[risk_store.TABLE]

        assert sorted(kept.index.unique()) == fitted[1:]


class TestKeptWithTheRun:

    @pytest.fixture(scope="class")
    def folder(self,
               analytics,
               tmp_path_factory):
        folder = tmp_path_factory.mktemp("risk")
        tables = analytics._context.rebalance_risk

        # Kept as for the default factors: the fixture's read a function,
        # which is never kept.
        assert risk_store.keep(tables, default_factors(), OPTIONS.risk_model.settings(), folder)

        return folder

    def test_a_stored_run_reads_it_without_the_data_store(self,
                                                          found,
                                                          folder):
        stored = Analytics(noisy(), request(), None, folder)

        pd.testing.assert_frame_equal(stored.instrument.profile(NAME).risk, found,
                                      check_freq=False)

    def test_another_risk_model_is_worked_out_again(self,
                                                    folder):
        other = AnalysisOptions(risk_model=RiskModelOptions(window=120))
        stored = Analytics(noisy(), request(other), None, folder)

        with pytest.raises(Unavailable) as raised:
            _ = stored.instrument.profile(NAME).risk

        assert raised.value.code == NO_DATA_STORE

    def test_factors_read_from_a_function_are_never_kept(self,
                                                         analytics,
                                                         tmp_path):
        assert not risk_store.keep(analytics._context.rebalance_risk, factors(),
                                   OPTIONS.risk_model.settings(), tmp_path)

    def test_the_analysis_keeps_it_only_when_it_fitted_the_model(self,
                                                                 monkeypatch,
                                                                 tmp_path):
        monkeypatch.setattr(risk_store, "fingerprint", lambda chosen, settings: "key")
        monkeypatch.setattr(factor_store, "keep", lambda analytics, folder: False)
        context = Analytics(noisy(), request(chosen=factors()), source=SOURCE)._context
        _ = context.factor_analytics, context.risk_over_time

        assert not context.keep_work(tmp_path / "unfitted")

        context.risk_over_time.models.at(REBALANCES[2])

        assert context.keep_work(tmp_path / "fitted")
        assert (tmp_path / "fitted" / risk_store.FOLDER / risk_store.KEY_FILE).exists()


class TestItsLiquidity:
    """test_backtest_flows' fund of two names, rebalanced each month, with
    their volumes."""

    @pytest.fixture(scope="class")
    def data(self):
        from beacon.analysis.data import RunData
        from test_backtest_flows import run

        return RunData.from_result(run(capital=2e8))

    def test_each_rebalance_s_days_to_sell(self,
                                           data):
        names = Analytics(data).instrument
        found = names.profile("GROW").liquidity

        assert list(found.columns) == ["value", "traded_value", "days_to_sell"]
        assert len(found) == 6

        for day in found.index:
            held = names.days_to_sell(as_of=day).loc["GROW"]

            assert found.loc[day].tolist() == pytest.approx(held.tolist(), rel=1e-12)

    def test_each_name_shares_the_run_s_liquidity(self,
                                                  data):
        # BN-481: measured once for the run, not once a name.
        from beacon.analysis.analytics.profile import liquidity_of

        analytics = Analytics(data)
        context = analytics._context

        assert liquidity_of(context) is liquidity_of(context)
        assert analytics.instrument.profile("GROW").liquidity.equals(
            Analytics(data).instrument.profile("GROW").liquidity)

    def test_a_rebalance_the_fund_did_not_hold_it(self,
                                                  data):
        import dataclasses

        from beacon.analysis.panels import FUND_WEIGHTS

        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS].assign(FLAT=0.0)
        found = Analytics(dataclasses.replace(data, tables=tables)).instrument.profile(
            "FLAT").liquidity

        assert (found["value"] == 0.0).all() and (found["days_to_sell"] == 0.0).all()
        assert (found["traded_value"] == 5e7).all()

    def test_a_run_without_traded_values(self):
        from test_risk_contributions_over_time import noisy

        with pytest.raises(Unavailable):
            _ = Analytics(noisy()).instrument.profile(NAME).liquidity
