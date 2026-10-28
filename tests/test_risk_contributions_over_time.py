# tests/test_risk_contributions_over_time.py
"""BN-389: a run's expected and measured risk, and where it came from.

The run is test_factor_analytics' thirty names over 300 sessions, their
returns the market's plus two factors' plus a name's own noise, in three
sectors. The model is estimated on the 100 sessions before each rebalance,
which the last three have.
Every expected figure is checked against the model's covariance written out
in full, and the contributions against the totals they divide.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, NAME_RETURNS, REFERENCE
from beacon.analysis.risk.contributions import OTHER, RiskOverTime
from beacon.analysis.risk.factors import FactorAnalytics
from test_factor_analytics import (
    DAYS,
    EQUAL,
    FUND,
    NAME_RETURNS_TABLE,
    NAMES,
    REBALANCES,
    SOURCE,
    factors,
    run_data,
)

SECTORS = pd.Series(["Tech", "Energy", "Health"] * 10, index=NAMES)
NOISE = pd.DataFrame(np.random.default_rng(3).normal(0.0, 0.004, (len(DAYS), len(NAMES))),
                     index=DAYS, columns=NAMES)
NOISE.iloc[0] = 0.0


def noisy() -> RunData:
    """The run with each name's own noise, and its sectors kept."""
    data = run_data()
    returns = NAME_RETURNS_TABLE + NOISE
    fund = returns @ FUND
    fund.iloc[0] = 0.0
    tables = dict(data.tables)
    tables[NAME_RETURNS] = returns
    tables[REFERENCE] = tables[REFERENCE].assign(
        SECTOR=SECTORS.reindex(tables[REFERENCE]["IDENTIFIER"]).to_numpy())

    return RunData(currency="USD", returns=fund, nav=1e6 * (1.0 + fund).cumprod(),
                   units=fund * 0 + 1, initial_capital=1e6, periods_per_year=252,
                   risk_free_rate=0.0, tables=tables, comparators=data.comparators)


@pytest.fixture(scope="module")
def risk():
    return RiskOverTime(FactorAnalytics(noisy(), SOURCE, factors=factors()), window=100)


@pytest.fixture(scope="module")
def day(risk):
    return risk.dates()[0]


def dense(risk, day) -> pd.DataFrame:
    return risk.models.at(day).covariance()


class TestExpected:

    def test_only_rebalances_with_the_window_before_them(self,
                                                         risk):
        assert risk.dates() == REBALANCES[2:]

    def test_the_volatility_is_the_model_s(self,
                                           risk,
                                           day):
        sigma = dense(risk, day)
        weights = FUND.reindex(sigma.index)

        found = risk.expected(day)

        assert found["volatility"] == pytest.approx(float(np.sqrt(weights @ sigma @ weights)))
        assert found["volatility"] ** 2 == pytest.approx(
            found["factor_volatility"] ** 2 + found["specific_volatility"] ** 2)
        assert found["covered_weight"] == pytest.approx(1.0)

    def test_the_tracking_error_is_the_model_s(self,
                                               risk,
                                               day):
        sigma = dense(risk, day)
        active = (FUND - EQUAL).reindex(sigma.index)
        model = risk.models.at(day)

        found = risk.expected(day)

        assert found["tracking_error"] == pytest.approx(float(np.sqrt(active @ sigma @ active)))
        assert found["tracking_error"] == pytest.approx(
            model.decompose_active_risk(FUND.to_dict(), EQUAL.to_dict()).tracking_error)

    def test_diversification_by_hand(self,
                                     risk,
                                     day):
        sigma = dense(risk, day)
        weights = FUND.reindex(sigma.index)
        alone = np.sqrt(np.diag(sigma))
        total = float(weights @ sigma @ weights)
        weighted = float(weights @ alone)
        own = float(((weights * alone) ** 2).sum())

        found = risk.expected(day)

        assert found["diversification_ratio"] == pytest.approx(weighted / np.sqrt(total))
        assert found["average_correlation"] == pytest.approx((total - own) / (weighted ** 2 - own))

    def test_a_summary_row_a_rebalance(self,
                                       risk):
        assert list(risk.summary().index) == risk.dates()


class TestContributions:

    def test_they_add_up_to_the_volatility(self,
                                           risk):
        for day in risk.dates():
            table = risk.contributions(day)
            assert table["contribution"].sum() == pytest.approx(risk.expected(day)["volatility"])
            assert table["share"].sum() == pytest.approx(1.0)

    def test_marginal_by_hand(self,
                              risk,
                              day):
        sigma = dense(risk, day)
        weights = FUND.reindex(sigma.index)
        marginal = sigma @ weights / np.sqrt(weights @ sigma @ weights)

        table = risk.contributions(day)

        assert table["marginal"].sort_index().to_numpy() == pytest.approx(
            marginal.sort_index().to_numpy())

    def test_active_ones_add_up_to_the_tracking_error(self,
                                                      risk,
                                                      day):
        table = risk.contributions(day, against="index")

        assert table["contribution"].sum() == pytest.approx(
            risk.expected(day)["tracking_error"])

    def test_by_sector(self,
                       risk,
                       day):
        groups = risk.group_contributions(day, "SECTOR")

        assert sorted(groups.index) == ["Energy", "Health", "Tech"]
        assert groups["contribution"].sum() == pytest.approx(risk.expected(day)["volatility"])


class TestMeasured:

    def test_the_shares_add_up_to_one(self,
                                      risk):
        table = risk.measured()

        assert table["share"].sum() == pytest.approx(1.0)
        assert table["volatility"].sum() == pytest.approx(
            float(risk.run.returns.iloc[1:].std() * np.sqrt(252)))

    def test_nothing_is_left_when_the_names_are_the_whole_fund(self,
                                                               risk):
        assert abs(risk.measured().loc[OTHER, "share"]) < 1e-9

    def test_by_sector(self,
                       risk):
        table = risk.measured("SECTOR")

        assert set(table.index) == {"Tech", "Energy", "Health", OTHER}
        assert table["share"].sum() == pytest.approx(1.0)

    def test_name_returns_from_before_the_run_are_left_out(self):
        """The run's name returns start before its first day, as a run's do
        when its names' histories are read from before it."""
        data = noisy()
        tables = dict(data.tables)
        earlier = pd.bdate_range(end=DAYS[0] - pd.Timedelta(days=1), periods=30)
        tables[NAME_RETURNS] = pd.concat([NOISE.iloc[:30].set_axis(earlier), tables[NAME_RETURNS]])
        early = RunData(currency="USD", returns=data.returns, nav=data.nav, units=data.units,
                        initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                        tables=tables, comparators=data.comparators)

        table = RiskOverTime(FactorAnalytics(early, SOURCE, factors=factors()),
                             window=100).measured()

        assert table["share"].sum() == pytest.approx(1.0)

    def test_cash_is_other(self):
        data = noisy()
        tables = dict(data.tables)
        tables[FUND_WEIGHTS] = tables[FUND_WEIGHTS] * 0.8
        returns = (tables[NAME_RETURNS] @ (FUND * 0.8)).where(lambda found: found.index != DAYS[0],
                                                              0.0) + 0.0001
        held = RunData(currency="USD", returns=returns, nav=data.nav, units=data.units,
                       initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                       tables=tables, comparators=data.comparators)

        table = RiskOverTime(FactorAnalytics(held, SOURCE, factors=factors()),
                             window=100).measured()

        assert table["share"].sum() == pytest.approx(1.0)
        assert abs(table.loc[OTHER, "share"]) < 1e-6
