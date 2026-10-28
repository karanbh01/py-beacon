# tests/test_concentration_analytics.py
"""BN-391: concentration by weight, risk and outcome, on known weights and
contributions, and over a run.

By hand: weights of 40%, 30%, 20% and 10% have a Herfindahl of 0.30, an
effective number of 3.33 names, a normalised Herfindahl of (0.30 - 0.25) /
0.75, and a Lorenz curve through 0.1, 0.3 and 0.6, so a Gini of 0.25. The
run is test_risk_contributions_over_time's thirty names in three sectors.
"""
import math

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS
from beacon.analysis.performance.crises import Episode
from beacon.analysis.risk.concentration import (
    ConcentrationAnalytics,
    lorenz,
    outcome_figures,
    risk_figures,
    weight_figures,
)
from beacon.analysis.risk.contributions import RiskOverTime
from beacon.analysis.risk.factors import FactorAnalytics
from test_factor_analytics import DAYS, REBALANCES, SOURCE, factors
from test_risk_contributions_over_time import noisy

KNOWN = pd.DataFrame([[0.4, 0.3, 0.2, 0.1]], columns=list("ABCD"))


class TestByWeight:

    def test_by_hand(self):
        found = weight_figures(KNOWN, optional=True).iloc[0]

        assert found["herfindahl"] == pytest.approx(0.30)
        assert found["effective_number"] == pytest.approx(1 / 0.30)
        assert found["normalised_herfindahl"] == pytest.approx(0.05 / 0.75)
        assert (found["largest"], found["top_5"]) == pytest.approx((0.4, 1.0))
        assert (found["names_for_50"], found["names_for_80"], found["names_for_90"]) == (2, 3, 3)
        assert found["gini"] == pytest.approx(0.25)
        assert found["entropy"] == pytest.approx(-sum(w * math.log(w) for w in (0.4, 0.3, 0.2,
                                                                               0.1)))

    def test_an_equal_book_and_a_single_name(self):
        equal = weight_figures(pd.DataFrame([[0.25] * 4]), optional=True).iloc[0]
        single = weight_figures(pd.DataFrame([[1.0, 0.0, 0.0]])).iloc[0]

        assert equal["normalised_herfindahl"] == pytest.approx(0.0)
        assert equal["gini"] == pytest.approx(0.0)
        assert equal["effective_number"] == pytest.approx(4.0)
        assert single["normalised_herfindahl"] == 1.0
        assert single["names"] == 1

    def test_cash_is_left_out(self):
        assert weight_figures(KNOWN * 0.9).iloc[0]["herfindahl"] == pytest.approx(0.30)

    def test_the_lorenz_curve(self):
        curve = lorenz(KNOWN.iloc[0])

        assert curve["share_of_book"].tolist() == pytest.approx([0.0, 0.1, 0.3, 0.6, 1.0])
        assert curve["share_of_names"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


class TestByRiskAndOutcome:

    def test_risk_by_hand(self):
        found = risk_figures(pd.Series([0.5, 0.3, 0.2]), tops=(1, 2))

        assert found["risk_herfindahl"] == pytest.approx(0.38)
        assert found["effective_risk_sources"] == pytest.approx(1 / 0.38)
        assert (found["top_1_risk"], found["top_2_risk"]) == pytest.approx((0.5, 0.8))

    def test_outcome_by_hand(self):
        found = outcome_figures(pd.Series([0.03, 0.02, -0.01, 0.005]), held=4, top=2)

        assert found["return"] == pytest.approx(0.045)
        assert found["top_2_share"] == pytest.approx(0.05 / 0.045)
        assert found["breadth"] == pytest.approx(0.75)


@pytest.fixture(scope="module")
def run():
    return noisy()


@pytest.fixture(scope="module")
def analytics(run):
    risk = RiskOverTime(FactorAnalytics(run, SOURCE, factors=factors()), window=100)

    return ConcentrationAnalytics(run, risk)


class TestOverARun:

    def test_at_each_rebalance_and_each_day(self,
                                            analytics):
        assert list(analytics.by_weight().index) == REBALANCES
        assert len(analytics.by_weight(daily=True)) == len(DAYS)

    def test_across_sectors(self,
                            analytics):
        found = analytics.by_weight(field="SECTOR").iloc[-1]

        assert found["names"] == 3
        assert found["effective_number"] <= 3.0

    def test_against_the_index(self,
                               analytics):
        found = analytics.against().iloc[-1]

        # The index is equally weighted, the most diversified a book can be.
        assert found["herfindahl"] > 0.0
        assert found["effective_number"] < 0.0

    def test_by_period_and_in_crises(self,
                                     analytics):
        years = analytics.by_period()
        crises = analytics.in_crises(named=[Episode("spring", pd.Timestamp("2023-03-01"),
                                                    pd.Timestamp("2023-05-31"))])

        assert list(years.index) == ["2023", "2024"]
        assert list(crises.index) == ["spring", "calm"]

    def test_by_risk(self,
                     analytics):
        risk = analytics.by_risk()
        day = risk.index[-1]
        shares = analytics.risk.contributions(day)["share"]

        assert risk.loc[day, "effective_risk_sources"] == pytest.approx(1 / (shares ** 2).sum())
        assert risk.loc[day, "diversification_ratio"] > 1.0

    def test_the_absorption_ratio(self,
                                  analytics):
        day = analytics.risk.dates()[-1]
        values = np.sort(np.linalg.eigvalsh(analytics.risk.models.at(day).covariance()))[::-1]

        assert analytics.absorption_ratio(day) == pytest.approx(values[:6].sum() / values.sum())

    def test_by_outcome(self,
                        analytics):
        found = analytics.by_outcome(top=30)

        assert found["top_30_share"] == pytest.approx(1.0)
        assert 0.0 <= found["breadth"] <= 1.0


class TestDrift:

    def test_half_the_gaps_since_the_last_rebalance(self,
                                                    run):
        tables = dict(run.tables)
        weights = tables[FUND_WEIGHTS].copy()
        moved = DAYS[10]
        weights.loc[moved, "N00"] += 0.02
        weights.loc[moved, "N01"] -= 0.02
        tables[FUND_WEIGHTS] = weights
        drifted = RunData(currency="USD", returns=run.returns, nav=run.nav, units=run.units,
                          initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                          tables=tables, comparators=run.comparators)

        drift = ConcentrationAnalytics(drifted).drift()

        assert drift.loc[DAYS[0]] == 0.0
        assert drift.loc[moved] == pytest.approx(0.02)
        assert drift.loc[DAYS[11]] == pytest.approx(0.0)

    def test_by_risk_needs_the_risk(self,
                                    run):
        with pytest.raises(ValueError, match="RiskOverTime"):
            ConcentrationAnalytics(run).by_risk()
