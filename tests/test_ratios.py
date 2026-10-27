# tests/test_ratios.py
"""BN-375: the risk-adjusted ratios, hand-computed.

Four excess returns small enough to work through on paper for the downside
ratios; given drawdown depths for the drawdown ratios; known values for the
probabilistic and deflated Sharpe ratios; and returns that do and do not
trend for Lo's correction.
"""
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance import ratios

DAYS = pd.bdate_range("2024-01-02", periods=4)
EXCESS = pd.Series([0.02, -0.01, 0.03, -0.02], index=DAYS)
DOWNSIDE = math.sqrt((0.01 ** 2 + 0.02 ** 2) / 4)
DEPTHS = (-0.1, -0.3, -0.05, -0.2)


class TestTheDownside:

    def test_the_downside_deviation_is_over_every_day(self):
        assert ratios.downside_deviation(EXCESS) == pytest.approx(DOWNSIDE)

    def test_sortino(self):
        assert ratios.sortino(EXCESS, 4) == pytest.approx(0.005 / DOWNSIDE * 2)

    def test_kappa_3(self):
        third = ((0.01 ** 3 + 0.02 ** 3) / 4) ** (1 / 3)

        assert ratios.kappa_3(EXCESS) == pytest.approx(0.005 / third)

    def test_upside_potential(self):
        assert ratios.upside_potential(EXCESS) == pytest.approx(0.0125 / DOWNSIDE)

    def test_omega(self):
        assert ratios.omega(EXCESS) == pytest.approx(0.05 / 0.03)

    def test_without_a_shortfall_there_is_no_ratio(self):
        gains = pd.Series([0.01, 0.02], index=DAYS[:2])

        assert ratios.sortino(gains, 252) is None
        assert ratios.kappa_3(gains) is None
        assert ratios.upside_potential(gains) is None
        assert ratios.omega(gains) is None

    def test_the_risk_free_rate_is_taken_off_daily(self):
        excess = ratios.excess_returns(pd.Series([0.001] * 3), 252, 0.03)

        assert excess.iloc[0] == pytest.approx(0.001 - (1.03 ** (1 / 252) - 1))


class TestGainToPain:

    def test_the_months_sum_over_the_losing_months(self):
        monthly = pd.Series([0.05, -0.02, 0.03, -0.01])

        assert ratios.gain_to_pain(monthly) == pytest.approx(0.05 / 0.03)
        assert ratios.gain_to_pain(pd.Series([0.01, 0.02])) is None


class TestTheDrawdownRatios:

    def test_calmar(self):
        assert ratios.calmar(0.2, -0.25) == pytest.approx(0.8)
        assert ratios.calmar(0.2, 0.0) is None

    def test_sterling_averages_the_three_deepest(self):
        assert ratios.sterling(0.1, DEPTHS) == pytest.approx(0.1 / 0.2)

    def test_burke(self):
        assert ratios.burke(0.1, DEPTHS) == pytest.approx(
            0.1 / math.sqrt(0.01 + 0.09 + 0.0025 + 0.04))

    def test_martin(self):
        excess = pd.Series([0.001] * 10)

        assert ratios.martin(excess, 0.05, 252) == pytest.approx(0.001 * 252 / 0.05)

    def test_without_a_drawdown_there_is_no_ratio(self):
        assert ratios.sterling(0.1, []) is None
        assert ratios.burke(0.1, []) is None
        assert ratios.martin(pd.Series([0.001]), 0.0, 252) is None


class TestTheProbabilityTheSharpeRatioIsReal:

    @pytest.fixture
    def normal(self):
        random = np.random.default_rng(5)

        return pd.Series(random.normal(0.0005, 0.01, 1500))

    def test_the_probabilistic_sharpe_ratio(self,
                                            normal):
        daily = normal.mean() / normal.std()
        skew, kurtosis = ratios.moments(normal)
        expected = NormalDist().cdf(daily * math.sqrt(1499)
                                    / math.sqrt(1 - skew * daily + (kurtosis - 1) / 4 * daily ** 2))

        assert ratios.probabilistic_sharpe(normal, 252) == pytest.approx(expected)

    def test_it_is_even_odds_against_its_own_ratio(self,
                                                   normal):
        own = normal.mean() / normal.std() * math.sqrt(252)

        assert ratios.probabilistic_sharpe(normal, 252, own) == pytest.approx(0.5)

    def test_the_moments_of_a_symmetric_series(self):
        skew, kurtosis = ratios.moments(pd.Series([-1.0, 0.0, 1.0] * 50))

        assert skew == pytest.approx(0.0) and kurtosis == pytest.approx(1.5)

    def test_the_best_of_ten_tries_by_luck(self):
        """About 1.57 standard deviations for ten trials (Bailey and
        Lopez de Prado)."""
        assert ratios.expected_best_sharpe(10, 1.0) == pytest.approx(1.5745, abs=1e-3)

    def test_the_deflated_ratio_is_below_the_probabilistic_one(self,
                                                               normal):
        deflated = ratios.deflated_sharpe(normal, 252, trials=20, trial_variance=0.25)

        assert deflated < ratios.probabilistic_sharpe(normal, 252)
        assert ratios.deflated_sharpe(normal, 252, None, None) is None
        assert ratios.deflated_sharpe(normal, 252, 1, 0.25) is None


class TestLo:

    def test_returns_that_trend_have_a_lower_ratio(self):
        random = np.random.default_rng(9)
        noise = random.normal(0.0005, 0.01, 3000)
        trending = np.empty_like(noise)
        trending[0] = noise[0]

        for day in range(1, len(noise)):
            trending[day] = 0.4 * trending[day - 1] + noise[day]

        series = pd.Series(trending)
        naive = series.mean() / series.std() * math.sqrt(252)

        assert ratios.lo_sharpe(series, 252) < naive

    def test_returns_that_do_not_trend_are_close_to_the_sharpe_ratio(self):
        random = np.random.default_rng(10)
        series = pd.Series(random.normal(0.0005, 0.01, 20000))
        naive = series.mean() / series.std() * math.sqrt(252)

        assert ratios.lo_sharpe(series, 252) == pytest.approx(naive, rel=0.1)

    def test_too_few_returns_have_none(self):
        assert ratios.lo_sharpe(pd.Series([0.01, 0.02, -0.01]), 252) is None


class TestThroughTheAnalytics:

    def test_every_ratio_is_there_and_calmar_is_the_headline_s(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.analytics import PerformanceAnalytics
        from test_strategy_futures import index, run

        result = run(index())
        figures = PerformanceAnalytics(RunData.from_result(result)).ratios()
        summary = result.summary()

        assert set(figures) == {"sortino", "kappa_3", "upside_potential", "omega", "calmar",
                                "martin", "sterling", "burke", "gain_to_pain",
                                "probabilistic_sharpe", "deflated_sharpe", "lo_sharpe"}
        assert figures["calmar"] == pytest.approx(
            summary["annualised_return"] / abs(summary["max_drawdown"]))
        assert figures["deflated_sharpe"] is None
        assert 0.0 <= figures["probabilistic_sharpe"] <= 1.0
