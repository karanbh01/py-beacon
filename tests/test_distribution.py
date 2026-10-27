# tests/test_distribution.py
"""BN-376: the shape of a run's daily returns, hand-computed.

Four returns small enough to work through for the spreads below a target and
below the mean; series with a known skew, tails or serial correlation; and a
generated year for the histogram and the resampled intervals.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from beacon.analysis.performance import distribution as shape

DAYS = pd.bdate_range("2024-01-02", periods=4)
FOUR = pd.Series([0.02, -0.01, 0.03, -0.02], index=DAYS)
YEAR = pd.Series(np.random.default_rng(12).normal(0.0004, 0.01, 252))


class TestSpreadsBelow:

    def test_the_downside_deviation(self):
        assert shape.downside_deviation(FOUR, 4) == pytest.approx(
            np.sqrt((0.01 ** 2 + 0.02 ** 2) / 4 * 4))

    def test_the_downside_is_below_the_risk_free_rate(self):
        assert shape.downside_deviation(pd.Series([0.0, 0.0]), 252, 0.03) == pytest.approx(
            (1.03 ** (1 / 252) - 1) * np.sqrt(252))

    def test_the_semi_variance_is_below_the_mean(self):
        assert shape.semi_variance(FOUR, 4) == pytest.approx((0.015 ** 2 + 0.025 ** 2) / 4 * 4)


class TestShape:

    def test_skew_and_kurtosis_are_the_sample_s(self):
        assert shape.skew(YEAR) == pytest.approx(stats.skew(YEAR, bias=False))
        assert shape.excess_kurtosis(YEAR) == pytest.approx(stats.kurtosis(YEAR, bias=False))

    def test_a_symmetric_series_has_no_skew(self):
        assert shape.skew(pd.Series([-0.01, 0.0, 0.01] * 20)) == pytest.approx(0.0, abs=1e-12)

    def test_too_few_or_flat_returns_have_none(self):
        assert shape.skew(FOUR.iloc[:2]) is None
        assert shape.excess_kurtosis(FOUR.iloc[:3]) is None
        assert shape.skew(pd.Series([0.001] * 10)) is None

    def test_the_tail_ratio(self):
        returns = pd.Series(np.linspace(-0.02, 0.04, 101))

        assert shape.tail_ratio(returns) == pytest.approx(
            abs(np.quantile(returns, 0.95)) / abs(np.quantile(returns, 0.05)))

    def test_serial_correlation_of_returns_that_alternate(self):
        returns = pd.Series([0.01, -0.01] * 50)

        found = shape.serial_correlation(returns, 3)

        assert found[1] == pytest.approx(-1.0) and found[2] == pytest.approx(1.0)
        assert set(found) == {1, 2, 3}
        assert shape.serial_correlation(FOUR.iloc[:2], 2) == {1: None, 2: None}


class TestHistogram:

    def test_every_return_is_counted(self):
        result = shape.histogram(YEAR, 20)

        assert len(result.edges) == 21 and len(result.counts) == 20
        assert sum(result.counts) == 252
        assert result.edges[0] == YEAR.min() and result.edges[-1] == YEAR.max()

    def test_the_normal_puts_most_of_the_returns_in_the_bins(self):
        result = shape.histogram(YEAR, 20)

        assert 0.9 * 252 < sum(result.normal) < 252

    def test_no_returns(self):
        assert shape.histogram(pd.Series(dtype=float)).counts == []


class TestBootstrap:

    def test_the_interval_holds_the_run_s_own_figures(self):
        intervals = shape.bootstrap(YEAR, 252)

        for figure in intervals.values():
            assert figure.low < figure.value < figure.high
            assert figure.confidence == 0.95

    def test_the_same_seed_draws_the_same_runs(self):
        first = shape.bootstrap(YEAR, 252, samples=200, seed=4)
        again = shape.bootstrap(YEAR, 252, samples=200, seed=4)
        other = shape.bootstrap(YEAR, 252, samples=200, seed=5)

        assert first == again and first != other

    def test_the_run_s_own_figures_are_its_annualised_return_and_sharpe_ratio(self):
        from beacon.analysis.performance import formulas

        intervals = shape.bootstrap(YEAR, 252, risk_free_rate=0.02, samples=50)

        assert intervals["sharpe_ratio"].value == formulas.sharpe_ratio(YEAR, 252, 0.02)
        assert intervals["annualised_return"].value == pytest.approx(
            (1 + YEAR).prod() ** (252 / 252) - 1)

    def test_blocks_keep_neighbouring_days_together(self):
        intervals = shape.bootstrap(YEAR, 252, samples=100, block=21)

        assert intervals["sharpe_ratio"].low < intervals["sharpe_ratio"].high

    def test_too_few_returns_have_none(self):
        assert shape.bootstrap(FOUR, 252, block=3) is None


class TestThroughTheAnalytics:

    def test_every_figure_is_there(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.analytics import PerformanceAnalytics
        from test_strategy_futures import index, run

        analytics = PerformanceAnalytics(RunData.from_result(run(index())))
        figures = analytics.distribution(bins=30)

        assert set(figures) == {"downside_deviation", "semi_variance", "skew",
                                "excess_kurtosis", "tail_ratio", "serial_correlation",
                                "histogram"}
        assert len(figures["histogram"].counts) == 30
        assert set(analytics.bootstrap(samples=100)) == {"annualised_return", "sharpe_ratio"}
