# tests/test_relative_analytics.py
"""BN-377: a run against one comparator, hand-computed and against scipy.

A fund that is 1.2 times a comparator plus a small edge and noise, for the
regression and its t-statistics; four months small enough to work through on
paper for capture, the batting average and the relative drawdowns; and the
futures tests' run for agreement with its comparisons.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from beacon.analysis.data import RunData
from beacon.analysis.performance.relative_analytics import RelativeAnalytics

PPY = 252


def relative(ours: pd.Series,
             theirs: pd.Series,
             periods_per_year: int = PPY,
             risk_free_rate: float = 0.0) -> RelativeAnalytics:
    """A run of *ours* against a comparator whose returns are *theirs*, both
    starting with a day of nothing, so every other day is compared."""
    levels = 100.0 * (1.0 + theirs).cumprod()
    run = RunData.of_returns(ours, periods_per_year, risk_free_rate, {"index": levels})

    return RelativeAnalytics(run, "index")


DAYS = pd.bdate_range("2023-12-29", periods=1001)
RANDOM = np.random.default_rng(21)
THEIRS = pd.Series(np.r_[0.0, RANDOM.normal(0.0003, 0.01, 1000)], index=DAYS)
OURS = pd.Series(np.r_[0.0, 0.0002 + 1.2 * THEIRS.iloc[1:].to_numpy()
                       + RANDOM.normal(0.0, 0.002, 1000)], index=DAYS)

# One return a month, the first month's nothing.
MONTHS = pd.to_datetime(["2023-12-29", "2024-01-31", "2024-02-29", "2024-03-29",
                         "2024-04-30"])
MONTHLY_THEIRS = pd.Series([0.0, 0.02, -0.01, 0.03, -0.02], index=MONTHS)
MONTHLY_OURS = pd.Series([0.0, 0.03, -0.005, 0.02, -0.03], index=MONTHS)


class TestTheRegression:

    def test_alpha_beta_and_r_squared_are_ordinary_least_squares(self):
        fit = stats.linregress(THEIRS, OURS)

        regression = relative(OURS, THEIRS).regression()

        assert regression.beta == pytest.approx(fit.slope, rel=1e-10)
        assert regression.alpha == pytest.approx(fit.intercept * PPY, rel=1e-10)
        assert regression.r_squared == pytest.approx(fit.rvalue ** 2, rel=1e-10)

    def test_alpha_s_t_statistic_and_its_probability(self):
        fit = stats.linregress(THEIRS, OURS)
        t = fit.intercept / fit.intercept_stderr

        regression = relative(OURS, THEIRS).regression()

        assert regression.alpha_t == pytest.approx(t, rel=1e-9)
        assert regression.alpha_p == pytest.approx(2 * stats.t.sf(abs(t), len(OURS) - 2))

    def test_the_risk_free_rate_comes_off_both(self):
        """Taking a constant off both leaves the slope and moves the
        intercept by the rate times one less the slope."""
        base = relative(OURS, THEIRS).regression()
        with_rate = relative(OURS, THEIRS, risk_free_rate=0.03).regression()
        daily = 1.03 ** (1 / PPY) - 1

        assert with_rate.beta == pytest.approx(base.beta, rel=1e-10)
        assert with_rate.alpha == pytest.approx(base.alpha + daily * (base.beta - 1) * PPY,
                                                rel=1e-8)

    def test_treynor(self):
        analytics = relative(OURS, THEIRS)

        assert analytics.treynor() == pytest.approx(
            OURS.mean() * PPY / analytics.regression().beta)

    def test_a_comparator_that_does_not_vary_has_no_regression(self):
        flat = pd.Series(0.0, index=DAYS)

        assert relative(OURS, flat).regression() is None
        assert relative(OURS, flat).treynor() is None


class TestTheInformationRatio:

    def test_its_t_statistic_is_it_times_the_root_of_the_years(self):
        analytics = relative(OURS, THEIRS)
        active = OURS - THEIRS
        ratio = active.mean() / active.std() * np.sqrt(PPY)

        found = analytics.information_ratio_t()

        assert found["t"] == pytest.approx(ratio * np.sqrt(len(active) / PPY))
        assert found["p"] == pytest.approx(2 * stats.t.sf(abs(found["t"]), len(active) - 1))

    def test_it_is_the_t_statistic_of_the_mean_active_return(self):
        active = OURS - THEIRS

        assert relative(OURS, THEIRS).information_ratio_t()["t"] == pytest.approx(
            stats.ttest_1samp(active, 0.0).statistic)


class TestMonths:

    def test_up_and_down_capture(self):
        found = relative(MONTHLY_OURS, MONTHLY_THEIRS, 12).capture()
        down = ((0.995 * 0.97) ** 0.5 - 1) / ((0.99 * 0.98) ** 0.5 - 1)

        assert found["up"] == pytest.approx(1.0)
        assert found["down"] == pytest.approx(down)
        assert found["ratio"] == pytest.approx(1.0 / down)

    def test_the_batting_average(self):
        assert relative(MONTHLY_OURS, MONTHLY_THEIRS, 12).batting_average() == pytest.approx(
            2 / 5)

    def test_the_tracking_difference_by_year(self):
        years, spread = relative(OURS, THEIRS).tracking_difference_by_year()

        assert [year.year for year in years] == ["2023", "2024", "2025", "2026", "2027"]
        assert years[1].difference == pytest.approx(
            (1 + OURS.loc["2024"]).prod() - (1 + THEIRS.loc["2024"]).prod())
        assert spread == pytest.approx(pd.Series([year.difference for year in years]).std())


class TestFallingBehind:

    def test_relative_drawdowns_are_of_the_wealth_ratio(self):
        analytics = relative(MONTHLY_OURS, MONTHLY_THEIRS, 12)
        wealth = (1 + MONTHLY_OURS).cumprod() / (1 + MONTHLY_THEIRS).cumprod()

        levels = analytics.relative_levels()
        found = analytics.relative_drawdowns()

        assert levels.iloc[1:].to_numpy() == pytest.approx(wealth.to_numpy())
        assert found.worst[0].depth == pytest.approx(
            (wealth / wealth.cummax() - 1).min())

    def test_the_downside_tracking_error(self):
        active = (MONTHLY_OURS - MONTHLY_THEIRS).to_numpy()

        assert relative(MONTHLY_OURS, MONTHLY_THEIRS, 12).downside_tracking_error() == (
            pytest.approx(np.sqrt(np.mean(np.minimum(active, 0) ** 2) * 12)))


class TestAgainstTheRunsComparisons:

    @pytest.fixture(scope="class")
    def run(self):
        from test_strategy_futures import index, run

        return RunData.from_result(run(index()))

    def test_today_s_figures_are_the_comparison_s(self,
                                                  run):
        from beacon.analysis.performance.analytics import PerformanceAnalytics

        figures = RelativeAnalytics(run, "index").figures()
        comparison = PerformanceAnalytics(run).comparison("index").relative

        assert {name: figures[name] for name in comparison} == comparison

    def test_every_figure_is_there(self,
                                   run):
        figures = RelativeAnalytics(run).figures()

        assert {"alpha", "r_squared", "alpha_t", "alpha_p", "residual_risk",
                "information_ratio_t", "treynor", "capture", "batting_average",
                "downside_tracking_error", "relative_drawdowns",
                "tracking_difference_by_year", "tracking_difference_spread"} <= set(figures)
        assert figures["batting_average"] == figures["hit_rate_monthly"]

    def test_a_comparator_the_run_lacks_is_refused(self,
                                                   run):
        with pytest.raises(KeyError):
            RelativeAnalytics(run, "benchmark")
