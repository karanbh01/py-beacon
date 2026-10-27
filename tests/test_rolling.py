# tests/test_rolling.py
"""BN-378: each rolling figure is the static formula on its window.

Two related generated series over 400 days; each day's rolling figure is
checked against the formula over the window ending that day, on days across
the series; and the outcomes over every window.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance import formulas, ratios
from beacon.analysis.performance.rolling import outcomes, rolling_figures, rolling_relative

PPY = 252
WINDOW = 63
DAYS = pd.bdate_range("2023-01-02", periods=400)
RANDOM = np.random.default_rng(31)
THEIRS = pd.Series(RANDOM.normal(0.0004, 0.01, 400), index=DAYS)
OURS = THEIRS * 0.9 + RANDOM.normal(0.0001, 0.004, 400)
CHECKED = [WINDOW - 1, 100, 250, 399]


class TestTheFigures:

    @pytest.fixture(scope="class")
    def figures(self):
        return rolling_figures(OURS, WINDOW, PPY, risk_free_rate=0.02)

    def test_blank_until_the_first_full_window(self,
                                               figures):
        assert figures.iloc[:WINDOW - 1].isna().all().all()
        assert figures.iloc[WINDOW - 1:].notna().all().all()

    @pytest.mark.parametrize("day", CHECKED)
    def test_each_is_the_formula_on_its_window(self,
                                               figures,
                                               day):
        window = OURS.iloc[day - WINDOW + 1:day + 1]
        excess = ratios.excess_returns(window, PPY, 0.02)
        row = figures.iloc[day]

        assert row["return"] == pytest.approx(formulas.compounded_return(window), rel=1e-10)
        assert row["volatility"] == pytest.approx(formulas.volatility(window, PPY), rel=1e-10)
        assert row["sharpe_ratio"] == pytest.approx(formulas.sharpe_ratio(window, PPY, 0.02),
                                                    rel=1e-9)
        assert row["sortino_ratio"] == pytest.approx(ratios.sortino(excess, PPY), rel=1e-9)
        assert row["max_drawdown"] == pytest.approx(
            formulas.max_drawdown(formulas.path_of(window)), rel=1e-10, abs=1e-15)

    def test_a_window_that_only_rose_has_no_drawdown_or_sortino(self):
        rising = pd.Series(0.001, index=DAYS[:10])

        figures = rolling_figures(rising, 5, PPY)

        assert (figures["max_drawdown"].dropna() == 0.0).all()
        assert figures["sortino_ratio"].dropna().empty
        assert (figures["sharpe_ratio"].dropna() == 0.0).all()


class TestAgainstAComparator:

    @pytest.fixture(scope="class")
    def relative(self):
        return rolling_relative(OURS, THEIRS, WINDOW, PPY)

    @pytest.mark.parametrize("day", CHECKED)
    def test_each_is_the_formula_on_its_window(self,
                                               relative,
                                               day):
        ours = OURS.iloc[day - WINDOW + 1:day + 1]
        theirs = THEIRS.iloc[day - WINDOW + 1:day + 1]
        row = relative.iloc[day]

        assert row["relative_return"] == pytest.approx(
            formulas.compounded_return(ours) - formulas.compounded_return(theirs), rel=1e-9)
        assert row["tracking_error"] == pytest.approx(
            formulas.tracking_error(ours - theirs, PPY), rel=1e-9)
        assert row["information_ratio"] == pytest.approx(
            formulas.information_ratio(ours - theirs, PPY), rel=1e-8)
        assert row["beta"] == pytest.approx(formulas.beta(ours, theirs), rel=1e-9)
        assert row["correlation"] == pytest.approx(formulas.correlation(ours, theirs), rel=1e-9)

    def test_against_a_comparator_that_does_not_vary(self):
        flat = pd.Series(0.0005, index=DAYS)

        relative = rolling_relative(OURS, flat, WINDOW, PPY)

        assert (relative["beta"].dropna() == 0.0).all()
        assert (relative["correlation"].dropna() == 0.0).all()


class TestOutcomes:

    def test_over_every_window(self):
        returns = pd.Series([np.nan, np.nan, 0.03, -0.01, 0.02, 0.05])
        relative = pd.Series([np.nan, np.nan, 0.01, -0.02, 0.0, 0.01])

        found = outcomes(returns, relative)

        assert (found["worst"], found["best"]) == (-0.01, 0.05)
        assert found["median"] == pytest.approx(0.025)
        assert found["share_positive"] == 0.75
        assert found["share_ahead"] == 0.5

    def test_no_full_window(self):
        assert outcomes(pd.Series([np.nan]))["worst"] is None
        assert outcomes(pd.Series([0.01]))["share_ahead"] is None


class TestThroughTheAnalytics:

    def test_the_run_rolled_against_its_index(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.analytics import PerformanceAnalytics
        from test_strategy_futures import index, run

        analytics = PerformanceAnalytics(RunData.from_result(run(index())))

        alone = analytics.rolling(63)
        against = analytics.rolling(63, "index")
        found = analytics.rolling_outcomes(63, "index")

        assert list(alone.columns) == ["return", "volatility", "sharpe_ratio",
                                       "sortino_ratio", "max_drawdown"]
        assert {"tracking_error", "beta"} <= set(against.columns)
        assert found["worst"] <= found["median"] <= found["best"]
        assert 0.0 <= found["share_ahead"] <= 1.0
        assert analytics.rolling(63, "benchmark").equals(alone)
