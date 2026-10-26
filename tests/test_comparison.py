# tests/test_comparison.py
"""BN-323 and BN-324: a strategy against a comparator, and by period; since
BN-372 through `beacon.analysis.performance.comparisons`.

Deterministic daily returns over 2024: a strategy and a comparator whose
figures are known, and windows that start together and apart.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance.comparisons import comparison_of, period_against, periods_of
from beacon.analysis.performance.formulas import hit_rate_monthly
from beacon.analysis.performance.formulas import returns_from_start as daily_returns

PPY = 252


def compare(returns, levels):
    return comparison_of(returns, levels, PPY)


def by_period(returns, frequency):
    return periods_of(returns, frequency, PPY)


def compare_returns(ours, theirs):
    return period_against(ours, theirs, PPY)

DAYS = pd.bdate_range("2024-01-02", "2024-12-31")
RANDOM = np.random.default_rng(41)
COMPARATOR_RETURNS = pd.Series(RANDOM.normal(0.0004, 0.01, len(DAYS)), index=DAYS)
# The strategy: the comparator plus a little noise and a small edge, its first
# return from its starting point.
STRATEGY_RETURNS = COMPARATOR_RETURNS + RANDOM.normal(0.0001, 0.002, len(DAYS))
COMPARATOR_LEVELS = 100.0 * (1.0 + COMPARATOR_RETURNS).cumprod()


class TestHitRate:

    def test_it_is_the_share_of_positive_months(self):
        returns = pd.Series([0.01, -0.02, 0.03, 0.01],
                            index=pd.to_datetime(["2024-01-15", "2024-02-15",
                                                  "2024-03-15", "2024-03-20"]))

        assert hit_rate_monthly(returns) == pytest.approx(2 / 3)

    def test_months_compound_within(self):
        """A month with a gain and a larger loss is a losing month."""
        returns = pd.Series([0.05, -0.06], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

        assert hit_rate_monthly(returns) == 0.0

    def test_no_returns_have_no_rate(self):
        assert hit_rate_monthly(pd.Series(dtype=float)) is None


class TestComparing:

    def test_over_the_whole_window_the_comparator_starts_at_its_first_level(self):
        comparison = compare(STRATEGY_RETURNS, COMPARATOR_LEVELS)

        assert comparison.start == DAYS[0] and comparison.observations == len(DAYS)
        assert comparison.level.iloc[0] == pytest.approx(100.0)
        assert comparison.own["total_return"] == pytest.approx(
            COMPARATOR_LEVELS.iloc[-1] / COMPARATOR_LEVELS.iloc[0] - 1)

    def test_the_relative_figures_are_a_backtest_s_tracking(self):
        """The tracking difference and error a backtest reports: the strategy
        from its starting point, the comparator from its first level."""
        comparison = compare(STRATEGY_RETURNS, COMPARATOR_LEVELS)
        theirs = daily_returns(COMPARATOR_LEVELS)
        ours_total = (1 + STRATEGY_RETURNS).prod() - 1
        theirs_total = COMPARATOR_LEVELS.iloc[-1] / COMPARATOR_LEVELS.iloc[0] - 1

        assert comparison.relative["relative_return"] == pytest.approx(ours_total - theirs_total)
        assert comparison.relative["tracking_error"] == pytest.approx(
            (STRATEGY_RETURNS - theirs).std() * np.sqrt(252))

    def test_the_information_ratio_is_the_annualised_difference_over_the_error(self):
        relative = compare(STRATEGY_RETURNS, COMPARATOR_LEVELS).relative

        assert relative["information_ratio"] == pytest.approx(
            relative["annualised_relative_return"] / relative["tracking_error"])
        assert relative["beta"] == pytest.approx(1.0, abs=0.1)
        assert relative["correlation"] > 0.9

    def test_a_later_comparator_is_measured_over_the_days_both_have(self):
        later = COMPARATOR_LEVELS.loc["2024-04-01":]

        comparison = compare(STRATEGY_RETURNS, later)

        assert comparison.start == later.index[0]
        assert comparison.relative["relative_return"] == pytest.approx(
            (1 + STRATEGY_RETURNS.loc[later.index[1]:]).prod()
            - later.iloc[-1] / later.iloc[0])

    def test_too_few_common_days_compare_nothing(self):
        assert compare(STRATEGY_RETURNS, COMPARATOR_LEVELS.iloc[:1]) is None


class TestByPeriod:

    @pytest.mark.parametrize(("frequency", "first", "count"), [
        ("annual", "2024", 1), ("quarterly", "2024 Q1", 4), ("monthly", "2024-01", 12)])
    def test_periods_are_named_and_in_order(self,
                                            frequency,
                                            first,
                                            count):
        periods = by_period(STRATEGY_RETURNS, frequency)

        assert periods[0].period == first and len(periods) == count

    def test_the_periods_compound_to_the_whole(self):
        periods = by_period(STRATEGY_RETURNS, "quarterly")
        compounded = np.prod([1 + period.figures["total_return"] for period in periods]) - 1

        assert compounded == pytest.approx((1 + STRATEGY_RETURNS).prod() - 1)

    def test_a_period_counts_its_positive_days(self):
        march = next(period for period in by_period(STRATEGY_RETURNS, "monthly")
                     if period.period == "2024-03")

        assert march.figures["positive_days"] == pytest.approx(
            (STRATEGY_RETURNS.loc["2024-03"] > 0).mean())

    def test_a_period_compares_over_its_own_days(self):
        theirs = daily_returns(COMPARATOR_LEVELS)
        first = by_period(STRATEGY_RETURNS, "annual")[0]

        figures = compare_returns(first.returns, theirs)
        whole = compare(STRATEGY_RETURNS, COMPARATOR_LEVELS)

        assert figures["relative"]["relative_return"] == pytest.approx(
            whole.relative["relative_return"])
        assert figures["own"]["total_return"] == pytest.approx(whole.own["total_return"])

    def test_an_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="weekly"):
            by_period(STRATEGY_RETURNS, "weekly")
