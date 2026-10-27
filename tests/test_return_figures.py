# tests/test_return_figures.py
"""BN-373: a run's return figures, hand-computed on small series.

A steady 0.1% a business day from 15 November 2023 to 28 February 2025, so a
window's return is 1.001 to the power of its days, counted here
independently; patterned series for the best and worst periods, the gains and
losses and the streaks of months.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance import return_figures as figures

DAYS = pd.bdate_range("2023-11-15", "2025-02-28")
STEADY = pd.Series(0.001, index=DAYS)
PPY = 252


def grown(days: int) -> float:
    return 1.001 ** days - 1


def business_days(start: str,
                  end: str) -> int:
    return len(pd.bdate_range(start, end))


class TestTrailing:

    def test_a_month_runs_from_the_close_a_month_before(self):
        """28 January 2025 was a Tuesday, so the month runs from the 29th."""
        month = figures.trailing_returns(STEADY, PPY)["1M"]

        assert month.start == pd.Timestamp("2025-01-29")
        assert month.days == business_days("2025-01-29", "2025-02-28")
        assert month.total == pytest.approx(grown(month.days), rel=1e-12)
        assert month.annualised is None

    def test_a_year_runs_from_the_close_a_year_before(self):
        """28 February 2024 was a Wednesday, so the year runs from the 29th.
        It holds more business days than the 252 a year, and is not
        annualised: only windows longer than a year are."""
        year = figures.trailing_returns(STEADY, PPY)["1Y"]

        assert year.start == pd.Timestamp("2024-02-29")
        assert year.days == business_days("2024-02-29", "2025-02-28") > PPY
        assert year.total == pytest.approx(grown(year.days), rel=1e-12)
        assert year.annualised is None

    def test_a_window_longer_than_the_run_has_none(self):
        trailing = figures.trailing_returns(STEADY, PPY)

        assert trailing["3Y"] is None and trailing["10Y"] is None
        assert trailing["6M"] is not None

    def test_a_close_on_a_weekend_anchor_falls_back_to_the_friday(self):
        """Six months before Friday 28 February 2025 is Wednesday 28 August
        2024; one month before Sunday 30 June 2024 is Thursday 30 May."""
        ending = STEADY.loc[:"2024-06-28"]

        month = figures.trailing_returns(ending, PPY)["1M"]

        assert month.start == pd.Timestamp("2024-05-29")

    def test_a_trailing_return_links_with_what_came_before_to_the_whole(self):
        random = np.random.default_rng(3)
        returns = pd.Series(random.normal(0.0004, 0.01, len(DAYS)), index=DAYS)

        for name, window in figures.trailing_returns(returns, PPY).items():
            if window is None:
                continue

            before = returns.loc[:window.start].iloc[:-1]
            linked = (1 + (1 + before).prod() - 1) * (1 + window.total) - 1

            assert linked == pytest.approx((1 + returns).prod() - 1, rel=1e-12), name

    def test_no_returns_have_no_trailing_returns(self):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))

        assert set(figures.trailing_returns(empty, PPY).values()) == {None}


class TestToDate:

    def test_the_year_runs_from_its_first_day(self):
        to_date = figures.returns_to_date(STEADY, PPY)

        assert to_date["YTD"].start == pd.Timestamp("2025-01-01")
        assert to_date["YTD"].total == pytest.approx(grown(business_days("2025-01-01",
                                                                          "2025-02-28")))
        assert to_date["QTD"].start == pd.Timestamp("2025-01-01")
        assert to_date["MTD"].start == pd.Timestamp("2025-02-03")

    def test_a_run_that_began_within_the_year_runs_from_its_start(self):
        to_date = figures.returns_to_date(STEADY.loc[:"2023-12-29"], PPY)

        assert to_date["YTD"].start == DAYS[0]
        assert to_date["MTD"].start == pd.Timestamp("2023-12-01")


class TestPeriods:

    def test_weeks_are_iso_weeks(self):
        periods = figures.period_returns(STEADY.loc[:"2023-11-30"], figures.WEEKLY)

        assert [period.period for period in periods] == ["2023-W46", "2023-W47", "2023-W48"]
        assert periods[0].value == pytest.approx(grown(3))

    def test_the_first_and_last_periods_are_partial_only_when_cut(self):
        months = figures.period_returns(STEADY, "monthly")
        years = figures.period_returns(STEADY, "annual")

        assert months[0].partial and not months[-1].partial
        assert not any(month.partial for month in months[1:-1])
        assert years[0].partial and years[-1].partial and not years[1].partial

    def test_best_and_worst_carry_their_dates(self):
        returns = STEADY.copy()
        returns.loc["2024-06-12"] = 0.05
        returns.loc["2024-09-10"] = -0.04

        best, worst = figures.best_and_worst(returns, figures.DAILY)
        best_month, worst_month = figures.best_and_worst(returns, "monthly")

        assert best.period == "2024-06-12" and best.value == pytest.approx(0.05)
        assert worst.period == "2024-09-10" and worst.value == pytest.approx(-0.04)
        assert best_month.period == "2024-06" and worst_month.period == "2024-09"
        assert worst_month.start == pd.Timestamp("2024-09-02")

    def test_the_share_of_positive_periods(self):
        returns = pd.Series([0.01, -0.01, 0.02, 0.0],
                            index=pd.to_datetime(["2024-01-02", "2024-01-03",
                                                  "2024-02-01", "2024-03-01"]))

        assert figures.positive_share(returns, figures.DAILY) == 0.5
        assert figures.positive_share(returns, "monthly") == pytest.approx(1 / 3)

    def test_an_unknown_frequency_is_refused(self):
        with pytest.raises(ValueError, match="hourly"):
            figures.period_returns(STEADY, "hourly")


class TestGainsAndLosses:

    def test_the_averages_and_their_ratio(self):
        returns = pd.Series([0.02, -0.01, 0.01, -0.03, 0.0], index=DAYS[:5])

        result = figures.gains_and_losses(returns)

        assert result.average_gain == pytest.approx(0.015)
        assert result.average_loss == pytest.approx(-0.02)
        assert result.ratio == pytest.approx(0.75)
        assert (result.gains, result.losses) == (2, 2)

    def test_without_a_loss_there_is_no_ratio(self):
        result = figures.gains_and_losses(STEADY.iloc[:10])

        assert result.average_loss is None and result.ratio is None


class TestStreaks:

    def test_the_longest_runs_of_months(self):
        values = [0.01, 0.02, 0.01, -0.01, -0.02, 0.0, 0.03, -0.01, -0.01, -0.02, -0.01, 0.01]
        months = pd.Series(values, index=pd.date_range("2024-01-31", periods=12, freq="ME"))

        winning, losing = figures.longest_streaks(months)

        assert (winning.months, winning.start, winning.end) == (3, "2024-01", "2024-03")
        assert winning.total == pytest.approx(1.01 * 1.02 * 1.01 - 1)
        assert (losing.months, losing.start, losing.end) == (4, "2024-08", "2024-11")

    def test_no_losing_month_has_no_losing_streak(self):
        winning, losing = figures.longest_streaks(STEADY)

        assert winning.months == 16 and losing is None


class TestMeans:

    def test_the_arithmetic_and_geometric_mean(self):
        returns = pd.Series([0.1, -0.05], index=DAYS[:2])

        result = figures.means(returns, 2)

        assert result["arithmetic"] == pytest.approx(0.025)
        assert result["geometric"] == pytest.approx((1.1 * 0.95) ** 0.5 - 1)
        assert result["arithmetic_annual"] == pytest.approx(0.05)
        assert result["geometric_annual"] == pytest.approx(1.1 * 0.95 - 1)


class TestThroughTheAnalytics:

    @pytest.fixture(scope="class")
    def analytics(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.analytics import PerformanceAnalytics
        from test_strategy_futures import index, run

        result = run(index())

        return result, PerformanceAnalytics(RunData.from_result(result))

    def test_since_inception_is_the_summary_s(self,
                                              analytics):
        result, figures_of = analytics
        inception = figures_of.trailing_returns()["inception"]
        summary = result.summary()

        assert inception.total == summary["total_return"]
        longer = inception.end > inception.start + pd.DateOffset(years=1)
        assert inception.annualised == (summary["annualised_return"] if longer else None)

    def test_every_figure_is_there(self,
                                   analytics):
        _, figures_of = analytics

        assert set(figures_of.trailing_returns()) == {
            "1M", "3M", "6M", "1Y", "3Y", "5Y", "10Y", "MTD", "QTD", "YTD", "inception"}
        assert set(figures_of.best_and_worst()) == set(figures.EXTREMES)
        assert set(figures_of.positive_shares()) == {"daily", "monthly", "annual"}
        assert figures_of.gains_and_losses("monthly").gains >= 0
        assert figures_of.means()["arithmetic"] is not None


class TestAnnualising:

    def test_three_years_and_more_are_annualised(self):
        long = pd.Series(0.0004, index=pd.bdate_range("2019-01-02", "2025-02-28"))
        trailing = figures.trailing_returns(long, PPY)

        assert trailing["3Y"].annualised == pytest.approx(
            (1 + trailing["3Y"].total) ** (PPY / trailing["3Y"].days) - 1)
        assert trailing["5Y"].annualised is not None
        assert trailing["1Y"].annualised is None and trailing["6M"].annualised is None
        assert all(window.annualised is None
                   for window in figures.returns_to_date(long, PPY).values())
