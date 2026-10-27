# tests/test_drawdowns.py
"""BN-374: a run's drawdowns, hand-computed.

A level over ten business days with three drawdowns: 110 to 88 and back,
121 to 115 and back, and 130 to 117, still open on the last day.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance.drawdowns import drawdown_figures, episodes

DAYS = pd.bdate_range("2024-03-04", periods=10)
LEVELS = pd.Series([100.0, 110, 99, 88, 110, 121, 115, 121, 130, 117], index=DAYS)


class TestEpisodes:

    def test_each_fall_from_a_peak_is_found(self):
        found = episodes(LEVELS)

        assert [(d.peak, d.trough, d.recovery) for d in found] == [
            (DAYS[1], DAYS[3], DAYS[4]), (DAYS[5], DAYS[6], DAYS[7]), (DAYS[8], DAYS[9], None)]

    def test_depths_and_lengths(self):
        first, second, third = episodes(LEVELS)

        assert first.depth == pytest.approx(88 / 110 - 1)
        assert (first.to_trough, first.to_recover, first.length) == (2, 1, 3)
        assert second.depth == pytest.approx(115 / 121 - 1)
        assert (second.to_trough, second.to_recover, second.length) == (1, 1, 2)
        assert (third.to_trough, third.to_recover, third.length) == (1, None, 1)

    def test_a_level_that_only_rises_has_none(self):
        assert episodes(pd.Series([1.0, 1.1, 1.2], index=DAYS[:3])) == []
        assert episodes(LEVELS.iloc[:1]) == []

    def test_a_level_back_at_its_peak_has_recovered(self):
        found = episodes(pd.Series([1.0, 0.9, 1.0, 0.95], index=DAYS[:4]))

        assert found[0].recovery == DAYS[2]
        assert found[1].peak == DAYS[2] and found[1].recovery is None


class TestFigures:

    @pytest.fixture
    def figures(self):
        return drawdown_figures(LEVELS, total_return=0.17, worst=2)

    def test_the_worst_are_listed_deepest_first(self,
                                                figures):
        assert [d.depth for d in figures.worst] == pytest.approx([-0.2, 117 / 130 - 1])

    def test_the_current_drawdown(self,
                                  figures):
        assert figures.current == pytest.approx(117 / 130 - 1)
        assert figures.current_drawdown.peak == DAYS[8]

    def test_the_averages_and_time_below_a_peak(self,
                                                figures):
        assert figures.average_depth == pytest.approx(
            np.mean([88 / 110 - 1, 115 / 121 - 1, 117 / 130 - 1]))
        assert figures.average_length == pytest.approx(2.0)
        assert figures.time_below_peak == pytest.approx(0.4)
        assert figures.longest_below_peak == 3

    def test_recovery_ulcer_and_pain(self,
                                     figures):
        daily = np.array([0, 0, 99 / 110 - 1, 88 / 110 - 1, 0, 0, 115 / 121 - 1, 0, 0,
                          117 / 130 - 1])

        assert figures.recovery_factor == pytest.approx(0.17 / 0.2)
        assert figures.ulcer_index == pytest.approx(np.sqrt(np.mean(daily ** 2)))
        assert figures.pain_index == pytest.approx(np.mean(np.abs(daily)))

    def test_at_a_high_there_is_no_current_drawdown(self):
        figures = drawdown_figures(LEVELS.iloc[:9], total_return=0.3)

        assert figures.current == 0.0 and figures.current_drawdown is None

    def test_without_a_drawdown(self):
        figures = drawdown_figures(pd.Series([1.0, 1.1], index=DAYS[:2]), total_return=0.1)

        assert figures.worst == [] and figures.recovery_factor is None
        assert figures.average_depth == 0.0 and figures.longest_below_peak == 0


class TestThroughTheAnalytics:

    @pytest.fixture(scope="class")
    def run(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.analytics import PerformanceAnalytics
        from test_strategy_futures import index, run

        result = run(index())

        return result, PerformanceAnalytics(RunData.from_result(result))

    def test_the_deepest_is_the_summary_s_maximum_drawdown(self,
                                                           run):
        result, analytics = run

        assert analytics.drawdowns().worst[0].depth == result.summary()["max_drawdown"]

    def test_the_level_starts_the_day_before_the_first_return(self,
                                                              run):
        result, analytics = run
        levels = analytics.levels()

        assert levels.iloc[0] == result.portfolio.initial_capital
        assert levels.index[0] == result.get_returns().index[0] - pd.offsets.BDay(1)
        assert analytics.drawdown_series().min() == result.summary()["max_drawdown"]
