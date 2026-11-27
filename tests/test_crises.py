# tests/test_crises.py
"""BN-380: crises found on a market series, and how the fund held up in them.

A constructed market over 400 business days: it rises, falls 20% over ten
days (a crisis), recovers, then dips 10% (not one, at the 15% threshold). The
comparator is the market; the fund moves half as much, every day, so its
beta is 0.5 in and out of the crisis and its alpha against that beta is 0.
"""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.crises import (
    NAMED_CRISES,
    CrisisAnalytics,
    Episode,
    crisis_paths,
    market_crises,
)

DAYS = pd.bdate_range("2021-01-04", periods=400)
# A little noise each day, so the market varies inside the crisis too; the
# crisis's days still compound to exactly a 20% fall.
NOISE = np.random.default_rng(51).normal(0.0, 0.002, 10)
CRISIS = (1 + NOISE) * (0.8 / np.prod(1 + NOISE)) ** 0.1 - 1
DAILY = np.r_[0.0, np.full(49, 0.002), CRISIS, np.full(100, 0.003),
              np.full(5, 0.9 ** 0.2 - 1), np.full(235, 0.001)]
MARKET_RETURNS = pd.Series(DAILY, index=DAYS)
MARKET = 100.0 * (1 + MARKET_RETURNS).cumprod()
FUND = MARKET_RETURNS * 0.5
PEAK, TROUGH = DAYS[49], DAYS[59]


def analytics(**options) -> CrisisAnalytics:
    run = RunData.of_returns(FUND, 252, 0.0, {"index": MARKET})

    return CrisisAnalytics(run, **options)


class TestFindingCrises:

    def test_only_falls_beyond_the_threshold_are_crises(self):
        found = market_crises(MARKET, 0.15)

        assert [(episode.start, episode.end) for episode in found] == [(PEAK, TROUGH)]

    def test_a_lower_threshold_finds_the_dip_too(self):
        assert len(market_crises(MARKET, 0.05)) == 2

    def test_the_named_episodes_are_peak_before_trough(self):
        assert len(NAMED_CRISES) == 8
        assert all(episode.start < episode.end for episode in NAMED_CRISES)


class TestOneCrisis:

    @pytest.fixture(scope="class")
    def crisis(self):
        return analytics().crises()[0]

    def test_the_returns_and_outperformance(self,
                                            crisis):
        theirs = (1 + MARKET_RETURNS.loc[DAYS[50]:TROUGH]).prod() - 1
        ours = (1 + FUND.loc[DAYS[50]:TROUGH]).prod() - 1

        assert crisis.comparator == pytest.approx(-0.2)
        assert crisis.comparator == pytest.approx(theirs)
        assert crisis.fund == pytest.approx(ours)
        assert crisis.outperformance == pytest.approx(ours - theirs)
        assert crisis.days == 10 and not crisis.partial

    def test_the_drawdowns(self,
                           crisis):
        assert crisis.comparator_drawdown == pytest.approx(-0.2)
        assert crisis.fund_drawdown == pytest.approx(crisis.fund)

    def test_beta_and_alpha_against_the_run_s_beta(self,
                                                   crisis):
        assert crisis.beta == pytest.approx(0.5)
        assert crisis.alpha == pytest.approx(crisis.fund - 0.5 * crisis.comparator)

    def test_what_came_after_the_trough(self,
                                        crisis):
        until = TROUGH + pd.DateOffset(months=3)
        window = (DAYS > TROUGH) & (until >= DAYS)

        assert crisis.after["comparator 3m"] == pytest.approx(
            (1 + MARKET_RETURNS[window]).prod() - 1)
        assert crisis.after["fund 6m"] is not None

    def test_the_time_to_recover(self,
                                 crisis):
        level = (1 + FUND).cumprod()
        start = level.loc[PEAK]
        back = level.loc[DAYS[60]:]
        expected = int(np.flatnonzero(back.to_numpy() >= start)[0]) + 1

        assert crisis.recovery_days == expected


class TestAcrossCrises:

    def test_the_figures_over_every_crisis(self):
        summary = analytics(threshold=0.05).summary()
        crises = analytics(threshold=0.05).crises()

        assert summary["average_outperformance"] == pytest.approx(
            np.mean([crisis.outperformance for crisis in crises]))
        assert summary["share_outperformed"] == 1.0
        assert 0.0 < summary["capture"] < 1.0

    def test_the_pooled_alpha_needs_a_year_of_crisis_days(self):
        # BN-466: a rate, and these crises add up to a few weeks.
        from beacon.analysis.performance.settings import AnalysisSettings

        anywhere = AnalysisSettings(min_annualised_rate_window="0M")

        assert analytics(threshold=0.05).summary()["pooled_alpha"] is None
        assert analytics(threshold=0.05, settings=anywhere).summary()["pooled_alpha"] == (
            pytest.approx(0.0, abs=1e-12))

    def test_no_crisis_has_no_figures(self):
        summary = analytics(threshold=0.5).summary()

        assert summary == dict.fromkeys(("average_outperformance", "share_outperformed",
                                         "capture", "pooled_alpha"))


class TestNamedEpisodes:

    def test_the_user_s_episodes_are_used_instead(self):
        named = [Episode("A wobble", DAYS[150], DAYS[170]), Episode("Before the run",
                                                                     pd.Timestamp("2010-01-04"),
                                                                     pd.Timestamp("2010-06-01"))]

        crises = analytics(named=named).crises()

        assert [crisis.name for crisis in crises] == ["A wobble"]

    def test_one_the_run_covers_part_of_is_partial(self):
        named = [Episode("Straddling the end", DAYS[390], DAYS[-1] + pd.DateOffset(days=30))]

        crisis = analytics(named=named).crises()[0]

        assert crisis.partial and crisis.days == 9
        assert crisis.after["fund 3m"] is None and crisis.recovery_days is None


class TestTheMarket:

    def test_the_benchmark_is_the_market_unless_told(self):
        flat_index = pd.Series(100.0, index=DAYS)
        run = RunData.of_returns(FUND, 252, 0.0, {"index": flat_index, "benchmark": MARKET})

        assert len(CrisisAnalytics(run).episodes) == 1
        assert CrisisAnalytics(run, market="index").episodes == []

    def test_any_market_series_can_be_given(self):
        assert len(analytics(market=MARKET * 3).episodes) == 1


class TestStatus:
    """BN-431: each crisis says how much of it the run covered, and a named
    one the run saw none of keeps its row."""

    def test_covered_and_partial(self):
        named = [Episode("A wobble", DAYS[150], DAYS[170]),
                 Episode("Straddling the end", DAYS[390], DAYS[-1] + pd.DateOffset(days=30))]

        crises = analytics(named=named).crises()

        assert [crisis.status for crisis in crises] == ["covered", "partial"]

    def test_crises_the_run_saw_none_of(self):
        named = [Episode("Before", pd.Timestamp("2010-01-04"), pd.Timestamp("2010-06-01")),
                 Episode("A wobble", DAYS[150], DAYS[170]),
                 Episode("After", DAYS[-1] + pd.DateOffset(days=30),
                         DAYS[-1] + pd.DateOffset(days=90))]

        unseen = analytics(named=named).unseen()

        assert [(episode.name, status) for episode, status in unseen] == [
            ("Before", "before_run"), ("After", "after_run")]

    def test_the_recovery_date(self):
        crisis = analytics().crises()[0]
        back = DAYS[DAYS > TROUGH][crisis.recovery_days - 1]

        assert crisis.recovery_date == back


class TestPaths:
    """BN-431: each book's level through a crisis and on past it."""

    def paths(self,
              crisis=None):
        crisis = crisis or analytics().crises()[0]

        return crisis, crisis_paths([crisis], {"fund": FUND, "index": MARKET_RETURNS})

    def test_from_100_at_the_start(self):
        crisis, found = self.paths()
        fund = found[found["portfolio"] == "fund"].set_index("date")["level"]

        assert fund.iloc[0] == 100.0 and fund.index[0] == PEAK
        assert fund.loc[TROUGH] == pytest.approx(100.0 * (1 + crisis.fund))

    def test_the_phases(self):
        _, found = self.paths()
        index = found[found["portfolio"] == "index"]

        assert (index[index["date"] <= TROUGH]["phase"] == "crisis").all()
        assert (index[index["date"] > TROUGH]["phase"] == "after").all()

    def test_a_month_past_the_recovery_or_three_past_the_end(self):
        crisis, found = self.paths()
        last = found["date"].max()
        expected = max(TROUGH + pd.DateOffset(months=3),
                       crisis.recovery_date + pd.DateOffset(months=1))

        assert last == DAYS[DAYS.to_series() <= expected].max()

    def test_eighteen_months_at_most_and_never_past_the_run(self):
        crisis = replace(analytics().crises()[0], recovery_date=None)

        _, found = self.paths(crisis)

        assert found["date"].max() == DAYS[-1]

    def test_a_crisis_the_run_joined_late_starts_on_its_first_day(self):
        named = [Episode("Joined late", pd.Timestamp("2020-12-01"), DAYS[20])]
        crisis = analytics(named=named).crises()[0]

        _, found = self.paths(crisis)
        fund = found[found["portfolio"] == "fund"].set_index("date")["level"]

        assert fund.index[0] == DAYS[0] and fund.iloc[0] == 100.0
        assert fund.loc[DAYS[20]] == pytest.approx(100.0 * (1 + FUND.loc[DAYS[1]:DAYS[20]]).prod())


class TestThroughTheCatalogue:

    def test_a_run_with_no_crisis_has_an_empty_table(self):
        from beacon.analysis.reports.catalogue import CATALOGUE
        from beacon.analysis.reports.context import Context

        run = RunData.of_returns(FUND, 252, 0.0, {"index": pd.Series(100.0, index=DAYS)})

        assert CATALOGUE["crisis_returns"].compute(Context(run)).empty
        assert CATALOGUE["crisis_paths"].compute(Context(run)).empty
