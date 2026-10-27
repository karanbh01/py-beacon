# tests/test_crises.py
"""BN-380: crises found on a market series, and how the fund held up in them.

A constructed market over 400 business days: it rises, falls 20% over ten
days (a crisis), recovers, then dips 10% (not one, at the 15% threshold). The
comparator is the market; the fund moves half as much, every day, so its
beta is 0.5 in and out of the crisis and its alpha against that beta is 0.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.crises import (
    NAMED_CRISES,
    CrisisAnalytics,
    Episode,
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
        assert summary["pooled_alpha"] == pytest.approx(0.0, abs=1e-12)
        assert 0.0 < summary["capture"] < 1.0

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
