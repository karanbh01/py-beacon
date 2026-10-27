# tests/test_regimes.py
"""BN-381: market regimes found on a constructed market, and the fund in each.

Over 500 business days the market is quiet for its first half and noisy for
its second, falls 25% (a bear market), later 12% (a correction) and 5%
(neither). The fund moves half as much as the comparator, which is the
market, so its beta in every regime is 0.5 and its alpha 0.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import formulas
from beacon.analysis.performance.regimes import (
    REGIMES,
    RegimeAnalytics,
    calm_and_volatile,
    falls,
    regime_of,
    up_and_down_months,
)

DAYS = pd.bdate_range("2022-01-03", periods=500)
RANDOM = np.random.default_rng(61)
NOISE = np.r_[RANDOM.normal(0, 0.002, 250), RANDOM.normal(0, 0.02, 250)]
TREND = np.full(500, 0.0008)
TREND[0] = 0.0
TREND[60:70] = 0.75 ** 0.1 - 1
TREND[200:205] = 0.88 ** 0.2 - 1
TREND[400:403] = 0.95 ** (1 / 3) - 1
# Rallies after the two falls, so each recovers before the next begins.
TREND[70:110] = 0.008
TREND[205:230] = 0.006
NOISE[[0, *range(60, 70), *range(200, 205), *range(400, 403)]] = 0.0
MARKET_RETURNS = pd.Series(TREND + NOISE, index=DAYS)
MARKET = 100.0 * (1 + MARKET_RETURNS).cumprod()
FUND = MARKET_RETURNS * 0.5


class TestFindingRegimes:

    def test_up_and_down_months(self):
        up, down = up_and_down_months(MARKET_RETURNS)
        monthly = formulas.period_returns(MARKET_RETURNS, formulas.MONTHLY)
        march = DAYS[(DAYS.year == 2022) & (DAYS.month == 3)]

        assert bool(up[march].all()) == bool(monthly["2022-03"] > 0)
        assert not (up & down).any()

    def test_bear_markets_and_corrections(self):
        bear = falls(MARKET, 0.20)
        correction = falls(MARKET, 0.10, 0.20)

        assert bear[DAYS[60]:DAYS[69]].all() and bear.sum() >= 10
        assert correction[DAYS[200]:DAYS[204]].all()
        assert not correction[DAYS[400]:DAYS[402]].any()
        assert not (bear & correction).any()

    def test_calm_and_volatile_fifths(self):
        calm, volatile = calm_and_volatile(MARKET_RETURNS)

        measured = len(DAYS) - 62

        assert calm.sum() >= 0.19 * measured and volatile.sum() >= 0.19 * measured
        assert calm[DAYS[300]:].sum() == 0
        assert volatile[:DAYS[250]].sum() == 0
        assert not calm.iloc[:62].any()


class TestARegime:

    def test_the_fund_against_its_comparator(self):
        days = pd.Series(DAYS[300] <= DAYS, index=DAYS)

        regime = regime_of(FUND, MARKET_RETURNS, days, 0.0, 252)
        chosen = MARKET_RETURNS.loc[DAYS[300]:]

        assert regime.days == 200
        assert regime.comparator == pytest.approx((1 + chosen).prod() - 1)
        assert regime.difference == pytest.approx(regime.fund - regime.comparator)
        assert regime.comparator_annual == pytest.approx(chosen.mean() * 252)
        assert regime.beta == pytest.approx(0.5)
        assert regime.alpha == pytest.approx(0.0, abs=1e-12)

    def test_no_days_no_regime(self):
        assert regime_of(FUND, MARKET_RETURNS, pd.Series(False, index=DAYS), 0.0, 252) is None


class TestTheAnalytics:

    def test_every_regime_against_the_market(self):
        run = RunData.of_returns(FUND, 252, 0.0, {"index": MARKET})

        regimes = RegimeAnalytics(run).regimes()

        assert set(regimes) == set(REGIMES)
        assert regimes["bear markets"].beta == pytest.approx(0.5)
        assert regimes["down months"].comparator < 0 < regimes["up months"].comparator

    def test_a_regime_the_run_never_saw_is_none(self):
        calm = 100.0 * (1 + pd.Series(0.0005, index=DAYS)).cumprod()
        run = RunData.of_returns(FUND, 252, 0.0, {"index": MARKET, "benchmark": calm})

        regimes = RegimeAnalytics(run).regimes()

        assert regimes["bear markets"] is None and regimes["down months"] is None
