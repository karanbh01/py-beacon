# tests/test_factor_analytics.py
"""BN-387: factor returns and returns-based betas, recovered on constructed
returns with known loadings.

Thirty names over 300 sessions, rebalanced every 60, each with a fixed market
cap and two fixed characteristics. Each name's return is the market's plus
its exposure to each factor times that factor's return, with no noise:

    r_i = f_market + z_i1 f_1 + z_i2 f_2

so the cross-sectional regression recovers the factor returns exactly, and a
book's betas are its holdings' exposures, the market's its total weight.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, NAME_RETURNS, REFERENCE
from beacon.analysis.risk.factors import ACTIVE, ALPHA, FactorAnalytics
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.strategy import Factor
from beacon.strategy.base import StrategyContext
from beacon.strategy.signals import FunctionSignal

RANDOM = np.random.default_rng(11)
DAYS = pd.bdate_range("2023-01-02", periods=300)
REBALANCES = list(DAYS[::60])
NAMES = [f"N{i:02d}" for i in range(30)]
CAPS = dict(zip(NAMES, RANDOM.uniform(1.0, 50.0, len(NAMES)), strict=True))
FIRST = dict(zip(NAMES, RANDOM.normal(0.0, 1.0, len(NAMES)), strict=True))
SECOND = dict(zip(NAMES, RANDOM.normal(0.0, 1.0, len(NAMES)), strict=True))
TRUE = pd.DataFrame({"market": RANDOM.normal(0.0003, 0.01, len(DAYS)),
                     "first": RANDOM.normal(0.0, 0.004, len(DAYS)),
                     "second": RANDOM.normal(0.0, 0.003, len(DAYS))}, index=DAYS)
FUND = pd.Series(RANDOM.uniform(0.5, 1.5, len(NAMES)), index=NAMES)
FUND = FUND / FUND.sum()
EQUAL = pd.Series(1.0 / len(NAMES), index=NAMES)


def factors() -> list[Factor]:
    return [Factor("first", FunctionSignal(lambda name, date, fetched: FIRST[name])),
            Factor("second", FunctionSignal(lambda name, date, fetched: SECOND[name]))]


def source() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": 10.0,
             "SHARES_OUTSTANDING": CAPS[name] * 1e5} for name in NAMES for day in DAYS]
    rows += [{"IDENTIFIER": "FIRST_INDEX", "DATE": day, "CLOSE": level}
             for day, level in (100.0 * (1.0 + TRUE["first"]).cumprod()).items()]
    reference = pd.DataFrame([{"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"}
                              for name in [*NAMES, "FIRST_INDEX"]])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


SOURCE = source()


def exposures() -> pd.DataFrame:
    """The factors' scores, which do not move: every rebalance's."""
    context = StrategyContext(SOURCE, "USD")

    return pd.DataFrame({factor.name: factor.exposures(NAMES, DAYS[0], context)
                         for factor in factors()})


Z = exposures()
NAME_RETURNS_TABLE = pd.DataFrame(
    np.outer(TRUE["market"], np.ones(len(NAMES)))
    + np.outer(TRUE["first"], Z["first"]) + np.outer(TRUE["second"], Z["second"]),
    index=DAYS, columns=NAMES)
NAME_RETURNS_TABLE.iloc[0] = 0.0


def book(weights: pd.Series) -> pd.Series:
    returns = NAME_RETURNS_TABLE @ weights
    returns.iloc[0] = 0.0

    return returns


def run_data() -> RunData:
    fund, index = book(FUND), book(EQUAL)
    reference = pd.DataFrame({"IDENTIFIER": [name for _ in REBALANCES for name in NAMES]},
                             index=pd.DatetimeIndex([day for day in REBALANCES for _ in NAMES]))
    tables = {NAME_RETURNS: NAME_RETURNS_TABLE, REFERENCE: reference,
              FUND_WEIGHTS: pd.DataFrame([FUND] * len(DAYS), index=DAYS),
              INDEX_WEIGHTS: pd.DataFrame([EQUAL] * len(DAYS), index=DAYS)}

    return RunData(currency="USD", returns=fund, nav=1e6 * (1.0 + fund).cumprod(),
                   units=fund * 0 + 1, initial_capital=1e6, periods_per_year=252,
                   risk_free_rate=0.0, tables=tables,
                   comparators={"index": 100.0 * (1.0 + index).cumprod()})


@pytest.fixture(scope="module")
def analytics():
    return FactorAnalytics(run_data(), SOURCE, factors=factors())


class TestExposures:

    def test_each_rebalance_s_are_the_factors_scores(self,
                                                     analytics):
        found = analytics.exposures(REBALANCES[1])

        pd.testing.assert_frame_equal(found.loc[NAMES], Z.loc[NAMES])

    def test_a_day_that_is_not_a_rebalance_is_refused(self,
                                                      analytics):
        with pytest.raises(KeyError, match="rebalances"):
            analytics.exposures(DAYS[5])


class TestFactorReturns:

    def test_they_are_recovered_exactly(self,
                                        analytics):
        found = analytics.factor_returns()

        assert found.index[0] == DAYS[1]
        assert found.to_numpy() == pytest.approx(TRUE.loc[found.index].to_numpy(), abs=1e-12)

    def test_a_factor_may_take_its_own_series(self):
        named = [Factor("first", FunctionSignal(lambda name, date, fetched: FIRST[name]),
                        returns="FIRST_INDEX"), factors()[1]]
        found = FactorAnalytics(run_data(), SOURCE, factors=named).factor_returns()

        assert found["first"].to_numpy() == pytest.approx(
            TRUE["first"].loc[found.index].to_numpy())


class TestBetas:

    def test_the_fund_s_betas_are_its_holdings_exposures(self,
                                                         analytics):
        fit = analytics.regression()
        held = Z.T @ FUND

        assert fit.betas["market"] == pytest.approx(1.0)
        assert fit.betas[["first", "second"]].to_numpy() == pytest.approx(held.to_numpy())
        assert fit.r_squared == pytest.approx(1.0)
        assert fit.coefficients.loc[ALPHA, "coefficient"] == pytest.approx(0.0, abs=1e-12)

    def test_the_active_betas_are_the_active_exposures(self,
                                                       analytics):
        fit = analytics.regression(ACTIVE)
        active = Z.T @ (FUND - EQUAL)

        assert fit.betas["market"] == pytest.approx(0.0, abs=1e-9)
        assert fit.betas[["first", "second"]].to_numpy() == pytest.approx(active.to_numpy())

    def test_the_index_s(self,
                         analytics):
        assert analytics.regression("index").betas[["first", "second"]].to_numpy() == (
            pytest.approx((Z.T @ EQUAL).to_numpy()))

    def test_with_noise_they_are_near_and_have_t_statistics(self):
        noisy = run_data()
        tables = dict(noisy.tables)
        noise = RANDOM.normal(0.0, 0.002, (len(DAYS), len(NAMES)))
        tables[NAME_RETURNS] = NAME_RETURNS_TABLE + noise
        fund = (tables[NAME_RETURNS] @ FUND).where(lambda found: found.index != DAYS[0], 0.0)
        noisy = RunData(currency="USD", returns=fund, nav=noisy.nav, units=noisy.units,
                        initial_capital=1e6, periods_per_year=252, risk_free_rate=0.0,
                        tables=tables, comparators=noisy.comparators)

        fit = FactorAnalytics(noisy, SOURCE, factors=factors()).regression()

        assert fit.betas["market"] == pytest.approx(1.0, abs=0.05)
        assert abs(fit.coefficients.loc["market", "t_stat"]) > 20

    def test_rolling(self,
                     analytics):
        rolling = analytics.rolling_betas(window=100)

        assert len(rolling) == len(analytics.factor_returns()) - 99
        assert rolling["market"].to_numpy() == pytest.approx(1.0)


class TestRefused:

    def test_an_unknown_book(self,
                             analytics):
        with pytest.raises(KeyError, match="benchmark"):
            analytics.regression("benchmark")

    @pytest.mark.parametrize("names", [["value", "value"], ["market"]])
    def test_factors_need_distinct_names(self,
                                         names):
        with pytest.raises(ValueError, match="distinct"):
            FactorAnalytics(run_data(), SOURCE,
                            factors=[Factor(name, FunctionSignal(lambda *_: 1.0))
                                     for name in names])

    def test_too_few_days(self):
        short = run_data()
        tables = dict(short.tables)
        tables[NAME_RETURNS] = NAME_RETURNS_TABLE.iloc[:10]

        with pytest.raises(CalculationError, match="fewer than"):
            FactorAnalytics(RunData(currency="USD", returns=short.returns, nav=short.nav,
                                    units=short.units, initial_capital=1e6,
                                    periods_per_year=252, risk_free_rate=0.0, tables=tables),
                            SOURCE, factors=factors()).regression()
