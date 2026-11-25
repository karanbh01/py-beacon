# tests/test_analytics_instrument.py
"""BN-452: `result.analytics.instrument`, a figure for each name, each the
fund's formula on that name's returns.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.panels import NAME_RETURNS
from beacon.analysis.performance import formulas
from beacon.backtest import Backtest

NAME = "N1"


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def instrument(result):
    return result.analytics.instrument


@pytest.fixture(scope="module")
def mine(result):
    """The name's daily returns over the run's days, and the index's."""
    days = result.run_data.returns.index
    returns = result.run_data.tables[NAME_RETURNS][NAME]
    returns = returns.loc[(returns.index >= days[0]) & (returns.index <= days[-1])]
    theirs = result.analytics._context.book("index").returns.reindex(returns.index)
    both = returns.notna() & theirs.notna()

    return returns[both], theirs[both]


def regression(ours: pd.Series,
               theirs: pd.Series) -> tuple[float, float]:
    """The beta and the annualised residual spread, written out."""
    beta = float(np.cov(ours, theirs, ddof=1)[0, 1] / np.var(theirs, ddof=1))
    residuals = ours - (ours.mean() - beta * theirs.mean()) - beta * theirs

    return beta, float(np.sqrt(np.sum(residuals ** 2) / (len(ours) - 2) * 252))


class TestEachIsTheFormulaOnTheName:
    def test_its_own_figures(self,
                             instrument,
                             mine):
        ours, _ = mine

        assert instrument.total_return()[NAME] == pytest.approx(
            formulas.compounded_return(ours), rel=1e-12)
        assert instrument.volatility()[NAME] == pytest.approx(
            formulas.volatility(ours, 252), rel=1e-12)
        assert instrument.max_drawdown()[NAME] == pytest.approx(
            formulas.max_drawdown(formulas.path_of(ours)), rel=1e-12)

    def test_its_figures_against_the_index(self,
                                           instrument,
                                           mine):
        ours, theirs = mine
        beta, residual = regression(ours, theirs)

        assert instrument.beta()[NAME] == pytest.approx(beta, rel=1e-9)
        assert instrument.correlation()[NAME] == pytest.approx(
            float(np.corrcoef(ours, theirs)[0, 1]), rel=1e-9)
        assert instrument.tracking_error()[NAME] == pytest.approx(
            formulas.tracking_error(ours - theirs, 252), rel=1e-9)
        assert instrument.residual_risk()[NAME] == pytest.approx(residual, rel=1e-9)

    def test_it_says_its_basis(self,
                               instrument):
        assert instrument.volatility().attrs == {"basis": "daily"}
        assert instrument.beta(against="benchmark").attrs == {"basis": "daily",
                                                               "against": "benchmark"}


class TestTheForms:
    def test_some_names(self,
                        instrument):
        found = instrument.volatility(names=["N0", NAME])

        assert list(found.index) == ["N0", NAME]

        with pytest.raises(ValueError, match="The run measured no ZZZ"):
            instrument.volatility(names="ZZZ")

    def test_each_period(self,
                         instrument,
                         mine):
        ours, theirs = mine
        quarters = instrument.beta(by="quarter")
        inside = ours.index.to_period("Q").astype(str) == "2024Q2"

        assert list(quarters.index) == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
        assert quarters.loc["2024Q2", NAME] == pytest.approx(
            regression(ours[inside], theirs[inside])[0], rel=1e-9)

    def test_rolling_windows(self,
                             instrument,
                             mine):
        ours, theirs = mine
        rolled = instrument.tracking_error(rolling=63)
        last = rolled.index[-1]

        assert rolled.loc[last, NAME] == pytest.approx(
            formulas.tracking_error((ours - theirs).iloc[-63:], 252), rel=1e-9)
        assert instrument.beta(rolling=63).iloc[-1][NAME] == pytest.approx(
            regression(ours.iloc[-63:], theirs.iloc[-63:])[0], rel=1e-6)
        assert instrument.residual_risk(rolling=63).iloc[-1][NAME] == pytest.approx(
            regression(ours.iloc[-63:], theirs.iloc[-63:])[1], rel=1e-6)

    def test_a_rolling_drawdown_ends_each_month(self,
                                                instrument,
                                                mine):
        ours, _ = mine
        rolled = instrument.max_drawdown(rolling="3M")

        assert len(rolled) <= 12
        assert rolled.iloc[-1][NAME] == pytest.approx(
            formulas.max_drawdown(formulas.path_of(ours.iloc[-63:])), rel=1e-12)

    def test_weekly(self,
                    instrument,
                    mine):
        ours, _ = mine
        weeks = (1.0 + ours).resample("W-FRI").prod(min_count=1) - 1.0

        assert instrument.volatility(frequency="weekly")[NAME] == pytest.approx(
            formulas.volatility(weeks, 52), rel=1e-12)
        assert instrument.volatility(frequency="weekly").attrs["basis"] == "weekly"

    def test_what_does_not_combine(self,
                                   instrument):
        with pytest.raises(ValueError, match="one of by and rolling"):
            instrument.volatility(by="year", rolling=63)

        with pytest.raises(ValueError, match="frequency is"):
            instrument.volatility(frequency="hourly")


class TestTheStatisticsANameAtATime:
    def test_member_risk_return_is_the_analysis_s(self,
                                                  result,
                                                  instrument):
        from beacon.analysis.reports.catalogue import CATALOGUE

        today = CATALOGUE["member_risk_return"].compute(result.analytics._context).table

        assert instrument.member_risk_return().equals(today)

    def test_its_methods(self,
                         instrument):
        assert set(instrument.extras) == {"total_return", "volatility", "max_drawdown", "beta",
                                          "correlation", "tracking_error", "residual_risk",
                                          "average_weight", "days_to_sell", "traded_value"}
        assert repr(instrument).startswith("InstrumentLevel(member_risk_return, style_positions")


class TestWeightsAndLiquidity:
    """BN-453: test_liquidity_analytics' fund of two names worked out by
    hand, read back from its data as a stored run is."""

    @pytest.fixture(scope="class")
    def data(self):
        from beacon.analysis.data import RunData
        from test_backtest_flows import run

        return RunData.from_result(run(capital=2e8))

    @pytest.fixture(scope="class")
    def names(self,
              data):
        from beacon.analysis.analytics import Analytics

        return Analytics(data).instrument

    def test_average_weights(self,
                             data,
                             names):
        from beacon.analysis.panels import FUND_WEIGHTS

        weights = data.tables[FUND_WEIGHTS]
        days = data.returns.index
        held = weights.loc[(weights.index >= days[0]) & (weights.index <= days[-1])]

        pd.testing.assert_series_equal(names.average_weight(), held.fillna(0.0).mean(),
                                       check_names=False)
        assert "fund" in names.average_weight(of="all").columns

    def test_a_portfolio_it_kept_no_weights_for(self,
                                               names):
        with pytest.raises(ValueError, match="no weights for 'benchmark'"):
            names.average_weight(of="benchmark")

    def test_days_to_sell(self,
                          data,
                          names):
        from beacon.analysis.risk.liquidity import LiquidityAnalytics

        liquidity = LiquidityAnalytics(data)
        dates = liquidity.dates()

        pd.testing.assert_frame_equal(names.days_to_sell(), liquidity.days_to_sell(dates[-1]))
        assert names.days_to_sell(as_of=dates[0]).attrs["date"] == dates[0]
        assert list(names.days_to_sell(names=["GROW"]).index) == ["GROW"]

    def test_traded_value(self,
                          data,
                          names):
        from beacon.analysis.risk.liquidity import LiquidityAnalytics

        pd.testing.assert_frame_equal(names.traded_value(),
                                      LiquidityAnalytics(data).traded_values())
