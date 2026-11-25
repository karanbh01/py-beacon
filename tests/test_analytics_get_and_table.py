# tests/test_analytics_get_and_table.py
"""BN-444 and BN-445: any analytic by its id, and single-figure analytics side
by side.

The run is test_strategy_futures' eight-name market-cap index held physically
through 2024, with its total-return version as the benchmark.
"""
import pandas as pd
import pytest

from beacon.analysis import Statistic
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.context import Context
from beacon.backtest import Backtest


@pytest.fixture(scope="module")
def analytics():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(
        index(), start=START, end=END).analytics


class TestGet:
    def test_a_member_and_its_method_agree(self,
                                           analytics):
        found = analytics.get(Statistic.TOTAL_RETURN, by="quarter")

        pd.testing.assert_series_equal(
            found, analytics.portfolio.performance.total_return(by="quarter"))

    def test_an_id_does_too(self,
                            analytics):
        assert analytics.get("tracking_error", against="benchmark") == (
            analytics.portfolio.relative.tracking_error(against="benchmark"))

    def test_an_instrument_statistic(self,
                                     analytics):
        found = analytics.get(Statistic.MEMBER_RISK_RETURN)
        today = CATALOGUE["member_risk_return"].compute(
            Context(analytics.run_data, None, analytics._source)).table

        assert found.equals(today)

    def test_an_unknown_id(self,
                           analytics):
        with pytest.raises(ValueError, match="Unknown statistic 'sharp_ratio'"):
            analytics.get("sharp_ratio")

    def test_a_keyword_it_does_not_take(self,
                                        analytics):
        with pytest.raises(TypeError, match="tracking_error takes"):
            analytics.get(Statistic.TRACKING_ERROR, of="index")


class TestTable:
    STATISTICS = (Statistic.TOTAL_RETURN, Statistic.VOLATILITY, Statistic.SHARPE_RATIO)

    def test_a_row_a_statistic_a_column_a_window(self,
                                                 analytics):
        found = analytics.table(self.STATISTICS, trailing=["3M", "6M", "YTD"])

        assert list(found.index) == ["total_return", "volatility", "sharpe_ratio"]
        assert list(found.columns) == ["3M", "6M", "YTD"]
        assert found.loc["volatility", "6M"] == analytics.portfolio.risk.volatility(
            trailing=["6M"])["6M"]

    def test_a_column_a_period(self,
                               analytics):
        found = analytics.table(["total_return", "volatility"], by="quarter")

        pd.testing.assert_series_equal(found.loc["total_return"],
                                       analytics.portfolio.performance.total_return(
                                           by="quarter"),
                                       check_names=False)

    def test_every_portfolio_and_comparator(self,
                                            analytics):
        found = analytics.table(["total_return", "tracking_error"], of="all")

        assert list(found.columns) == ["fund", "index", "benchmark", "active"]
        assert found.loc["tracking_error", "benchmark"] == (
            analytics.portfolio.relative.tracking_error(against="benchmark"))
        assert pd.isna(found.loc["tracking_error", "fund"])

    def test_a_window_and_every_portfolio(self,
                                          analytics):
        found = analytics.table(["total_return", "tracking_error"], trailing=["3M", "YTD"],
                                of="all")

        assert list(found.index.names) == ["statistic", "window"]
        assert found.loc[("total_return", "YTD"), "index"] == (
            analytics.portfolio.performance.total_return(of="index"))

    def test_bare_each_is_the_fund_s(self,
                                     analytics):
        found = analytics.table(["total_return", "sharpe_ratio"])

        assert list(found.columns) == ["value"]
        assert found.loc["sharpe_ratio", "value"] == (
            analytics.portfolio.performance.sharpe_ratio())

    def test_a_statistic_of_several_figures_is_refused(self,
                                                       analytics):
        with pytest.raises(ValueError, match="value_at_risk gives more than one"):
            analytics.table(["total_return", "value_at_risk"])

    def test_a_rolling_window_is_refused(self,
                                         analytics):
        with pytest.raises(TypeError, match="does not take rolling"):
            analytics.table(["total_return"], rolling=63)

    def test_a_period_a_statistic_does_not_take(self,
                                                analytics):
        with pytest.raises(TypeError, match="r_squared takes"):
            analytics.table(["total_return", "r_squared"], by="year")
