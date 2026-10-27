# tests/test_performance_analytics.py
"""BN-371: a run's performance figures through `PerformanceAnalytics`, from
a finished run or a stored one, with the settings they are measured with.

The runs are the futures tests' three-name index and the flows tests' fund
with money arriving part way.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import formulas
from beacon.analysis.performance.analytics import PerformanceAnalytics
from beacon.analysis.performance.settings import AnalysisSettings


@pytest.fixture(scope="module")
def plain():
    from test_strategy_futures import index, run

    return run(index())


@pytest.fixture(scope="module")
def with_flows():
    from beacon.backtest import DatedFlows
    from test_backtest_flows import DAYS, holdings_mode, run

    return run(flows=DatedFlows({DAYS[60]: 5e6}), implementation=holdings_mode())


def stored(result,
           folder) -> RunData:
    RunData.from_result(result).write(folder)

    return RunData.from_record(folder)


class TestTheSummary:

    def test_it_is_the_backtest_s(self,
                                  plain):
        assert PerformanceAnalytics(RunData.from_result(plain)).summary() == plain.summary()

    def test_a_stored_run_gives_the_same_figures(self,
                                                 plain,
                                                 tmp_path):
        assert PerformanceAnalytics(stored(plain, tmp_path / "run")).summary() == plain.summary()

    def test_a_stored_run_with_flows_gives_the_same_figures(self,
                                                            with_flows,
                                                            tmp_path):
        figures = PerformanceAnalytics(stored(with_flows, tmp_path / "run")).summary()

        assert figures == with_flows.summary()
        assert figures["money_weighted_return"] is not None

    def test_a_run_stored_before_measures_from_its_nav(self,
                                                       plain,
                                                       tmp_path):
        data = stored(plain, tmp_path / "run")
        older = RunData(currency=data.currency, returns=data.returns, nav=data.nav,
                        units=data.units, initial_capital=data.initial_capital,
                        periods_per_year=data.periods_per_year,
                        risk_free_rate=data.risk_free_rate, comparators=data.comparators)

        assert PerformanceAnalytics(older).summary() == plain.summary()

    def test_the_summary_builds_none_of_the_run_s_tables(self,
                                                         plain,
                                                         monkeypatch):
        from beacon.analysis import data as module

        def refuse(result):
            raise AssertionError("the tables were built")

        monkeypatch.setattr(module, "run_panels", refuse)

        assert plain.summary()["total_return"] == pytest.approx(
            plain.trading_nav.iloc[-1] / plain.portfolio.initial_capital - 1)


class TestSettings:

    def test_the_rate_and_periods_can_be_changed(self,
                                                 plain):
        data = RunData.from_result(plain)
        base = PerformanceAnalytics(data).headline()
        changed = PerformanceAnalytics(data, AnalysisSettings(periods_per_year=12,
                                                              risk_free_rate=0.03)).headline()

        assert changed["total_return"] == base["total_return"]
        assert changed["volatility"] == formulas.volatility(data.returns, 12)
        assert changed["sharpe_ratio"] == formulas.sharpe_ratio(data.returns, 12, 0.03)

    def test_tracking_can_be_measured_against_another_comparator(self,
                                                                 plain):
        data = RunData.from_result(plain)
        levels = data.comparators["index"]
        renamed = RunData(**{**data.__dict__, "comparators": {"benchmark": levels}})
        analytics = PerformanceAnalytics(renamed, AnalysisSettings(tracked="benchmark"))

        assert analytics.tracking_error() == plain.summary()["tracking_error"]
        assert PerformanceAnalytics(renamed).tracking_error() is None

    def test_a_window_is_measured_from_the_close_before_it(self,
                                                           plain):
        data = RunData.from_result(plain)
        start, end = data.returns.index[100], data.returns.index[200]
        window = PerformanceAnalytics(data, AnalysisSettings(start=start, end=end))
        returns = data.returns.loc[start:end]

        figures = window.headline()

        assert len(window.returns) == 101
        assert figures["total_return"] == pytest.approx(float((1 + returns).prod() - 1))
        assert figures["max_drawdown"] == formulas.max_drawdown(formulas.path_of(returns))
        assert window.tracking_difference() == formulas.tracking_difference(
            returns, formulas.returns_from_start(data.comparators["index"]).loc[start:end])

    def test_a_window_with_no_days_has_no_figures(self,
                                                  plain):
        data = RunData.from_result(plain)
        empty = PerformanceAnalytics(data, AnalysisSettings(start=pd.Timestamp("1990-01-01"),
                                                            end=pd.Timestamp("1990-02-01")))

        assert empty.headline()["total_return"] == 0.0
        assert empty.hit_rate_monthly() is None
        assert empty.tracking_difference() is None
