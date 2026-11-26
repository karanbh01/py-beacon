# tests/test_result_deprecations.py
"""BN-461: the result's 0.10.0 paths warn, naming their replacement, and go
in 0.12.0. Each is called once and checked against its new path.

The runs are test_strategy_futures' eight-name index held through 2024,
test_backtest_flows' two names with money arriving and leaving, and the same
as an ETF.
"""
import re

import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.backtest.result import MOVED, MOVED_METHODS

ATTRIBUTES = ("modelling_assumptions", "analysis_request", "analysis_timings", "analysis_block",
              "run_data", "portfolio_analytics", "risk", "attribution")


@pytest.fixture(scope="module")
def result():
    from test_strategy_futures import index, run

    return run(index(), transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def flowed():
    from beacon.backtest import DatedFlows
    from test_backtest_flows import DAYS, run

    return run(flows=DatedFlows({DAYS[10]: 2e5, DAYS[30]: -1e5}))


def deprecated(result,
               name: str,
               *call):
    """*name* of *result*, called when *call* is given, checking it warns."""
    path = MOVED_METHODS[name][0] if call else MOVED[name][0]
    shown = f"{name}()" if call else name

    expected = (f"BacktestResult.{shown} is deprecated and will be removed in 0.12.0; "
                f"use result.{path.split('(')[0]}")

    with pytest.warns(DeprecationWarning, match=re.escape(expected)):
        found = getattr(result, name)

        return found() if call else found


class TestTheAttributes:

    def test_each_warns_and_reads_its_new_home(self,
                                               result):
        analytics = result.analytics

        assert deprecated(result, "modelling_assumptions") is result.assumptions
        assert deprecated(result, "analysis_request") is analytics.request
        assert deprecated(result, "analysis_timings") == analytics.timings
        assert deprecated(result, "run_data") is analytics.run_data
        assert deprecated(result, "analysis_block") is None

    def test_the_portfolio_s_analytics(self,
                                       result):
        assert type(deprecated(result, "portfolio_analytics")).__name__ == "PortfolioAnalytics"
        assert type(deprecated(result, "attribution")).__name__ == "Attribution"
        assert deprecated(result, "risk") is not None

    def test_all_are_covered(self):
        assert set(ATTRIBUTES) <= set(MOVED)


class TestTheMethods:

    def test_returns_and_tracking(self,
                                  result):
        portfolio = result.analytics.portfolio

        pd.testing.assert_series_equal(deprecated(result, "get_returns", ()),
                                       portfolio.performance.returns(), check_names=False)
        yearly = deprecated(result, "get_annual_returns", ())
        assert yearly.tolist() == pytest.approx(
            portfolio.performance.total_return(by="year").tolist())
        assert deprecated(result, "get_tracking_error", ()) == pytest.approx(
            portfolio.relative.tracking_error())
        assert deprecated(result, "get_tracking_difference", ()) == pytest.approx(
            portfolio.relative.tracking_difference())
        assert deprecated(result, "information_ratio", ()) == pytest.approx(
            portfolio.relative.information_ratio())

    def test_the_levels(self,
                        result,
                        flowed):
        pd.testing.assert_series_equal(deprecated(result, "performance_levels", ()),
                                       result.fund.trading_nav)
        pd.testing.assert_series_equal(deprecated(flowed, "performance_levels", ()),
                                       flowed.fund.nav_per_unit)

    def test_the_money_weighted_return(self,
                                       flowed):
        assert deprecated(flowed, "money_weighted_return", ()) == pytest.approx(
            flowed.analytics.portfolio.performance.money_weighted_return())

    def test_an_etf_s_trading(self):
        from test_backtest_etf import etf
        from test_backtest_flows import run

        listed = run(vehicle=etf())
        vehicle = listed.analytics.portfolio.vehicle

        assert deprecated(listed, "market_return", ()) == pytest.approx(vehicle.market_return())
        assert deprecated(listed, "market_summary", ()) == pytest.approx(vehicle.market_summary())

    def test_the_active_figures(self,
                                result):
        portfolio = result.analytics.portfolio
        attributed = deprecated(result, "active_attribution", ())
        theirs = portfolio.attribution.active_attribution()

        assert deprecated(result, "active_share", ()).empty
        assert attributed.total_return == pytest.approx(
            theirs.loc[theirs["level"] == "name", "contribution"].sum(), abs=1e-6)

    def test_all_are_covered(self):
        assert set(MOVED_METHODS) == {
            "get_returns", "get_annual_returns", "get_tracking_error", "get_tracking_difference",
            "information_ratio", "money_weighted_return", "active_share", "active_attribution",
            "market_return", "market_summary", "performance_levels"}


class TestWhatStays:

    def test_summary_asset_against_and_plot_do_not_warn(self,
                                                        result):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            assert result.summary()["total_return"] == pytest.approx(
                result.analytics.portfolio.performance.total_return())
            _ = result.against(result.fund.trading_nav), result.plot
            assert isinstance(result.analytics, Analytics)
