# tests/test_analytics_groupings.py
"""BN-446: tables split by groups of their own (``by="reason"``,
``by="sector"``, a period of attribution or concentration) and the monthly
returns as a calendar grid, each the calculator method it replaces.

The runs are test_strategy_futures' eight-name index held physically through
2024 with its total-return version as the benchmark, test_run_attribution's
consistent run for Brinson, test_risk_contributions_over_time's thirty names
in three sectors for the risk model, and test_backtest_flows' ETF with money
arriving and leaving.
"""
import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.groupings import CALCULATIONS
from beacon.analysis.reports.catalogue import CATALOGUE
from beacon.analysis.reports.catalogue.forms import BY, GRID, GROUPINGS, PERIODS
from beacon.analysis.reports.options import AnalysisOptions, RiskModelOptions
from beacon.analysis.reports.statistics import Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.backtest import Backtest


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    transaction_cost_bps=5.0, benchmark=benchmark).run(index(), start=START,
                                                                       end=END)


@pytest.fixture(scope="module")
def portfolio(result):
    return result.analytics.portfolio


class TestTheCatalogue:
    def test_each_grouping_has_its_calculation(self):
        assert set(GROUPINGS) == set(CALCULATIONS)

    def test_a_grouped_table_takes_by_and_no_calendar_period(self):
        for statistic_id in GROUPINGS:
            assert BY in CATALOGUE[statistic_id].forms, statistic_id
            assert statistic_id not in PERIODS, statistic_id

    def test_only_the_total_return_is_a_grid(self):
        assert [statistic.id for statistic in CATALOGUE if GRID in statistic.forms] == [
            "total_return"]


class TestTrading:
    def test_costs_by_reason(self,
                             result,
                             portfolio):
        found = portfolio.trading.costs(by="reason")

        assert found.equals(_calculator(result).costs_by_reason())

    def test_costs_each_year(self,
                             result,
                             portfolio):
        assert portfolio.trading.costs(by="year").equals(_calculator(result).costs("year"))

    def test_trades_by_reason(self,
                              result,
                              portfolio):
        found = portfolio.trading.trade_counts(by="reason")

        assert found.equals(_calculator(result).trading_by_reason())
        assert found["share"].sum() == pytest.approx(1.0)

    def test_a_group_it_is_not_split_by(self,
                                        portfolio):
        with pytest.raises(ValueError, match="costs by is one of reason, year"):
            portfolio.trading.costs(by="month")


class TestFlows:
    def test_dealing_costs_by_kind_and_year(self):
        from beacon.backtest import DatedFlows
        from test_backtest_etf import etf
        from test_backtest_flows import DAYS, QUIET, run

        listed = run(vehicle=etf(in_kind=True),
                     flows=DatedFlows({QUIET: 500_000.0, DAYS[60]: -300_000.0}))
        flows = listed.analytics.portfolio.flows
        calculator = _calculator(listed)

        assert flows.dealing_costs(by="kind").equals(calculator.dealing_summary("run"))
        assert flows.dealing_costs(by="year").equals(calculator.dealing_summary("year"))

    def test_none_without_flows(self,
                                portfolio):
        with pytest.raises(Unavailable):
            portfolio.flows.dealing_costs(by="year")


class TestAttribution:
    def test_brinson_each_period(self):
        from test_run_attribution import consistent

        run = consistent()
        analytics = Analytics(run)
        found = analytics.portfolio.attribution.brinson(by="year")
        tracked = analytics._context.require_tracked()

        pd.testing.assert_frame_equal(
            found, analytics._context.attribution.brinson_by_period("SECTOR", tracked, "Y"))
        assert found["residual"].abs().max() < 1e-9


class TestRisk:
    def test_herfindahl_each_period(self,
                                    portfolio):
        found = portfolio.risk.herfindahl(by="quarter")

        assert list(found.index) == ["2024Q1", "2024Q2", "2024Q3", "2024Q4"]
        assert list(found.columns) == ["names", "herfindahl", "normalised_herfindahl",
                                       "effective_number"]

    def test_risk_contributions_by_sector(self):
        from test_factor_analytics import SOURCE, factors
        from test_risk_contributions_over_time import noisy

        request = AnalysisRequest(options=AnalysisOptions(risk_model=RiskModelOptions(window=100)),
                                  factors=factors())
        risk = Analytics(noisy(), request, source=SOURCE).portfolio.risk
        by_name = risk.risk_contributions()
        found = risk.risk_contributions(by="sector")

        assert sorted(found.index) == ["Energy", "Health", "Tech"]
        assert found["contribution"].sum() == pytest.approx(by_name["contribution"].sum())

    def test_a_field_the_run_did_not_keep(self):
        from test_factor_analytics import SOURCE, factors
        from test_risk_contributions_over_time import noisy

        request = AnalysisRequest(options=AnalysisOptions(risk_model=RiskModelOptions(window=100)),
                                  factors=factors())
        risk = Analytics(noisy(), request, source=SOURCE).portfolio.risk

        with pytest.raises(ValueError, match="kept with a COUNTRY field"):
            risk.risk_contributions(by="country")


class TestTheGrid:
    def test_the_fund_s_months(self,
                               result,
                               portfolio):
        found = portfolio.performance.total_return(by="month", grid=True)

        assert found.equals(_performance(result).monthly_grid())
        assert found.loc["2024", "Year"] == pytest.approx(portfolio.performance.total_return())

    def test_a_comparator_s_and_the_fund_less_its_own(self,
                                                      result,
                                                      portfolio):
        performance = portfolio.performance

        assert performance.total_return(by="month", grid=True, of="benchmark").equals(
            _performance(result).monthly_grid("benchmark"))
        assert performance.total_return(by="month", grid=True, of="active").equals(
            _performance(result).monthly_grid("difference index"))

    def test_only_by_month_and_one_portfolio(self,
                                             portfolio):
        with pytest.raises(ValueError, match="grid goes with by='month'"):
            portfolio.performance.total_return(by="year", grid=True)

        with pytest.raises(ValueError, match="A grid is one portfolio's"):
            portfolio.performance.total_return(by="month", grid=True, of="all")


def _calculator(result):
    from beacon.analysis.performance.portfolio import PortfolioAnalytics

    return PortfolioAnalytics(result.analytics.run_data)


def _performance(result):
    from beacon.analysis.performance.analytics import PerformanceAnalytics

    return PerformanceAnalytics(result.analytics.run_data)
