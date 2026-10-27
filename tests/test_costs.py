# tests/test_costs.py
"""BN-384: a run's costs, hand-computed and against its trades.

Four days over a year of four periods, at a NAV of 1,000: fixed costs of 2
and 1, impact of 1, a futures roll costing 3 and creation fees of 2 received;
and two trades, a rebalance of 1,000 costing 3 (1 of it impact) and a
subscription of 500 costing 1.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.portfolio import PortfolioAnalytics

DAYS = pd.bdate_range("2024-03-04", periods=4)
NAV = pd.Series(1000.0, index=DAYS)
COSTS = pd.DataFrame({"fixed": [0.0, 2.0, 0.0, 1.0], "impact": [0.0, 1.0, 0.0, 0.0],
                      "futures_rolls": [0.0, 0.0, 3.0, 0.0],
                      "creation_fees": [0.0, 0.0, 0.0, 2.0]}, index=DAYS)
TRADES = pd.DataFrame({"VALUE": [1000.0, 500.0], "COST": [3.0, 1.0], "IMPACT": [1.0, 0.0],
                       "REASON": ["rebalance", "subscription"], "SIDE": ["BUY", "BUY"]},
                      index=pd.DatetimeIndex([DAYS[1], DAYS[3]], name="DATE"))


@pytest.fixture
def portfolio():
    run = RunData(currency="USD", returns=NAV * 0.0, nav=NAV, units=NAV * 0 + 1,
                  initial_capital=1000.0, periods_per_year=4, risk_free_rate=0.0,
                  tables={"costs": COSTS, "trades": TRADES})

    return PortfolioAnalytics(run)


class TestOverTheRun:

    def test_each_kind_and_the_net(self,
                                   portfolio):
        table = portfolio.costs()

        assert table.loc["fixed", "total"] == 3.0
        assert table.loc["impact", "total"] == 1.0
        assert table.loc["futures_rolls", "total"] == 3.0
        assert table.loc["creation_fees_received", "total"] == 2.0
        assert table.loc["net", "total"] == pytest.approx(5.0)

    def test_in_basis_points_of_assets_a_year(self,
                                              portfolio):
        """Four days at four a year is one year, over a NAV of 1,000."""
        assert portfolio.costs().loc["net", "bps_a_year"] == pytest.approx(50.0)


class TestEachYear:

    def test_a_year_s_costs_and_its_net_in_basis_points(self,
                                                        portfolio):
        year = portfolio.costs("year").loc["2024"]

        assert year["fixed"] == 3.0 and year["net"] == pytest.approx(5.0)
        assert year["net_bps"] == pytest.approx(50.0)


class TestPerUnitTraded:

    def test_the_costs_in_basis_points_of_what_was_traded(self,
                                                          portfolio):
        found = portfolio.cost_per_traded()

        assert found["fixed_bps"] == pytest.approx(3 / 1500 * 1e4)
        assert found["impact_bps"] == pytest.approx(1 / 1500 * 1e4)
        assert found["total_bps"] == pytest.approx(4 / 1500 * 1e4)

    def test_by_why_the_trade_was_made(self,
                                       portfolio):
        table = portfolio.costs_by_reason()

        assert table.index[0] == "rebalance"
        assert table.loc["rebalance", "bps"] == pytest.approx(30.0)
        assert table.loc["subscription", "bps"] == pytest.approx(20.0)
        assert table["cost"].sum() == 4.0

    def test_an_unknown_view_is_refused(self,
                                        portfolio):
        with pytest.raises(ValueError, match="month"):
            portfolio.costs("month")


class TestAgainstARun:

    def test_the_trading_costs_reconcile_with_the_run_s(self):
        from beacon.backtest.flows import DatedFlows
        from test_backtest_flows import DAYS as FLOW_DAYS
        from test_backtest_flows import holdings_mode, run

        result = run(flows=DatedFlows({FLOW_DAYS[20]: 5e5}), implementation=holdings_mode(),
                     transaction_cost_bps=10.0)
        portfolio = PortfolioAnalytics(RunData.from_result(result))
        table = portfolio.costs()

        assert table.loc["fixed", "total"] + table.loc["impact", "total"] == pytest.approx(
            portfolio.costs_by_reason()["cost"].sum())
        assert portfolio.cost_per_traded()["total_bps"] == pytest.approx(10.0, rel=1e-6)
