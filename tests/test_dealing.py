# tests/test_dealing.py
"""BN-408: what dealing cost, by flow and by share class.

The class runs reuse test_backtest_hedged_classes' three dollar names through
2024: a dollar class charged 20 bp and a euro class hedged into euros,
charged 30 bp, its forwards costing 2 bp of what they deal. The dollar class
takes 200,000 in March and the euro class pays out 100,000 euros in June.
Trades cost 10 bp.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import dealing
from beacon.analysis.performance.portfolio import PortfolioAnalytics
from beacon.backtest import DatedFlows, ShareClass
from test_backtest_hedged_classes import DAYS, run

IN, OUT = DAYS[50], DAYS[120]


@pytest.fixture(scope="module")
def classes():
    result = run(ShareClass("USD", management_fee_bps=20),
                 ShareClass("EUR H", currency="EUR", hedged=True, hedge_cost_bps=2.0,
                            management_fee_bps=30),
                 flows={"USD": DatedFlows({IN: 2e5}), "EUR H": DatedFlows({OUT: -1e5})},
                 transaction_cost_bps=10.0)

    return result, RunData.from_result(result)


class TestEachFlow:

    def test_each_flow_s_cost_to_the_fund(self,
                                          classes):
        _, data = classes
        table = PortfolioAnalytics(data).dealing_costs()

        assert table["kind"].tolist() == ["subscription", "redemption"]
        assert table.loc[IN, "flow"] == pytest.approx(2e5)
        assert table["net_bps"].to_numpy() == pytest.approx(10.0, rel=0.05)
        assert table["ap_cost"].isna().all()

    def test_added_up_by_kind_and_year(self):
        table = pd.DataFrame({"kind": ["subscription", "subscription", "redemption"],
                              "flow": [100.0, 300.0, -200.0], "trading_cost": [0.1, 0.3, 0.2],
                              "fees_received": [0.05, 0.0, 0.0], "net_cost": [0.05, 0.3, 0.2],
                              "ap_cost": [1.0, 2.0, 1.5]},
                             index=pd.DatetimeIndex(["2023-05-01", "2024-02-01", "2024-03-01"]))

        run_total = dealing.dealing_summary(table)
        years = dealing.dealing_summary(table, "year")

        assert run_total.loc["subscription", "flows"] == 2
        assert run_total.loc["subscription", "net_bps"] == pytest.approx(0.35 / 400 * 1e4)
        assert run_total.loc["subscription", "ap_cost"] == pytest.approx(3.0)
        assert years.loc[("2024", "redemption"), "size"] == 200.0

    def test_an_unknown_view_is_refused(self):
        with pytest.raises(ValueError, match="month"):
            dealing.dealing_summary(pd.DataFrame(columns=["kind", "flow"]), "month")


class TestAnEtf:

    def test_an_ap_s_cost_is_the_quote_s(self):
        from test_backtest_etf import UNIT_VALUE, etf
        from test_backtest_flows import QUIET
        from test_backtest_flows import run as flow_run

        result = flow_run(flows=DatedFlows({QUIET: 3 * UNIT_VALUE}),
                          vehicle=etf(in_kind=False, creation_fee_bps=5.0),
                          transaction_cost_bps=10.0)
        table = PortfolioAnalytics(RunData.from_result(result)).dealing_costs()
        quote = result.market.loc[QUIET]
        flow = result.flows[0]

        assert table.loc[QUIET, "ap_cost"] == pytest.approx(flow.amount * quote["create_cost"])
        assert table.loc[QUIET, "fees_received"] == pytest.approx(flow.adjustment)
        assert table.loc[QUIET, "net_cost"] == pytest.approx(
            table.loc[QUIET, "trading_cost"] - flow.adjustment)


class TestEachClass:

    def test_the_classes_trading_adds_up_to_the_fund_s(self,
                                                       classes):
        _, data = classes
        table = PortfolioAnalytics(data).class_costs()

        assert (table["flow_trading"] + table["other_trading"]).sum() == pytest.approx(
            data.tables["trades"]["COST"].sum())
        assert table["fees"].sum() == pytest.approx(data.tables["cash"]["fee_charged"].sum())

    def test_a_class_bears_the_trading_its_own_flows_caused(self,
                                                            classes):
        _, data = classes
        portfolio = PortfolioAnalytics(data)
        table, flows = portfolio.class_costs(), portfolio.dealing_costs()

        assert table.loc["USD", "flow_trading"] == pytest.approx(flows.loc[IN, "trading_cost"])
        assert table.loc["EUR H", "flow_trading"] == pytest.approx(
            flows.loc[OUT, "trading_cost"])

    def test_only_the_hedged_class_pays_for_hedging(self,
                                                    classes):
        result, data = classes
        table = PortfolioAnalytics(data).class_costs()
        hedges = result.share_classes["EUR H"].hedges

        assert table.loc["EUR H", "hedge_costs"] == pytest.approx(
            sum(trade.cost for trade in hedges))
        assert table.loc["USD", ["hedge_costs", "hedge_carry"]].tolist() == [0.0, 0.0]
        # The euro earns less than the dollar, so the carry costs the class.
        assert table.loc["EUR H", "hedge_carry"] < 0.0
        assert table.loc["EUR H", "net_bps_a_year"] > table.loc["USD", "net_bps_a_year"]

    def test_each_year(self,
                       classes):
        _, data = classes
        years = PortfolioAnalytics(data).class_costs("year")

        assert list(years.index) == [("EUR H", "2024"), ("USD", "2024")]
        assert years.loc[("USD", "2024"), "fees"] == pytest.approx(
            PortfolioAnalytics(data).class_costs().loc["USD", "fees"])

    def test_carry_is_earned_for_the_share_of_its_term_held(self):
        hedges = pd.DataFrame({"CLASS": "H", "CARRY": [-30.0, -60.0],
                               "EXPIRY": pd.to_datetime(["2024-01-31", "2024-02-29"])},
                              index=pd.DatetimeIndex(["2024-01-01", "2024-01-11"]))

        earned = dealing._carry_earned(hedges, pd.Timestamp("2024-03-31"))

        assert earned["H"].tolist() == pytest.approx([-10.0, -60.0])

    def test_a_stored_run_keeps_its_classes(self,
                                            classes,
                                            tmp_path):
        _, data = classes
        data.write(tmp_path / "run")
        again = PortfolioAnalytics(RunData.from_record(tmp_path / "run"))

        pd.testing.assert_frame_equal(again.class_costs(), PortfolioAnalytics(data).class_costs())

    def test_a_run_without_classes_has_none(self):
        from test_backtest_flows import run as flow_run

        with pytest.raises(KeyError):
            PortfolioAnalytics(RunData.from_result(flow_run())).class_costs()
