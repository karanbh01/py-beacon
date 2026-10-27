# tests/test_turnover.py
"""BN-383: turnover and trading, hand-computed and against a run's trades.

A week: a rebalance on the second day buying 100 and selling 60, money
arriving the same day and bought with (50), and a redemption on the fourth
day selling 30, over a NAV of 1,000 rising to 1,090.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance.portfolio import PortfolioAnalytics

DAYS = pd.bdate_range("2024-03-04", periods=5)
NAV = pd.Series([1000.0, 1050.0, 1100.0, 1080.0, 1090.0], index=DAYS)
TRADES = pd.DataFrame({"IDENTIFIER": ["AAA", "BBB", "AAA", "BBB"],
                       "SIDE": ["BUY", "SELL", "BUY", "SELL"],
                       "VALUE": [100.0, 60.0, 50.0, 30.0],
                       "REASON": ["rebalance", "rebalance", "subscription", "redemption"]},
                      index=pd.DatetimeIndex([DAYS[1], DAYS[1], DAYS[1], DAYS[3]], name="DATE"))


def portfolio(trades: pd.DataFrame | None = TRADES) -> PortfolioAnalytics:
    tables = {"trades": trades} if trades is not None else {}
    run = RunData(currency="USD", returns=NAV.pct_change().fillna(0.0), nav=NAV,
                  units=NAV * 0 + 1, initial_capital=1000.0, periods_per_year=252,
                  risk_free_rate=0.0, tables=tables)

    return PortfolioAnalytics(run)


class TestEachDay:

    def test_a_day_s_turnover_is_over_the_close_before(self):
        day = portfolio().turnover("day").loc[DAYS[1]]

        assert (day["buys"], day["sells"], day["trades"]) == (150.0, 60.0, 3)
        assert day["mean_trade"] == pytest.approx(70.0)
        assert day["one_way"] == pytest.approx(60 / 1000)
        assert day["two_way"] == pytest.approx(210 / 1000)
        assert day["rebalance"]

    def test_a_day_that_only_sold_has_no_one_way_turnover(self):
        day = portfolio().turnover("day").loc[DAYS[3]]

        assert day["one_way"] == 0.0
        assert day["two_way"] == pytest.approx(30 / 1100)
        assert not day["rebalance"]


class TestEachRebalance:

    def test_only_the_rebalance_s_own_trades(self):
        rebalance = portfolio().turnover("rebalance")

        assert list(rebalance.index) == [DAYS[1]]
        assert rebalance.loc[DAYS[1], "trades"] == 2
        assert rebalance.loc[DAYS[1], "one_way"] == pytest.approx(60 / 1000)
        assert rebalance.loc[DAYS[1], "two_way"] == pytest.approx(160 / 1000)


class TestEachYear:

    def test_a_year_s_turnover_is_over_its_mean_nav(self):
        year = portfolio().turnover("year").loc["2024"]

        assert year["one_way"] == pytest.approx(90 / NAV.mean())
        assert year["two_way"] == pytest.approx(240 / NAV.mean())
        assert year["trades"] == 4


class TestByReason:

    def test_trading_by_why_it_was_made(self):
        table = portfolio().trading_by_reason()

        assert table.loc["rebalance", "trades"] == 2
        assert table.loc["subscription", "buys"] == 50.0
        assert table.loc["redemption", "sells"] == 30.0
        assert table["share"].sum() == pytest.approx(1.0)
        assert table.index[0] == "rebalance"


class TestTheChecks:

    def test_an_unknown_view_is_refused(self):
        with pytest.raises(ValueError, match="weekly"):
            portfolio().turnover("weekly")

    def test_a_run_without_trades_is_refused(self):
        with pytest.raises(KeyError):
            portfolio(None).turnover()

    def test_a_run_s_turnover_reconciles_with_its_trades(self):
        from beacon.backtest.flows import DatedFlows
        from test_backtest_flows import DAYS as FLOW_DAYS
        from test_backtest_flows import holdings_mode, run

        result = run(flows=DatedFlows({FLOW_DAYS[20]: 5e5, FLOW_DAYS[40]: -3e5}),
                     implementation=holdings_mode())
        data = RunData.from_result(result)
        days = PortfolioAnalytics(data).turnover("day")
        made = data.tables["trades"]

        assert (days["buys"] + days["sells"]).sum() == pytest.approx(made["VALUE"].sum())
        assert days["trades"].sum() == len(made)
        assert days["rebalance"].sum() == (made["REASON"] == "rebalance").groupby(
            made.index.normalize()).any().sum()
