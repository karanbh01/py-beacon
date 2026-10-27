# tests/test_fund_figures.py
"""BN-385: a fund's characteristics, income, AUM and flows, and an ETF's
premium, hand-computed on small books.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.performance import fund_figures as figures

NAMES = pd.DataFrame({"IDENTIFIER": ["A", "B", "C"], "MARKET_CAP": [100.0, 200.0, np.nan],
                      "PE_RATIO": [10.0, 20.0, -5.0], "PB_RATIO": [2.0, 0.0, 4.0],
                      "DIVIDEND_YIELD": [0.02, 0.03, 0.01]})
WEIGHTS = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


class TestCharacteristics:

    def test_means_over_the_names_with_a_value(self):
        found = figures.weighted_characteristics(NAMES, WEIGHTS)

        assert found["MARKET_CAP"] == pytest.approx((0.5 * 100 + 0.3 * 200) / 0.8)
        assert found["DIVIDEND_YIELD"] == pytest.approx(0.021)

    def test_harmonic_means_take_a_loss_in_their_stride(self):
        """C's loss lowers the earnings yield; it does not blow the P/E up."""
        found = figures.weighted_characteristics(NAMES, WEIGHTS)

        assert found["PE_RATIO"] == pytest.approx(1.0 / (0.5 / 10 + 0.3 / 20 + 0.2 / -5))
        assert found["PB_RATIO"] == pytest.approx(0.7 / (0.5 / 2 + 0.2 / 4))

    def test_a_book_without_the_names_has_none(self):
        found = figures.weighted_characteristics(NAMES, pd.Series({"Z": 1.0}))

        assert found["MARKET_CAP"] is None and found["PE_RATIO"] is None

    def test_each_book_at_each_date(self):
        dates = pd.bdate_range("2024-01-02", periods=3)
        taken = NAMES.set_index(pd.DatetimeIndex([dates[1]] * 3))
        fund = pd.DataFrame([WEIGHTS] * 3, index=dates)
        index = pd.DataFrame([{"A": 1.0, "B": 0.0, "C": 0.0}] * 3, index=dates)

        table = figures.characteristics_over_time(taken, {"fund": fund, "index": index})

        assert list(table.index) == [dates[1]]
        assert table.loc[dates[1], ("index", "MARKET_CAP")] == 100.0
        assert table.loc[dates[1], ("fund", "DIVIDEND_YIELD")] == pytest.approx(0.021)


class TestIncome:

    DAYS = pd.bdate_range("2023-01-02", periods=400)
    NAV = pd.Series(1000.0, index=DAYS)

    def cash(self):
        cash = pd.DataFrame(0.0, index=self.DAYS,
                            columns=["distributions", "dividends_gross", "flows_dealt"])
        cash.loc[self.DAYS[300], "distributions"] = -10.0
        cash.loc[self.DAYS[290], "dividends_gross"] = 20.0
        cash.loc[self.DAYS[100], "dividends_gross"] = 15.0

        return cash

    def test_the_trailing_yields_from_a_year_in(self):
        yields = figures.trailing_yields(self.cash(), self.NAV)
        last = yields.iloc[-1]

        assert yields.iloc[:200].isna().all().all()
        assert last["distribution_yield"] == pytest.approx(10 / 1000)
        assert last["dividend_yield"] == pytest.approx(20 / 1000)

    def test_dividend_growth_year_on_year(self):
        growth = figures.dividend_growth(self.cash())

        assert growth.loc["2023", "dividends"] == 15.0
        assert growth.loc["2024", "dividends"] == 20.0
        assert growth.loc["2024", "growth"] == pytest.approx(20 / 15 - 1)


class TestFlows:

    DAYS = pd.bdate_range("2024-03-01", periods=30)
    NAV = pd.Series(1000.0, index=DAYS)

    def test_net_flows_as_a_share_of_the_aum_they_began_with(self):
        cash = pd.DataFrame({"flows_dealt": 0.0}, index=self.DAYS)
        cash.loc[self.DAYS[5], "flows_dealt"] = 50.0
        cash.loc[self.DAYS[25], "flows_dealt"] = -20.0

        table = figures.net_flows(cash, self.NAV)

        assert table.loc["2024-03", "net_flows"] == 50.0
        assert table.loc["2024-03", "share_of_aum"] == pytest.approx(50 / 1000)
        assert table.loc["2024-04", "net_flows"] == -20.0

    def test_the_cost_of_trading_the_flows(self):
        quiet, rebalance = self.DAYS[3], self.DAYS[10]
        cash = pd.DataFrame({"flows_dealt": 0.0}, index=self.DAYS)
        cash.loc[quiet, "flows_dealt"] = 100.0
        cash.loc[rebalance, "flows_dealt"] = -50.0
        costs = pd.DataFrame({"creation_fees": 0.0}, index=self.DAYS)
        costs.loc[quiet, "creation_fees"] = 0.4
        trades = pd.DataFrame({"VALUE": [100.0, 200.0, 50.0], "COST": [1.0, 2.0, 0.5],
                               "REASON": ["subscription", "rebalance", "redemption"]},
                              index=pd.DatetimeIndex([quiet, rebalance, rebalance]))

        table = figures.flow_costs(trades, cash, costs)

        assert table.loc[quiet, "trading_cost"] == 1.0
        assert table.loc[quiet, "net_cost"] == pytest.approx(0.6)
        assert table.loc[rebalance, "trading_cost"] == pytest.approx(2.5 * 50 / 250)


class TestTheEtf:

    def test_the_premium_s_figures(self):
        market = pd.DataFrame({"premium": [0.001, -0.002, 0.003], "spread": [0.001] * 3})

        found = figures.premium_figures(market)

        assert found["mean_premium"] == pytest.approx(0.002 / 3)
        assert found["premium_volatility"] == pytest.approx(np.std([0.001, -0.002, 0.003],
                                                                   ddof=1))
        assert (found["widest_premium"], found["widest_discount"]) == (0.003, -0.002)
        assert found["mean_spread"] == pytest.approx(0.001)

    def test_a_run_s_quotes_are_kept_and_measured(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.portfolio import PortfolioAnalytics
        from test_backtest_etf import etf
        from test_backtest_flows import run

        result = run(vehicle=etf(in_kind=True))
        portfolio = PortfolioAnalytics(RunData.from_result(result))

        assert portfolio.etf_premium()["mean_premium"] == pytest.approx(
            result.market["premium"].mean())

    def test_another_vehicle_has_no_premium(self):
        from beacon.analysis.data import RunData
        from beacon.analysis.performance.portfolio import PortfolioAnalytics
        from test_strategy_futures import index, run

        assert PortfolioAnalytics(RunData.from_result(run(index()))).etf_premium() is None
