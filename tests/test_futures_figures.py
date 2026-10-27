# tests/test_futures_figures.py
"""BN-409: how a fund traded its index futures, and their basis risk.

Hand-computed on small tables, then on runs of test_strategy_futures' index
held through fractional futures with no dividend yield: one on its own index
at fair value with no cash rate, whose futures move exactly as the index, and
others that each add one source of basis.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.performance import futures_figures as figures
from beacon.analysis.performance.portfolio import PortfolioAnalytics
from beacon.backtest import DatedFlows
from test_strategy_futures import FLOW_DAY, NO_YIELD, futures, run

DAYS = pd.bdate_range("2024-03-04", periods=4)


def analytics(result) -> PortfolioAnalytics:
    return PortfolioAnalytics(RunData.from_result(result))


@pytest.fixture(scope="module")
def plain():
    return run(futures(**NO_YIELD))


class TestByHand:

    def test_trading_by_reason(self):
        trades = pd.DataFrame({"reason": ["open", "resize", "roll"],
                               "traded": [100.0, 10.0, 220.0],
                               "notional": [1e6, 1e5, 2.2e6], "cost": [200.0, 20.0, 330.0]},
                              index=DAYS[:3])
        nav = pd.Series(1e6, index=DAYS)

        table = figures.futures_trading(trades, nav, periods_per_year=4)

        assert table.loc["open", "cost_per_contract"] == pytest.approx(2.0)
        assert table.loc["roll", "cost_bps"] == pytest.approx(1.5)
        assert table.loc["between_rolls", "contracts"] == 110.0
        assert table.loc["all", "turnover"] == pytest.approx(3.3)

    def test_the_basis_splits_into_carry_and_mispricing(self):
        days = pd.DataFrame({"price": [1010.0], "fair_value": [1005.0], "level": [1000.0]},
                            index=DAYS[:1])

        basis = figures.daily_basis(days).iloc[0]

        assert basis["basis"] == pytest.approx(0.010)
        assert basis["carry"] == pytest.approx(0.005)
        assert basis["mispricing"] == pytest.approx(1010 / 1005 - 1)

    def test_a_roll_day_is_left_out_of_the_returns(self):
        expiry = pd.to_datetime(["2024-03-15"] * 3 + ["2024-06-21"])
        days = pd.DataFrame({"price": [100.0, 101.0, 102.0, 110.0],
                             "level": [100.0, 101.0, 101.0, 102.0],
                             "index_level": [100.0, 100.0, 101.0, 102.0], "expiry": expiry},
                            index=DAYS)

        returns = figures.basis_returns(days)

        assert list(returns.index) == list(DAYS[1:3])
        assert returns.loc[DAYS[1], figures.PRICING] == pytest.approx(0.0)
        assert returns.loc[DAYS[1], figures.MISMATCH] == pytest.approx(0.01)
        assert returns.loc[DAYS[2], figures.PRICING] == pytest.approx(102 / 101 - 1)


class TestOnRuns:

    def test_futures_on_their_own_index_at_fair_value_carry_no_basis(self,
                                                                     plain):
        risk = analytics(plain).futures_basis_risk()

        assert risk.loc[figures.PRICING, "tracking_error"] == pytest.approx(0.0, abs=1e-12)
        assert risk.loc[figures.MISMATCH, "tracking_error"] == pytest.approx(0.0, abs=1e-12)
        assert risk.loc[figures.TOTAL, "days"] > 200

    def test_a_mispricing_is_pricing_basis(self):
        risk = analytics(run(futures(mispricing_bps=10.0, **NO_YIELD))).futures_basis_risk()

        assert risk.loc[figures.PRICING, "tracking_error"] > 0.001
        assert risk.loc[figures.MISMATCH, "tracking_error"] == pytest.approx(0.0, abs=1e-12)

    def test_another_index_is_underlying_mismatch(self,
                                                  plain):
        other = plain.index.target.levels ** 1.02
        risk = analytics(run(futures(underlying=other, **NO_YIELD))).futures_basis_risk()

        assert risk.loc[figures.MISMATCH, "tracking_error"] > 0.0005
        assert risk.loc[figures.PRICING, "tracking_error"] == pytest.approx(0.0, abs=1e-12)

    def test_the_cash_rate_is_given_back_through_the_pricing_basis(self):
        risk = analytics(run(futures(**NO_YIELD), cash_rate=0.03)).futures_basis_risk()

        # Futures converge to their index, giving up the carry: about the
        # cash rate a year, on calendar days over sessions.
        assert risk.loc[figures.PRICING, "mean_a_year"] == pytest.approx(-0.03, rel=0.1)

    def test_the_trading_reconciles_with_the_run_s_costs(self):
        result = run(futures(cost_per_contract=2.0, trading_cost_bps=1.0, roll_cost_bps=0.5,
                             **NO_YIELD), flows=DatedFlows({FLOW_DAY: 2e6}))
        portfolio = analytics(result)
        table = portfolio.futures_trading()
        costs = portfolio.run.tables["costs"]

        assert table.loc["all", "cost"] == pytest.approx(
            costs["futures_rolls"].sum() + costs["futures_trading"].sum())
        assert table.loc["between_rolls", "cost"] == pytest.approx(costs["futures_trading"].sum())
        assert table.loc["all", "trades"] == len(result.futures_trades)

    def test_the_basis_summary_is_in_basis_points(self):
        summary = analytics(run(futures(mispricing_bps=10.0, **NO_YIELD))).futures_basis()

        assert 5.0 < summary.loc["mispricing", "volatility_bps"] < 16.0

    def test_a_fund_without_futures_has_none(self):
        from test_backtest_flows import run as flow_run

        with pytest.raises(KeyError):
            analytics(flow_run()).futures_trading()
