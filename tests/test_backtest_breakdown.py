# tests/test_backtest_breakdown.py
"""BN-363: a run's costs, day by day and by kind.

Small runs from the execution, futures and dealing tests: one that pays
market impact as well as a fixed rate, one that rolls futures, and one whose
flow pays the fund through a swung price.
"""
import numpy as np
import pytest

from beacon.backtest import MarketImpact, SwingPricing
from beacon.backtest.breakdown import (
    CASH_KINDS,
    COST_KINDS,
    CREATION_FEES,
    FIXED,
    FUTURES_ROLLS,
    IMPACT,
    daily_cash,
    daily_costs,
)
from beacon.backtest.implementation import Implementation
from test_backtest_dealing import protected
from test_backtest_execution import run as run_execution
from test_strategy_futures import futures
from test_strategy_futures import run as run_strategy

COST_BPS = 10.0


@pytest.fixture(scope="module")
def with_impact():
    return run_execution(1e9, transaction_cost_bps=COST_BPS,
                         implementation=Implementation(impact=MarketImpact()))


class TestTrades:

    def test_the_fixed_rate_is_charged_on_what_traded_at_a_cost(self,
                                                                with_impact):
        trades = with_impact.portfolio.transactions
        expected = sum(trade.quantity * trade.price * COST_BPS / 10_000.0
                       for trade in trades if trade.transaction_cost > 0)

        assert daily_costs(with_impact)[FIXED].sum() == pytest.approx(expected, rel=1e-12)

    def test_impact_is_the_rest_of_each_trade_s_cost(self,
                                                     with_impact):
        costs = daily_costs(with_impact)
        total = sum(trade.transaction_cost for trade in with_impact.portfolio.transactions)

        assert costs[IMPACT].sum() > 0
        assert costs[FIXED].sum() + costs[IMPACT].sum() == pytest.approx(total, rel=1e-12)

    def test_without_impact_every_cost_is_fixed(self):
        result = run_execution(1e6, transaction_cost_bps=COST_BPS)
        costs = daily_costs(result)

        assert costs[IMPACT].sum() == 0.0
        assert costs[FIXED].sum() == pytest.approx(
            sum(trade.transaction_cost for trade in result.portfolio.transactions), rel=1e-12)

    def test_costs_fall_on_the_days_trades_were_made(self,
                                                     with_impact):
        costs = daily_costs(with_impact)
        traded = {trade.transaction_date.normalize()
                  for trade in with_impact.portfolio.transactions if trade.transaction_cost > 0}

        assert set(costs.index[costs[FIXED] > 0]) == traded


class TestFuturesAndFlows:

    def test_each_roll_s_cost_on_its_day(self):
        result = run_strategy(futures(roll_cost_bps=2.0))
        costs = daily_costs(result)

        assert len(result.futures_rolls) > 0
        for roll in result.futures_rolls:
            assert costs.loc[roll.date, FUTURES_ROLLS] == pytest.approx(roll.cost)
        assert costs[FUTURES_ROLLS].sum() == pytest.approx(
            sum(roll.cost for roll in result.futures_rolls), rel=1e-12)

    def test_a_flow_s_dealing_adjustment_is_a_creation_fee(self):
        result = protected(SwingPricing())
        costs = daily_costs(result)
        paid = sum(flow.adjustment for flow in result.flows)

        assert paid > 0
        assert costs[CREATION_FEES].sum() == pytest.approx(paid, rel=1e-12)


class TestShape:

    def test_a_row_for_every_day_and_a_column_for_every_kind(self,
                                                             with_impact):
        costs = daily_costs(with_impact)

        assert costs.index.equals(with_impact.portfolio.nav.index)
        assert list(costs.columns) == list(COST_KINDS)
        assert np.isfinite(costs.to_numpy()).all()


class TestCashAndIncome:
    """BN-364: each column reconciles with the run's own totals and series."""

    @pytest.fixture(scope="class")
    def paying(self):
        from beacon import ModellingAssumptions
        from beacon.backtest import Vehicle
        from test_backtest_dividends import run

        return run(modelling_assumptions=ModellingAssumptions(withholding_tax_rate=0.15,
                                                              cash_rate=0.03),
                   vehicle=Vehicle(management_fee_bps=50))

    def test_dividends_less_withholding_are_what_was_received(self,
                                                             paying):
        cash = daily_cash(paying)

        assert cash["dividends_gross"].sum() > 0
        assert cash["withholding_tax"].sum() == pytest.approx(
            0.15 * cash["dividends_gross"].sum(), rel=1e-12)
        assert (cash["dividends_gross"] - cash["withholding_tax"]).sum() == pytest.approx(
            cash["dividends_received"].sum(), rel=1e-12)

    def test_dividends_are_owed_on_the_ex_date(self,
                                               paying):
        from test_backtest_dividends import EX

        cash = daily_cash(paying)

        assert list(cash.index[cash["dividends_gross"] > 0]) == [EX]

    def test_the_fee_charged_is_what_was_paid_and_what_is_still_owed(self,
                                                                    paying):
        cash = daily_cash(paying)

        assert cash["fee_charged"].sum() == pytest.approx(
            -cash["fees_paid"].sum() + cash["fees_payable"].iloc[-1], rel=1e-12)
        assert (cash["fee_charged"] > 0).sum() > 20

    def test_cash_that_moved_sums_to_its_flows_by_kind(self,
                                                       paying):
        cash = daily_cash(paying)
        interest = sum(flow.amount for flow in paying.portfolio.cash_flows
                       if flow.kind == "INTEREST")

        assert interest > 0
        assert cash["interest"].sum() == pytest.approx(interest, rel=1e-12)

    def test_balances_are_the_run_s_own(self,
                                        paying):
        cash = daily_cash(paying)
        recorded = paying.portfolio.cash

        assert cash["cash"].reindex(recorded.index).to_numpy() == pytest.approx(
            recorded.to_numpy())
        assert cash["fees_payable"].to_numpy() == pytest.approx(
            paying.fees_payable.reindex(cash.index).to_numpy())

    def test_a_receivable_waits_for_its_pay_date(self):
        from test_backtest_dividends import DAYS, EX, fetcher, run

        pay = DAYS[DAYS.get_loc(EX) + 5]
        cash = daily_cash(run(fetcher(pay_date=pay)))

        assert cash.loc[EX, "dividends_receivable"] == pytest.approx(
            cash.loc[EX, "dividends_gross"], rel=1e-12)
        assert cash.loc[pay, "dividends_receivable"] == 0.0
        assert cash.loc[pay, "dividends_received"] == pytest.approx(
            cash.loc[EX, "dividends_gross"], rel=1e-12)

    def test_futures_exposure_and_flows_dealt(self):
        result = run_strategy(futures(roll_cost_bps=2.0))
        cash = daily_cash(result)

        assert cash["futures_exposure"].to_numpy() == pytest.approx(
            result.futures_notional.reindex(cash.index).fillna(0.0).to_numpy())

        dealt = protected(SwingPricing())
        flows = daily_cash(dealt)

        assert flows["flows_dealt"].sum() == pytest.approx(
            sum(flow.amount for flow in dealt.flows), rel=1e-12)
        assert flows["subscriptions"].sum() > 0

    def test_a_run_without_dividends_has_none(self,
                                              with_impact):
        cash = daily_cash(with_impact)

        assert list(cash.columns) == list(CASH_KINDS)
        assert cash["dividends_gross"].sum() == 0.0
        assert with_impact.dividends_entitled.empty
