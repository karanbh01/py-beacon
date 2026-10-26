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
    COST_KINDS,
    CREATION_FEES,
    FIXED,
    FUTURES_ROLLS,
    IMPACT,
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
