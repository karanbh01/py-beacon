# tests/test_trade_reasons.py
"""BN-406: every trade says why it was made, and a run keeps them as a table.

The reasons and their tags; the portfolio's blocks of trading; and runs that
trade for each kind of reason: a rebalance, flows in and out, cash kept in
its band, and an active strategy's model-driven rebalances.
"""
import pandas as pd
import pytest

from beacon.backtest import Implementation
from beacon.backtest.breakdown import TAG_SEPARATOR, daily_costs, trades
from beacon.backtest.flows import DatedFlows
from beacon.portfolio.base import Portfolio, TradeInstruction
from beacon.portfolio.reasons import (
    CASH_RAISE,
    CASH_SPEND_DOWN,
    CATEGORIES,
    DISTRIBUTION_FUNDING,
    DIVERSIFICATION,
    MODEL_DRIVEN,
    REASONS,
    REBALANCE,
    REDEMPTION,
    SUBSCRIPTION,
    tags_for,
)

DAY = pd.Timestamp("2024-03-04")


class TestTheReasons:

    def test_every_reason_has_a_category(self):
        assert set(REASONS) == set(CATEGORIES)
        # The 22 agreed, and settlement: cash moved for a derivative's
        # settlement (futures margin, a swap reset, a hedge).
        assert len(REASONS) == 23

    def test_a_reason_that_moves_cash_tags_its_direction(self):
        assert tags_for(DISTRIBUTION_FUNDING, "SELL") == (DISTRIBUTION_FUNDING, CASH_RAISE)
        assert tags_for(SUBSCRIPTION, "BUY") == (SUBSCRIPTION, CASH_SPEND_DOWN)
        assert tags_for(REDEMPTION, "SELL") == (REDEMPTION, CASH_RAISE)

    def test_a_rebalance_carries_its_stages_once(self):
        assert tags_for(REBALANCE, "BUY", (DIVERSIFICATION, REBALANCE)) == (REBALANCE,
                                                                             DIVERSIFICATION)

    def test_a_cash_raise_is_not_tagged_twice(self):
        assert tags_for(CASH_RAISE, "SELL") == (CASH_RAISE,)


class TestBlocksOfTrading:

    @pytest.fixture
    def book(self):
        return Portfolio("p", initial_cash=1_000.0)

    def test_a_block_s_reason_and_tags(self,
                                       book):
        with book.trading(REBALANCE, {"AAA": (DIVERSIFICATION,)}, (MODEL_DRIVEN,)):
            book.apply(TradeInstruction("AAA", "BUY", 1, 10.0, 0.0), DAY)
            book.apply(TradeInstruction("BBB", "BUY", 1, 10.0, 0.0), DAY)

        first, second = book.transactions

        assert first.reason == REBALANCE
        assert first.tags == (REBALANCE, DIVERSIFICATION, MODEL_DRIVEN)
        assert second.tags == (REBALANCE, MODEL_DRIVEN)

    def test_the_innermost_block_wins_and_a_trade_s_own_reason_over_both(self,
                                                                         book):
        with book.trading(REBALANCE), book.trading(SUBSCRIPTION):
            book.apply(TradeInstruction("AAA", "BUY", 1, 10.0, 0.0), DAY)
            book.apply(TradeInstruction("BBB", "BUY", 1, 10.0, 0.0, reason=REDEMPTION), DAY)

        assert [trade.reason for trade in book.transactions] == [SUBSCRIPTION, REDEMPTION]

    def test_a_trade_outside_any_block_has_no_reason(self,
                                                     book):
        book.apply(TradeInstruction("AAA", "BUY", 1, 10.0, 0.0), DAY)

        assert book.transactions[0].reason == "" and book.transactions[0].tags == ()


class TestRuns:

    def test_a_rebalance_s_trades(self):
        from test_strategy_futures import index, run

        result = run(index())

        assert {trade.reason for trade in result.portfolio.transactions} == {REBALANCE}

    def test_flows_in_and_out(self):
        from test_backtest_flows import DAYS, holdings_mode, run

        result = run(flows=DatedFlows({DAYS[20]: 5e5, DAYS[40]: -3e5}),
                     implementation=holdings_mode())
        reasons = {trade.reason for trade in result.portfolio.transactions}
        redeemed = [trade for trade in result.portfolio.transactions if trade.reason == REDEMPTION]

        assert {SUBSCRIPTION, REDEMPTION} <= reasons
        assert all(trade.transaction_type == "SELL" for trade in redeemed)
        assert all(CASH_RAISE in trade.tags for trade in redeemed)

    def test_cash_raised_back_into_its_band(self):
        from test_strategy_futures import FLOW_DAY, index, run

        result = run(index(), flows=DatedFlows({FLOW_DAY: -1e6}),
                     implementation=Implementation(cash_buffer=0.05, cash_band=(0.04, 0.06)))

        assert CASH_RAISE in {trade.reason for trade in result.portfolio.transactions}

    def test_an_active_strategy_s_rebalances_are_model_driven(self):
        from beacon.strategy.active import MaxAlpha
        from test_strategy_active import run, strategy

        result = run(strategy(construction=MaxAlpha(tracking_error=0.02)))
        rebalances = [trade for trade in result.portfolio.transactions
                      if trade.reason == REBALANCE]

        assert rebalances and all(MODEL_DRIVEN in trade.tags for trade in rebalances)


class TestTheTable:

    @pytest.fixture(scope="class")
    def result(self):
        from test_backtest_flows import DAYS, holdings_mode, run

        return run(flows=DatedFlows({DAYS[20]: 5e5, DAYS[40]: -3e5}),
                   implementation=holdings_mode(), transaction_cost_bps=10.0)

    def test_it_reconciles_with_the_run_s_transactions(self,
                                                       result):
        table = trades(result)
        made = result.portfolio.transactions

        assert len(table) == len(made)
        assert table["QUANTITY"].tolist() == [trade.quantity for trade in made]
        assert (table["VALUE"] == table["QUANTITY"] * table["PRICE"]).all()
        assert table["REASON"].tolist() == [trade.reason for trade in made]
        assert table["TAGS"].iloc[0] == TAG_SEPARATOR.join(made[0].tags)

    def test_its_costs_are_the_run_s(self,
                                     result):
        table = trades(result)
        costs = daily_costs(result)

        assert table["COST"].sum() == pytest.approx((costs["fixed"] + costs["impact"]).sum())
        assert table["COST"].sum() > 0

    def test_it_is_kept_with_the_run(self,
                                     result,
                                     tmp_path):
        from beacon.analysis.data import RunData

        RunData.from_result(result).write(tmp_path / "run")
        kept = RunData.from_record(tmp_path / "run").tables["trades"]

        pd.testing.assert_frame_equal(kept, trades(result), check_freq=False,
                                      check_column_type=False, check_dtype=False)
