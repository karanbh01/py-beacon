# tests/test_futures_trading.py
"""BN-409: every trade of a fund's index futures, and what it cost.

Runs reuse test_strategy_futures' eight-name market-cap index through 2024,
held through rolled futures by a fund of 10,000,000, in whole contracts of 50
times the index (200 contracts to start). 2,000,000 arriving mid-year makes
the position resize; without a flow or a cash rate it never drifts from the
fund's net assets.
"""
from itertools import pairwise

import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.panels import FUTURES_DAYS, FUTURES_TRADES
from beacon.backtest import Backtest, DatedFlows
from beacon.backtest.breakdown import daily_costs
from beacon.backtest.futures import OPEN, RESIZE, ROLL
from beacon.strategy.futures import IndexFutures
from test_strategy_futures import FLOW_DAY, NO_YIELD, futures, index, run

COSTLY = {"cost_per_contract": 2.0, "trading_cost_bps": 1.0, "roll_cost_bps": 0.5}
FLOWS = DatedFlows({FLOW_DAY: 2e6})


@pytest.fixture(scope="module")
def costly():
    return run(futures(dividend_yield=0.0, **COSTLY), flows=FLOWS)


class TestTheTrades:

    def test_the_position_opens_rolls_and_is_resized(self,
                                                     costly):
        trades = costly.futures_trades
        reasons = [trade.reason for trade in trades]

        assert reasons[0] == OPEN
        assert reasons.count(ROLL) == len(costly.futures_rolls)
        assert RESIZE in reasons

    def test_each_trade_carries_on_from_the_last(self,
                                                 costly):
        trades = costly.futures_trades

        for before, after in pairwise(trades):
            assert after.before == before.after
        for trade in trades:
            legs = trade.before + trade.after if trade.reason == ROLL else abs(
                trade.after - trade.before)
            assert trade.traded == legs
            assert trade.notional == pytest.approx(trade.traded * trade.price * 50.0)

        assert costly.futures_days["contracts"].iloc[-1] == trades[-1].after

    def test_each_trade_is_costed_per_contract_and_in_basis_points(self,
                                                                   costly):
        for trade in costly.futures_trades:
            if trade.reason == ROLL:
                continue
            assert trade.cost == pytest.approx(trade.traded * 2.0 + trade.notional * 1e-4)

    def test_a_roll_pays_its_roll_cost_and_both_legs(self,
                                                     costly):
        rolls = [trade for trade in costly.futures_trades if trade.reason == ROLL]

        # The roll cost is on the net assets the position is sized to, which
        # the close's NAV is after paying it.
        for trade, roll in zip(rolls, costly.futures_rolls, strict=True):
            assert trade.cost == pytest.approx(roll.cost)
            assert roll.cost - trade.traded * 2.0 == pytest.approx(
                costly.portfolio.nav.loc[roll.date] * 0.5e-4, rel=1e-3)

    def test_the_costs_are_the_run_s(self,
                                     costly):
        costs = daily_costs(costly)
        between = sum(trade.cost for trade in costly.futures_trades if trade.reason != ROLL)

        assert costs["futures_trading"].sum() == pytest.approx(between)
        assert costs["futures_rolls"].sum() == pytest.approx(
            sum(roll.cost for roll in costly.futures_rolls))

    def test_free_futures_cost_nothing(self):
        result = run(futures(**NO_YIELD))

        assert result.futures_trades
        assert {trade.cost for trade in result.futures_trades} == {0.0}

    def test_the_costs_are_what_the_fund_gives_up(self,
                                                  costly):
        free = run(futures(dividend_yield=0.0), flows=FLOWS)
        paid = sum(trade.cost for trade in costly.futures_trades)

        assert free.portfolio.nav.iloc[-1] - costly.portfolio.nav.iloc[-1] == pytest.approx(
            paid, rel=0.02)


class TestTheDays:

    def test_each_day_held_is_priced_at_fair_value(self,
                                                   costly):
        days = costly.futures_days

        assert days.index[0] == costly.futures_trades[0].date
        assert days["price"].to_numpy() == pytest.approx(days["fair_value"].to_numpy())
        assert (days["expiry"] > days.index).all()

    def test_a_stored_run_keeps_them(self,
                                     costly,
                                     tmp_path):
        data = RunData.from_result(costly)
        data.write(tmp_path / "run")
        again = RunData.from_record(tmp_path / "run")

        for name in (FUTURES_TRADES, FUTURES_DAYS):
            pd.testing.assert_frame_equal(again.tables[name], data.tables[name],
                                          check_freq=False, check_column_type=False)


class TestTheTerms:

    @pytest.mark.parametrize("name", ["cost_per_contract", "trading_cost_bps"])
    def test_a_negative_cost_is_refused(self,
                                        name):
        with pytest.raises(ValueError, match=name):
            IndexFutures(**{name: -1.0})

    def test_a_total_return_index_keeps_every_other_term(self):
        terms = IndexFutures(multiplier=10.0, band=0.01, whole_contracts=False,
                             cost_per_contract=1.5)

        priced = Backtest._priced(terms, index("TOTAL_RETURN"))

        assert priced == IndexFutures(multiplier=10.0, band=0.01, whole_contracts=False,
                                      cost_per_contract=1.5, dividend_yield=0.0)
