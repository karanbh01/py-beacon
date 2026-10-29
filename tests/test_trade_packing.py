# tests/test_trade_packing.py
"""BN-412: a stored run keeps what it traded and works the rest of its trades
out again, exactly.

Small runs from the other tests: a rebalancing run at a flat rate with a
benchmark, one that pays market impact, one whose orders are worked over
several days, one with share classes, one that reinvests dividends, one
holding a pound-quoted name in dollars, and one whose holding is delisted.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.data import RUN_FILE, RunData
from beacon.analysis.panels import TRADES
from beacon.analysis.trade_packing import (
    COST,
    IMPACT,
    PRICE,
    TRADES_KEPT,
    TRADES_PATCH,
    VALUE,
    fixed_rate,
    pack_trades,
    same_trades,
)
from beacon.backtest import ExecutionLimit, Implementation, MarketImpact
from beacon.backtest.engine import BacktestEngine
from beacon.portfolio.reasons import CORPORATE_ACTION, DIVIDEND_REINVESTMENT
from beacon.testing import index_result_from_weights

COST_BPS = 5.0
# At this rate the two orders of operation the fixed rate is charged in
# differ on the reinvesting run's trade, so the test tells them apart.
REINVESTING_BPS = 7.0


def flat():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index, run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)

    return run(index(), transaction_cost_bps=COST_BPS, benchmark=benchmark)


def with_impact():
    from test_backtest_execution import run

    return run(1e9, transaction_cost_bps=COST_BPS,
               implementation=Implementation(impact=MarketImpact()))


def worked_over_days():
    from test_backtest_execution import FIRST, run

    return run(1e4, {FIRST: {"THIN": 0.5, "BIG": 0.5}}, transaction_cost_bps=COST_BPS,
               implementation=Implementation(execution=ExecutionLimit(participation=0.1)))


def share_classes():
    from beacon.backtest import ShareClass
    from test_backtest_share_classes import classes

    return classes(ShareClass("Retail"), ShareClass("Inst", management_fee_bps=10),
                   capital={"Retail": 5e6, "Inst": 3e6}, transaction_cost_bps=COST_BPS)


def reinvesting():
    from test_backtest_dividends import run

    return run(transaction_cost_bps=REINVESTING_BPS)


def in_pounds():
    from test_book_currency import CAPITAL, DAYS, fetcher, pound_index

    data = fetcher()

    return BacktestEngine(start_date=str(DAYS[0].date()), end_date=str(DAYS[-1].date()),
                          initial_capital=CAPITAL, data_provider=data,
                          index_result=pound_index(data), currency="USD",
                          transaction_cost_bps=COST_BPS).run()


def delisted():
    from test_backtest_pricing import END, START, build_fetcher

    schedule = {pd.Timestamp("2021-01-06"): {"AAA": 0.5, "BBB": 0.5}}

    return BacktestEngine(start_date=START, end_date=END, initial_capital=100_000.0,
                          data_provider=build_fetcher(delist={"BBB": "2021-01-20"}),
                          index_result=index_result_from_weights(schedule), calendar="XNYS",
                          transaction_cost_bps=COST_BPS).run()


RUNS = {"flat": flat, "impact": with_impact, "worked over days": worked_over_days,
        "share classes": share_classes, "reinvesting": reinvesting, "in pounds": in_pounds,
        "delisted": delisted}


@pytest.fixture(scope="module", params=list(RUNS))
def run(request):
    return RunData.from_result(RUNS[request.param]())


def worked_out(packed: dict[str, pd.DataFrame],
               column: str) -> bool:
    """Whether *column* was worked out again for every trade: neither kept
    whole nor put right by the patch."""
    return (column not in packed[TRADES_KEPT].columns
            and column not in set(packed[TRADES_PATCH]["COLUMN"]))


def stored(run: RunData,
           folder) -> RunData:
    run.write(folder)

    return RunData.from_record(folder)


class TestTheRoundTrip:

    def test_the_trades_come_back_exactly(self,
                                          run,
                                          tmp_path):
        live = run.tables[TRADES]

        back = stored(run, tmp_path / "run").tables[TRADES]

        assert len(live) > 0
        assert same_trades(back, live)

    def test_what_was_traded_is_kept_and_the_rest_is_not(self,
                                                         run,
                                                         tmp_path):
        stored(run, tmp_path / "run")
        folder = tmp_path / "run"

        assert not (folder / f"{TRADES}.parquet").exists()
        assert (folder / f"{TRADES_KEPT}.parquet").exists()
        assert (folder / f"{TRADES_PATCH}.parquet").exists()

    def test_the_packed_parts_are_not_tables_of_their_own(self,
                                                          run,
                                                          tmp_path):
        again = stored(run, tmp_path / "run")

        assert TRADES in again.tables
        assert TRADES_KEPT not in again.tables and TRADES_PATCH not in again.tables


class TestWhatIsWorkedOut:

    @pytest.fixture(scope="class")
    def flat_run(self):
        return RunData.from_result(flat())

    def test_the_fixed_rate_is_read_off_the_trades(self,
                                                   flat_run):
        assert fixed_rate(flat_run.tables[TRADES]) == COST_BPS

    def test_a_flat_rate_run_keeps_only_what_it_traded(self,
                                                      flat_run):
        packed, rate = pack_trades(flat_run.tables[TRADES], flat_run.tables, flat_run.currency)

        assert rate == COST_BPS
        assert all(worked_out(packed, column) for column in (PRICE, VALUE, COST, IMPACT))
        assert packed[TRADES_PATCH].empty

    def test_the_rate_is_kept_with_the_run(self,
                                           flat_run,
                                           tmp_path):
        import json

        flat_run.write(tmp_path / "run")

        kept = json.loads((tmp_path / "run" / RUN_FILE).read_text(encoding="utf-8"))["trades"]

        assert kept["rate"] == COST_BPS
        assert kept["columns"] == list(flat_run.tables[TRADES].columns)

    def test_market_impact_keeps_the_costs_whole(self):
        run = RunData.from_result(with_impact())
        trades = run.tables[TRADES]

        packed, _ = pack_trades(trades, run.tables, run.currency)

        assert (trades[IMPACT] > 0).any()
        assert {COST, IMPACT} <= set(packed[TRADES_KEPT].columns)
        assert not {PRICE, VALUE} & set(packed[TRADES_KEPT].columns)

    def test_a_reinvested_dividend_is_costed_in_its_own_order(self):
        run = RunData.from_result(reinvesting())
        trades = run.tables[TRADES]

        packed, _ = pack_trades(trades, run.tables, run.currency)

        reinvested = trades[trades["REASON"] == DIVIDEND_REINVESTMENT]
        value = reinvested[VALUE].to_numpy()

        assert (value * REINVESTING_BPS / 10_000.0 != value * (REINVESTING_BPS / 10_000.0)).any()
        assert worked_out(packed, COST)

    def test_a_pound_quoted_name_is_priced_in_dollars(self):
        run = RunData.from_result(in_pounds())
        trades = run.tables[TRADES]

        packed, _ = pack_trades(trades, run.tables, run.currency)

        assert (trades["PRICE"] != 2.0).all()
        assert worked_out(packed, PRICE)

    def test_a_delisting_s_settlement_is_booked_at_no_cost(self):
        run = RunData.from_result(delisted())
        trades = run.tables[TRADES]
        settled = trades[trades["REASON"] == CORPORATE_ACTION]

        packed, _ = pack_trades(trades, run.tables, run.currency)

        assert len(settled) and (settled[COST] == 0.0).all()
        assert worked_out(packed, COST)


class TestTheCheck:

    def test_one_bit_apart_is_not_the_same(self):
        trades = RunData.from_result(flat()).tables[TRADES]
        nudged = trades.copy()
        nudged.iloc[0, nudged.columns.get_loc(PRICE)] = np.nextafter(
            trades[PRICE].iloc[0], np.inf)

        assert same_trades(trades, trades.copy())
        assert not same_trades(nudged, trades)

    def test_trades_that_would_not_come_back_are_kept_as_they_are(self,
                                                                  tmp_path,
                                                                  monkeypatch):
        from beacon.analysis import trade_packing

        run = RunData.from_result(flat())
        monkeypatch.setattr(trade_packing, "same_trades", lambda ours, theirs: False)

        run.write(tmp_path / "run")
        back = RunData.from_record(tmp_path / "run").tables[TRADES]

        # This module's own `same_trades` is the real check.
        assert (tmp_path / "run" / f"{TRADES}.parquet").exists()
        assert not (tmp_path / "run" / f"{TRADES_KEPT}.parquet").exists()
        assert same_trades(back, run.tables[TRADES])
