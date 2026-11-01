# tests/test_holding_replay.py
"""BN-413: the fund's holdings worked out again from its trades.

Runs through a 2-for-1 and a 1-for-2 split (test_splits' flat market), a
delisting, dividends reinvested and share classes (test_trade_packing's
runs): the holdings worked out from the trades and the splits equal the
recorded ones exactly, and a stored run keeps only what they do not give
back.
"""
import pandas as pd
import pytest

from beacon.analysis.data import RunData
from beacon.analysis.holding_replay import DROPS, PATCH, pack_holdings, replayed, unpack_holdings
from beacon.analysis.panels import ACTIONS, FUND_WEIGHTS, HOLDINGS, TRADES
from beacon.backtest.engine import BacktestEngine
from test_trade_packing import delisted, reinvesting, share_classes

HELD = FUND_WEIGHTS + HOLDINGS


def through_a_split(ratio: float):
    from test_splits import END, START, build_fetcher, run_index

    fetcher = build_fetcher(ratio)

    return BacktestEngine(start_date=START, end_date=END, initial_capital=1_000_000.0,
                          data_provider=fetcher, index_result=run_index(fetcher),
                          calendar="XNYS").run()


RUNS = {"a 2-for-1 split": lambda: through_a_split(2.0),
        "a 1-for-2 split": lambda: through_a_split(0.5),
        "delisted": delisted, "reinvesting": reinvesting, "share classes": share_classes}


@pytest.fixture(scope="module", params=list(RUNS))
def run(request):
    return RunData.from_result(RUNS[request.param]())


class TestTheReplay:

    def test_the_holdings_come_back_exactly(self,
                                            run):
        tables = run.tables
        kept = pack_holdings(tables[HELD], tables[TRADES], tables[ACTIONS])

        assert kept is not None
        found = unpack_holdings(tables[TRADES], tables[ACTIONS], kept,
                                tables[HELD].dtypes.to_dict())
        assert found.equals(tables[HELD])
        assert found.index.equals(tables[HELD].index)

    def test_a_split_is_replayed_not_patched(self):
        tables = RunData.from_result(through_a_split(2.0)).tables

        kept = pack_holdings(tables[HELD], tables[TRADES], tables[ACTIONS])

        assert kept is not None and kept[PATCH].empty

    def test_a_stored_run_keeps_only_what_the_trades_do_not_give_back(self,
                                                                      run,
                                                                      tmp_path):
        run.write(tmp_path / "run")

        again = RunData.from_record(tmp_path / "run")

        assert not (tmp_path / "run" / f"{HELD}.parquet").exists()
        assert again.tables[HELD].equals(run.tables[HELD])
        assert again.tables[FUND_WEIGHTS].equals(run.tables[FUND_WEIGHTS])


class TestByHand:

    def test_a_split_then_the_day_s_trades_in_order(self):
        trades = pd.DataFrame({"IDENTIFIER": ["A", "A", "A", "B"],
                               "SIDE": ["BUY", "BUY", "SELL", "BUY"],
                               "QUANTITY": [10.0, 0.1, 4.0, 3.0]},
                              index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05",
                                                    "2024-01-05"]))
        actions = pd.DataFrame({"IDENTIFIER": ["A"], "EX_DATE": pd.to_datetime(["2024-01-05"]),
                                "TYPE": ["SPLIT"], "VALUE": [2.0], "STATUS": ["paid"]})

        found = replayed(trades, actions)

        assert list(found["NAME"]) == ["A", "A", "A", "B"]
        assert list(found["AMOUNT"]) == [10.0, 10.1, 10.1 * 2.0 - 4.0, 3.0]

    def test_a_holding_sold_out_ends(self):
        trades = pd.DataFrame({"IDENTIFIER": ["A", "A"], "SIDE": ["BUY", "SELL"],
                               "QUANTITY": [5.0, 5.0]},
                              index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
        empty = pd.DataFrame(columns=["IDENTIFIER", "EX_DATE", "TYPE", "VALUE"])

        found = replayed(trades, empty)

        assert list(found["HELD"]) == [True, False]
        assert found["AMOUNT"].isna().tolist() == [False, True]

    def test_rows_the_replay_gets_wrong_are_patched(self):
        trades = pd.DataFrame({"IDENTIFIER": ["A"], "SIDE": ["BUY"], "QUANTITY": [5.0]},
                              index=pd.to_datetime(["2024-01-02"]))
        empty = pd.DataFrame(columns=["IDENTIFIER", "EX_DATE", "TYPE", "VALUE"])
        recorded = replayed(trades, empty).assign(AMOUNT=[5.5])

        kept = pack_holdings(recorded, trades, empty)

        assert kept is not None and len(kept[PATCH]) == 1 and kept[DROPS].empty
        assert unpack_holdings(trades, empty, kept, recorded.dtypes.to_dict()).equals(recorded)
