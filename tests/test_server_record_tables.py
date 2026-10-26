# tests/test_server_record_tables.py
"""BN-404: a backtest record's large parts kept as tables, and put back.

The round trip is checked on a finished run of the futures tests' three-name
index: the slim record, read back whole, is the record as it was stored
whole, to the last value.
"""
import json

import pytest

from beacon.analysis.data import RunData
from beacon.exceptions import ConfigurationError
from beacon.server.holdings import HOLDINGS_KEY, holdings_document
from beacon.server.record_tables import (
    RECORD_FOLDER,
    TABLES_KEY,
    frame_from,
    frame_table,
    slim_record,
    tree_from,
    tree_table,
    whole_record,
)
from beacon.server.schemas import BacktestResultSummary


def through_json(document):
    """*document* as the store writes and reads it."""
    return json.loads(json.dumps(document, sort_keys=True))


@pytest.fixture(scope="module")
def result():
    from test_strategy_futures import index, run

    return run(index())


@pytest.fixture(scope="module")
def stored(result,
           tmp_path_factory):
    """The whole record, the slim one as stored, and the run's folder."""
    folder = tmp_path_factory.mktemp("record") / "S.panels"
    run = RunData.from_result(result)
    run.write(folder)
    holdings = holdings_document(result, result.index.target.source)
    whole = {**BacktestResultSummary.from_result(result).model_dump(mode="json"),
             HOLDINGS_KEY: holdings}
    slim = {**BacktestResultSummary.from_result(result, weights=False).model_dump(mode="json"),
            HOLDINGS_KEY: holdings, "run_at": whole["run_at"]}

    return through_json(whole), through_json(slim_record(slim, run, folder)), folder


class TestTrees:

    def test_every_kind_of_value_comes_back(self):
        tree = {"a": [1, 2.5, None, True, False, "x"], "b": {}, "c": [], "d": {"e/f": [[0.1]]},
                "g": -0.0, "h": 10 ** 15}

        assert tree_from(tree_table({"root": tree})) == tree

    def test_a_list_at_the_top_comes_back(self):
        rows = [{"date": "2020-01-02", "weights": {"AAA": 0.25}}, {"date": "2020-04-01"}]

        assert tree_from(tree_table({"root": rows})) == rows


class TestFrames:

    def test_a_frame_comes_back_with_its_column_order(self):
        frame = {"index": [3, 4], "columns": ["date", "asset_id", "quantity"],
                 "data": [["2020-01-02T00:00:00", "AAA", 1.5],
                          ["2020-01-03T00:00:00", "BBB", None]]}

        assert frame_from(frame_table(frame)) == frame

    def test_a_column_of_text_and_numbers_is_not_a_frame_table(self):
        frame = {"index": [0, 1], "columns": ["mixed"], "data": [["a"], [1.0]]}

        assert frame_table(frame) is None


class TestRoundTrip:

    def test_the_slim_record_reads_back_as_the_whole_one(self,
                                                         stored):
        whole, slim, folder = stored

        assert whole_record(slim, folder) == whole

    def test_the_large_parts_are_left_empty_in_the_json(self,
                                                        stored):
        _, slim, _ = stored

        assert slim["portfolio"]["weights"]["data"] == []
        assert slim["portfolio"]["transactions"]["data"] == []
        assert slim["index"]["target"]["rebalances"] == []
        assert slim[HOLDINGS_KEY]["units"] == {}
        assert slim["portfolio"]["weights_dates_total"] > 0

    def test_weights_and_units_are_served_from_the_run_s_tables(self,
                                                                stored):
        _, slim, folder = stored
        parts = {tuple(part["path"]): part for part in slim[TABLES_KEY]["parts"]}

        assert parts[("portfolio", "weights")]["weights"] == "fund_weights"
        assert parts[("index", "target", "weights")]["weights"] == "index_weights"
        assert parts[(HOLDINGS_KEY, "units")]["units"] == "fund_weights_holdings"
        kept = {part.name for part in (folder / RECORD_FOLDER).iterdir()}
        assert len(kept) == sum(1 for part in parts.values() if "frame" in part or "tree" in part)

    def test_the_slim_record_still_satisfies_its_model(self,
                                                       stored):
        _, slim, _ = stored

        record = BacktestResultSummary.model_validate(slim)

        assert record.portfolio.weights.data == []

    def test_only_the_parts_asked_for_are_put_back(self,
                                                   stored):
        whole, slim, folder = stored

        back = whole_record(slim, folder, {HOLDINGS_KEY})

        assert back[HOLDINGS_KEY] == whole[HOLDINGS_KEY]
        assert back["portfolio"]["weights"]["data"] == []

    def test_a_record_kept_before_is_read_as_it_is(self,
                                                   stored,
                                                   tmp_path):
        whole, _, _ = stored

        assert whole_record(whole, tmp_path) is whole

    def test_missing_tables_are_reported(self,
                                         stored,
                                         tmp_path):
        _, slim, _ = stored

        with pytest.raises(ConfigurationError, match="tables"):
            whole_record(slim, tmp_path / "gone")


class TestWhatStaysInTheJson:

    def test_a_part_no_table_holds_stays_as_it_is(self,
                                                  stored,
                                                  result,
                                                  tmp_path):
        _, slim, folder = stored
        document = through_json(whole_record(slim, folder))
        document["index"]["target"]["rebalances"][0]["huge"] = 2 ** 70

        again = slim_record(document, RunData.from_result(result), tmp_path / "S.panels")
        moved = {tuple(part["path"]) for part in again[TABLES_KEY]["parts"]}

        assert ("index", "target", "rebalances") not in moved
        assert again["index"]["target"]["rebalances"][0]["huge"] == 2 ** 70

    def test_units_that_differ_from_the_holdings_are_kept_as_a_table(self,
                                                                     stored,
                                                                     result,
                                                                     tmp_path):
        _, slim, folder = stored
        document = through_json(whole_record(slim, folder))
        name = next(iter(document[HOLDINGS_KEY]["units"]))
        document[HOLDINGS_KEY]["units"][name][0][1] += 1.0
        run = RunData.from_result(result)
        run.write(tmp_path / "S.panels")

        again = through_json(slim_record(document, run, tmp_path / "S.panels"))

        assert whole_record(again, tmp_path / "S.panels")[HOLDINGS_KEY] == document[HOLDINGS_KEY]
