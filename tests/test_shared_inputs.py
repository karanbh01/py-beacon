# tests/test_shared_inputs.py
"""BN-413: a data store's market inputs, kept once for every run on it.

Tables in each layout a run keeps (closes and exchange rates by date and
name; traded values, characteristics and reference data as dated rows;
corporate actions; currencies) go into a shared folder as pieces of a name's
year, and come back exactly.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.analysis.panels import (
    ACTIONS,
    CHARACTERISTICS,
    CLOSES,
    CURRENCIES,
    RATES,
    REFERENCE,
    TRADED_VALUES,
)
from beacon.analysis.shared_inputs import MERGE_AFTER, SharedInputs

DAYS = pd.bdate_range("2023-11-01", "2025-02-28", name="DATE")
NAMES = ["A", "B", "C"]
MONTH_ENDS = pd.DatetimeIndex(["2023-11-30", "2024-06-28", "2025-01-31"])


def closes(days=DAYS, names=NAMES, bump: float = 0.0) -> pd.DataFrame:
    grid = 100.0 + np.arange(len(days))[:, None] * 0.01 + np.arange(len(names)) + bump
    frame = pd.DataFrame(np.round(grid, 2), index=days, columns=names)
    frame.iloc[:5, 2] = np.nan

    return frame


def dated(columns: dict[str, list]) -> pd.DataFrame:
    rows = [(day, name) for day in MONTH_ENDS for name in NAMES]
    frame = pd.DataFrame({"IDENTIFIER": [name for _, name in rows],
                          **{column: values * len(MONTH_ENDS)
                             for column, values in columns.items()}},
                         index=pd.DatetimeIndex([day for day, _ in rows]))

    return frame


def tables(**changes) -> dict[str, pd.DataFrame]:
    found = {
        CLOSES: closes(),
        RATES: pd.DataFrame({"EUR": 1.1, "GBP": 1.3}, index=DAYS),
        TRADED_VALUES: dated({"TRADED_VALUE": [1e6, 2e6, np.nan]}),
        CHARACTERISTICS: dated({"MARKET_CAP": [1e9, 2e9, 3e9], "PE_RATIO": [10.0, np.nan, 12.0]}),
        REFERENCE: dated({"SECTOR": ["Tech", "Energy", "Health"]}),
        ACTIONS: pd.DataFrame({"IDENTIFIER": ["A", "A", "C"],
                               "EX_DATE": pd.to_datetime(["2024-03-01", "2024-03-01",
                                                          "2024-09-02"]),
                               "TYPE": ["SPLIT", "CASH", "CASH"], "VALUE": [2.0, 0.5, 0.25],
                               "STATUS": ["paid"] * 3}),
        CURRENCIES: pd.DataFrame({"CURRENCY": ["USD", "EUR", "GBP"]}, index=NAMES),
    }
    found.update(changes)

    return found


def read_all(shared: SharedInputs,
             pieces: pd.DataFrame,
             layouts: dict) -> dict[str, pd.DataFrame]:
    return {table: shared.read(table, pieces.loc[pieces["table"] == table, "fingerprint"],
                               layouts[table]) for table in layouts}


def assert_same(found: pd.DataFrame,
                expected: pd.DataFrame) -> None:
    assert found.equals(expected)
    assert found.index.equals(expected.index)
    assert found.index.name == expected.index.name
    assert list(found.dtypes.astype(str)) == list(expected.dtypes.astype(str))


class TestTheRoundTrip:

    def test_every_table_comes_back_exactly(self,
                                            tmp_path):
        shared = SharedInputs(tmp_path)
        given = tables()

        pieces, layouts = shared.add(given)

        for table, frame in read_all(shared, pieces, layouts).items():
            assert_same(frame, given[table])

    def test_a_piece_is_a_name_s_year(self,
                                      tmp_path):
        pieces, _ = SharedInputs(tmp_path).add({CLOSES: closes()})
        catalogue = SharedInputs(tmp_path).catalogue()

        # Three names over 2023, 2024 and 2025.
        assert len(pieces) == 9
        assert sorted(set(catalogue["year"])) == [2023, 2024, 2025]

    def test_an_empty_table_comes_back_empty(self,
                                             tmp_path):
        shared = SharedInputs(tmp_path)
        empty = {table: frame.iloc[0:0] for table, frame in tables().items()}

        pieces, layouts = shared.add(empty)

        assert pieces.empty
        for table, frame in read_all(shared, pieces, layouts).items():
            assert_same(frame, empty[table])

    def test_rows_in_neither_order_come_back_in_theirs(self,
                                                       tmp_path):
        shuffled = tables()[REFERENCE].iloc[[4, 0, 8, 2, 6, 1, 3, 7, 5]]
        shared = SharedInputs(tmp_path)

        pieces, layouts = shared.add({REFERENCE: shuffled})

        assert_same(read_all(shared, pieces, layouts)[REFERENCE], shuffled)


class TestSharing:

    def test_a_repeat_run_adds_nothing(self,
                                       tmp_path):
        shared = SharedInputs(tmp_path)
        shared.add(tables())
        files = shared.files()

        again, _ = shared.add(tables())

        assert shared.files() == files
        assert len(again) == len(shared.catalogue())

    def test_a_longer_run_adds_the_earlier_year(self,
                                                tmp_path):
        shared = SharedInputs(tmp_path)
        longer = closes()
        shared.add({CLOSES: longer[longer.index >= "2024-01-01"]})
        before = len(shared.catalogue())

        # The longer run's 2024 and 2025 are the same pieces; 2023 is new.
        shared.add({CLOSES: longer})

        assert len(shared.catalogue()) == before + 3

    def test_a_revised_price_leaves_an_older_run_s_prices_as_they_were(self,
                                                                       tmp_path):
        shared = SharedInputs(tmp_path)
        old = closes()
        old_pieces, old_layouts = shared.add({CLOSES: old})
        revised = old.copy()
        revised.loc["2024-05-01", "B"] = 1.0

        new_pieces, new_layouts = shared.add({CLOSES: revised})

        assert_same(read_all(shared, old_pieces, old_layouts)[CLOSES], old)
        assert_same(read_all(shared, new_pieces, new_layouts)[CLOSES], revised)
        # Only B's 2024 is a new piece.
        assert len(shared.catalogue()) == len(old_pieces) + 1


class TestMerging:

    def test_a_merge_keeps_what_runs_use_and_reads_the_same(self,
                                                            tmp_path):
        shared = SharedInputs(tmp_path)
        kept, layouts = shared.add(tables())
        shared.add(tables(**{CLOSES: closes(bump=5.0)}))

        shared.merge(set(zip(kept["table"], kept["fingerprint"], strict=True)))

        assert len(shared.files()) == len(layouts)
        assert len(shared.catalogue()) == len(kept)
        for table, frame in read_all(shared, kept, layouts).items():
            assert_same(frame, tables()[table])

    def test_it_is_due_after_twenty_files(self,
                                          tmp_path):
        shared = SharedInputs(tmp_path)

        for bump in range(MERGE_AFTER + 1):
            shared.add({CLOSES: closes(bump=float(bump))})

        assert shared.due_a_merge()

    def test_a_piece_the_folder_lacks_is_named(self,
                                               tmp_path):
        shared = SharedInputs(tmp_path)
        pieces, layouts = shared.add({CLOSES: closes()})
        shared.merge(set())

        with pytest.raises(KeyError, match="closes pieces"):
            shared.read(CLOSES, pieces["fingerprint"], layouts[CLOSES])


class TestADeletedStore:

    def test_the_folder_says_it_was_kept_for_one(self,
                                                 tmp_path):
        shared = SharedInputs(tmp_path)
        assert shared.deleted() is None

        shared.mark_deleted({"id": "s1", "name": "Mine"})

        assert shared.deleted() == {"id": "s1", "name": "Mine"}
