# tests/test_shared_indices.py
"""BN-414: an index's results kept once in a data store's shared folder,
named by a fingerprint of the tables themselves."""
import pandas as pd

from beacon.analysis import shared_indices
from beacon.analysis.panels import INDEX_WEIGHTS
from beacon.analysis.shared_indices import fingerprint, index_tables, keep, keys, prune, read

DAYS = pd.bdate_range("2024-01-02", periods=4, name="DATE")


def results(weight: float = 0.5) -> dict[str, pd.DataFrame]:
    return {INDEX_WEIGHTS + "_holdings": pd.DataFrame({"NAME": ["A", "B"],
                                                       "AMOUNT": [10.0, 20.0],
                                                       "HELD": [True, True]},
                                                      index=DAYS[:2]),
            INDEX_WEIGHTS + "_patch": pd.DataFrame({"NAME": ["A"], "WEIGHT": [weight]},
                                                   index=DAYS[1:2])}


class TestKept:

    def test_once_for_the_same_results(self,
                                       tmp_path):
        first, second = keep(tmp_path, results()), keep(tmp_path, results())

        assert first == second and keys(tmp_path) == [first]

    def test_read_back_exactly(self,
                               tmp_path):
        key = keep(tmp_path, results())

        found = read(tmp_path, key, list(results()))

        for name, table in results().items():
            assert found[name].equals(table), name

    def test_other_results_are_another_entry(self,
                                             tmp_path):
        assert fingerprint(results()) != fingerprint(results(0.25))
        keep(tmp_path, results())
        keep(tmp_path, results(0.25))

        assert len(keys(tmp_path)) == 2

    def test_only_the_index_s_tables_are_taken(self):
        tables = {**results(), "fund_weights_holdings": pd.DataFrame()}

        assert set(index_tables(tables)) == set(results())


class TestPruning:

    def test_entries_no_run_uses_go_once_old_enough(self,
                                                    tmp_path,
                                                    monkeypatch):
        used, unused = keep(tmp_path, results()), keep(tmp_path, results(0.25))

        assert prune(tmp_path, {used}) == []

        monkeypatch.setattr(shared_indices, "GRACE_SECONDS", 0)
        assert prune(tmp_path, {used}) == [unused]
        assert keys(tmp_path) == [used]
