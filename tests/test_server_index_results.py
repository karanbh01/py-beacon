# tests/test_server_index_results.py
"""BN-414: an index's results within a run, kept once for every strategy
tracking it on the same data store over the same days.

test_server_market_inputs' two strategies track the same index; changing the
index's definition and backtesting one of them again gives a second entry,
and the other still reads the index it ran against.
"""
import pytest

from beacon.analysis import RunData
from beacon.analysis.panels import INDEX_WEIGHTS
from beacon.analysis.shared_indices import INDICES, PARTS, keys
from test_server_market_inputs import backtest, records, serving, shared
from test_server_strategy_runs import auth, index_document


@pytest.fixture
def client(tmp_path):
    yield from serving(tmp_path)


def own_copy(run: RunData,
             folder) -> RunData:
    """*run* written again keeping everything itself."""
    run.write(folder)

    return RunData.from_record(folder)


class TestKeptOnce:

    def test_two_strategies_on_one_index_share_one_copy(self,
                                                        client):
        for strategy in ("one", "two"):
            folder = records(client).panels_directory(strategy)
            assert not any((folder / f"{name}.parquet").exists() for name in PARTS), strategy

        assert len(keys(shared(client))) == 1

    def test_each_reads_its_index_as_a_run_keeping_its_own_does(self,
                                                                client,
                                                                tmp_path):
        for strategy in ("one", "two"):
            stored = RunData.from_record(records(client).panels_directory(strategy))
            own = own_copy(stored, tmp_path / strategy)

            assert stored.tables[INDEX_WEIGHTS].equals(own.tables[INDEX_WEIGHTS]), strategy
            assert not stored.tables[INDEX_WEIGHTS].empty


class TestAChangedDefinition:

    def test_is_a_new_copy_and_leaves_the_older_run_s_as_it_was(self,
                                                                client):
        before = RunData.from_record(records(client).panels_directory("one"))
        weights = before.tables[INDEX_WEIGHTS].copy()
        changed = index_document()
        changed["rebalancing_frequency"] = "MONTHLY"
        assert client.put("/indices/canon", json=changed, headers=auth()).status_code == 200

        backtest(client, "two")

        assert len(keys(shared(client))) == 2
        again = RunData.from_record(records(client).panels_directory("one"))
        assert again.tables[INDEX_WEIGHTS].equals(weights)
        newer = RunData.from_record(records(client).panels_directory("two"))
        assert not newer.tables[INDEX_WEIGHTS].equals(weights)


@pytest.mark.parametrize("cascade", [False, True])
def test_deleting_the_store_keeps_or_removes_them_with_the_inputs(client,
                                                                   cascade):
    client.post("/data/stores/other/activate", headers=auth())
    client.portal.call(client.app.state.jobs.drain)

    client.delete(f"/data/stores/canonical?cascade={str(cascade).lower()}", headers=auth())

    assert (shared(client) / INDICES).exists() is not cascade
