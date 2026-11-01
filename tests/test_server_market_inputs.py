# tests/test_server_market_inputs.py
"""BN-413: a stored run keeps its market inputs in its data store's shared
folder, once for every run on the store.

Two strategies on the canonical dataset, saved as a registered store and
served from it, are backtested; a second synthetic store is registered so the
first can be deleted.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.analysis import RunData
from beacon.analysis.data import RUN_FILE
from beacon.analysis.shared_inputs import SHARED
from beacon.analysis.shared_runs import FOLDER, UNATTRIBUTED, store_of
from beacon.data import store as data_store
from beacon.server import ServerConfig, create_app
from beacon.server.jobs import SUCCEEDED
from beacon.server.market_inputs import migrate
from beacon.testing import dataset
from test_server_strategy_runs import (
    END,
    START,
    auth,
    finished,
    index_document,
    strategy_document,
)

STRATEGIES = ("one", "two")


def backtest(client,
             strategy_id: str) -> None:
    job = finished(client, client.post(f"/strategies/{strategy_id}/backtest",
                                       json={"start": START, "end": END}, headers=auth()))
    assert job["status"] == SUCCEEDED, job


def activate(client,
             store_id: str) -> None:
    response = client.post(f"/data/stores/{store_id}/activate", headers=auth())
    assert response.status_code == 202, response.text
    client.portal.call(client.app.state.jobs.drain)


@pytest.fixture
def client(tmp_path):
    """A server serving the canonical dataset from a registered store, with
    both strategies backtested on it."""
    folder = data_store.save(dataset.data_fetcher(), tmp_path / "canonical")
    other = data_store.save(dataset.data_fetcher(), tmp_path / "other")
    config = ServerConfig(auth_token="test-token-value", storage_root=tmp_path / "documents")

    with TestClient(create_app(config), raise_server_exceptions=False) as started:
        for name, path in (("Canonical", folder), ("Other", other)):
            assert started.post("/data/stores", headers=auth(),
                                json={"name": name, "path": str(path)}).status_code == 201
        activate(started, "canonical")
        assert started.put("/indices/canon", json=index_document(),
                           headers=auth()).status_code == 200

        for strategy_id in STRATEGIES:
            assert started.post("/strategies", json=strategy_document(strategy_id),
                                headers=auth()).status_code == 200
            backtest(started, strategy_id)

        yield started


def records(client):
    return client.app.state.backtest_record_store


def shared(client, store_id: str = "canonical"):
    return records(client).directory.parent / FOLDER / store_id


class TestKeptOnce:

    def test_a_run_keeps_its_inputs_in_its_store_s_folder(self,
                                                          client):
        folder = records(client).panels_directory("one")
        names = {path.stem for path in folder.glob("*.parquet")}

        assert store_of(folder) == "canonical"
        assert not names & set(SHARED)
        assert (shared(client) / "catalogue.parquet").exists()

    def test_a_second_run_on_the_same_data_adds_nothing(self,
                                                        client):
        files = sorted(path.name for path in shared(client).iterdir())

        backtest(client, "one")

        assert sorted(path.name for path in shared(client).iterdir()) == files

    def test_its_tables_read_as_a_run_keeping_its_own_does(self,
                                                           client,
                                                           tmp_path):
        stored = RunData.from_record(records(client).panels_directory("two"))
        stored.write(tmp_path / "own")
        own = RunData.from_record(tmp_path / "own")

        assert set(stored.tables) == set(own.tables)
        for name in own.tables:
            assert stored.tables[name].equals(own.tables[name]), name
        assert not stored.tables["closes"].empty
        assert client.get("/strategies/two/record", headers=auth()).status_code == 200


class TestDeletingTheStore:

    def test_its_runs_keep_their_inputs_and_say_so(self,
                                                   client):
        activate(client, "other")

        response = client.delete("/data/stores/canonical", headers=auth())
        body = response.json()
        opened = RunData.from_record(records(client).panels_directory("one"))
        view = client.get("/strategies/one/analytics", headers=auth()).json()

        assert response.status_code == 200
        assert sorted(body["kept"]) == list(STRATEGIES) and body["deleted"] == []
        assert "2 stored runs" in body["warning"]
        assert shared(client).exists()
        assert opened.warnings and "Canonical" in opened.warnings[0]
        assert view["warnings"] == list(opened.warnings)
        assert not opened.tables["closes"].empty

    def test_with_cascade_its_runs_go_too(self,
                                          client):
        activate(client, "other")

        body = client.delete("/data/stores/canonical?cascade=true", headers=auth()).json()

        assert sorted(body["deleted"]) == list(STRATEGIES) and body["kept"] == []
        assert not records(client).exists("one")
        assert not shared(client).exists()


class TestRunsKeptBefore:

    def test_they_move_to_the_shared_layout_and_read_the_same(self,
                                                              client):
        folder = records(client).panels_directory("two")
        before = RunData.from_record(folder)
        tables = {name: before.tables[name] for name in before.tables}
        # Written again as before BN-413: its inputs in its own folder.
        before.write(folder)
        assert "inputs" not in (folder / RUN_FILE).read_text(encoding="utf-8")

        assert migrate(records(client)) == ["two"]

        after = RunData.from_record(folder)
        assert store_of(folder) == UNATTRIBUTED
        for name, table in tables.items():
            assert after.tables[name].equals(table), name
        assert migrate(records(client)) == []
