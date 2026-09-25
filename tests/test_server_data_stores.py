# tests/test_server_data_stores.py
"""BN-236: named data stores, and an engine that starts without data.

The engine used to take its data once, at startup, and keep it: started
without data, it stayed without, and every data endpoint answered 500. Now the
user registers named stores, one is active, loading one runs as a job, and the
choice is remembered for the next start.
"""
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data import store as data_store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.server.data_stores import StoreRegistry, resolve_startup
from beacon.testing import dataset

TOKEN = "stores-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def write_store(path: Path,
                names: list[str],
                source: str = data_store.SOURCE_LOCAL) -> Path:
    """A small store holding *names*, so two stores are told apart by them."""
    market = pd.DataFrame([{"IDENTIFIER": name, "DATE": day, "CLOSE": 100.0}
                           for name in names
                           for day in pd.bdate_range("2024-01-02", periods=5)])
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"} for name in names])
    fetcher = DataFetcher(MarketData.from_dataframe(market),
                          ReferenceData.from_dataframe(reference))

    return data_store.save(fetcher, path, source=source)


@pytest.fixture
def documents(tmp_path):
    return tmp_path / "documents"


@pytest.fixture
def client(documents):
    """An engine started with no data."""
    app = create_app(ServerConfig(auth_token=TOKEN, storage_root=documents))

    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


def register(client,
             name: str,
             path: Path):
    return client.post("/data/stores", headers=HEADERS,
                       json={"name": name, "path": str(path)})


def activate(client,
             store_id: str) -> dict:
    """Load a store and wait for the job, returning the finished job."""
    response = client.post(f"/data/stores/{store_id}/activate", headers=HEADERS)
    assert response.status_code == 202, response.json()

    client.portal.call(client.app.state.jobs.drain)

    return client.get(f"/jobs/{response.json()['job_id']}", headers=HEADERS).json()


def served(client) -> dict:
    return client.get("/health", headers=HEADERS).json()["data_source"]


class TestAnEngineWithoutData:

    def test_it_starts_and_says_nothing_is_loaded(self,
                                                  client):
        assert served(client) == {"configured": False, "identifiers": 0,
                                  "store_id": None, "store_name": None,
                                  "loading": False}

    def test_a_data_endpoint_refuses_with_409_saying_what_needs_data(self,
                                                                    client):
        """Not a server fault: the caller fixes it by loading a store."""
        response = client.get("/data/prices/AAA", headers=HEADERS)

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "NO_DATA_LOADED"
        assert "Load a data store" in response.json()["error"]["message"]

    def test_endpoints_that_need_no_data_still_work(self,
                                                    client):
        assert client.get("/universes", headers=HEADERS).status_code == 200
        assert client.get("/data/stores", headers=HEADERS).status_code == 200


class TestRegistering:

    def test_a_folder_store_is_registered_with_an_id_from_its_name(self,
                                                                   client,
                                                                   tmp_path):
        response = register(client, "My Data", write_store(tmp_path / "a", ["AAA"]))
        body = response.json()

        assert response.status_code == 201
        assert body["id"] == "my-data"
        assert body["readable"] is True
        assert body["active"] is False
        assert body["source"] == "local"

    def test_the_same_name_twice_gets_a_numbered_id(self,
                                                    client,
                                                    tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA"]))
        second = register(client, "Mine", write_store(tmp_path / "b", ["BBB"]))

        assert second.json()["id"] == "mine-2"

    @pytest.mark.parametrize("path", ["relative/folder", "missing"])
    def test_a_folder_that_is_not_a_store_is_refused_with_a_finding(self,
                                                                    client,
                                                                    tmp_path,
                                                                    path):
        target = path if path.startswith("relative") else str(tmp_path / path)
        response = client.post("/data/stores", headers=HEADERS,
                               json={"name": "Bad", "path": target})
        findings = response.json()["error"]["detail"]["findings"]

        assert response.status_code == 422
        assert findings[0]["code"] == "NOT_A_DATA_STORE"

    def test_a_folder_registered_twice_is_a_conflict(self,
                                                     client,
                                                     tmp_path):
        folder = write_store(tmp_path / "a", ["AAA"])
        register(client, "First", folder)

        response = register(client, "Second", folder)

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]["message"]


class TestLoading:

    def test_loading_serves_the_store_and_says_so(self,
                                                  client,
                                                  tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA", "BBB"]))

        job = activate(client, "mine")

        assert job["status"] == "succeeded", job
        assert job["result"]["identifiers"] == 2
        assert served(client)["store_id"] == "mine"
        assert served(client)["store_name"] == "Mine"
        assert client.get("/data/prices/AAA", headers=HEADERS).status_code == 200

    def test_switching_stores_changes_the_data_and_the_global_universe(self,
                                                                       client,
                                                                       tmp_path):
        register(client, "One", write_store(tmp_path / "a", ["AAA"]))
        register(client, "Two", write_store(tmp_path / "b", ["XXX", "YYY"]))

        activate(client, "one")
        activate(client, "two")

        members = client.get("/universes/GLOBAL", headers=HEADERS).json()
        listing = client.get("/data/stores", headers=HEADERS).json()

        assert served(client)["identifiers"] == 2
        assert members["identifiers"] == ["XXX", "YYY"]
        assert listing["active"] == "two"
        assert [store["active"] for store in listing["stores"]] == [False, True]

    def test_loading_announces_the_new_data_on_the_event_socket(self,
                                                                client,
                                                                tmp_path,
                                                                monkeypatch):
        """Everything a client derived from the old data may be stale."""
        announced = []
        monkeypatch.setattr(client.app.state.jobs, "publish_data_loaded",
                            lambda store_id, name: announced.append((store_id, name)))
        register(client, "Mine", write_store(tmp_path / "a", ["AAA"]))

        activate(client, "mine")

        assert announced == [("mine", "Mine")]

    def test_a_second_load_while_one_runs_is_refused(self,
                                                     client,
                                                     tmp_path):
        register(client, "One", write_store(tmp_path / "a", ["AAA"]))
        client.app.state.active_data.loading = True

        response = client.post("/data/stores/one/activate", headers=HEADERS)

        assert response.status_code == 409

    def test_a_store_that_cannot_be_read_is_refused_before_a_job_starts(self,
                                                                        client,
                                                                        tmp_path):
        folder = write_store(tmp_path / "a", ["AAA"])
        register(client, "Gone", folder)
        (folder / data_store.MARKET_FILE).unlink()

        response = client.post("/data/stores/gone/activate", headers=HEADERS)

        assert response.status_code == 409
        listing = client.get("/data/stores", headers=HEADERS).json()["stores"]
        assert listing[0]["readable"] is False

    def test_a_failed_load_keeps_the_data_already_served(self,
                                                         client,
                                                         tmp_path):
        register(client, "Good", write_store(tmp_path / "a", ["AAA"]))
        broken = write_store(tmp_path / "b", ["BBB"])
        register(client, "Broken", broken)
        activate(client, "good")
        (broken / data_store.MARKET_FILE).write_bytes(b"not gzip")

        job = activate(client, "broken")

        assert job["status"] == "failed"
        assert served(client)["store_id"] == "good"
        assert served(client)["loading"] is False


class TestManaging:

    def test_renaming_keeps_the_id_and_updates_what_health_shows(self,
                                                                 client,
                                                                 tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA"]))
        activate(client, "mine")

        response = client.patch("/data/stores/mine", headers=HEADERS,
                                json={"name": "Renamed"})

        assert response.json()["id"] == "mine"
        assert response.json()["name"] == "Renamed"
        assert served(client)["store_name"] == "Renamed"

    def test_the_store_being_served_cannot_be_forgotten(self,
                                                        client,
                                                        tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA"]))
        activate(client, "mine")

        assert client.delete("/data/stores/mine", headers=HEADERS).status_code == 409

    def test_forgetting_a_store_leaves_its_folder_alone(self,
                                                        client,
                                                        tmp_path):
        folder = write_store(tmp_path / "a", ["AAA"])
        register(client, "Mine", folder)

        response = client.delete("/data/stores/mine", headers=HEADERS)

        assert response.status_code == 204
        assert data_store.exists(folder)
        assert client.get("/data/stores/mine", headers=HEADERS).status_code == 404


class TestStartup:
    """What the engine serves when it starts, in order."""

    def test_the_active_store_is_served_on_the_next_start(self,
                                                          client,
                                                          documents,
                                                          tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA", "BBB"]))
        activate(client, "mine")

        startup = resolve_startup(None, documents)

        assert startup.store_id == "mine"
        assert startup.fetcher is not None
        assert len(startup.fetcher.identifiers) == 2

    def test_data_named_on_the_command_line_wins(self,
                                                 client,
                                                 documents,
                                                 tmp_path):
        register(client, "Mine", write_store(tmp_path / "a", ["AAA"]))
        activate(client, "mine")
        explicit = write_store(tmp_path / "b", ["XXX", "YYY", "ZZZ"])

        startup = resolve_startup(explicit, documents)

        assert startup.store_id is None
        assert len(startup.fetcher.identifiers) == 3

    def test_an_active_store_that_cannot_be_read_starts_the_engine_empty(self,
                                                                        client,
                                                                        documents,
                                                                        tmp_path):
        """A moved or damaged store must never stop the engine starting."""
        folder = write_store(tmp_path / "a", ["AAA"])
        register(client, "Mine", folder)
        activate(client, "mine")
        (folder / data_store.MARKET_FILE).write_bytes(b"not gzip")

        startup = resolve_startup(None, documents)

        assert startup.fetcher is None
        assert "unreadable" in startup.origin

    def test_an_existing_default_store_is_adopted_as_synthetic_data(self,
                                                                    documents,
                                                                    tmp_path,
                                                                    monkeypatch):
        """An install from before named stores keeps serving its data."""
        default = write_store(tmp_path / "default", ["AAA"],
                              source=data_store.SOURCE_SYNTHETIC)
        monkeypatch.setattr(data_store, "default_path", lambda: default)

        startup = resolve_startup(None, documents)

        assert startup.store_name == "Synthetic data"
        assert StoreRegistry(documents).active_id() == startup.store_id

    def test_nothing_registered_and_no_default_store_starts_empty(self,
                                                                  documents,
                                                                  tmp_path,
                                                                  monkeypatch):
        monkeypatch.setattr(data_store, "default_path", lambda: tmp_path / "none")

        assert resolve_startup(None, documents).fetcher is None


class TestTheRegistryStaysReadable:

    def test_two_stores_flagged_active_resolve_to_the_latest_loaded(self,
                                                                    documents,
                                                                    tmp_path):
        """A crash between the two writes in set_active leaves both flagged;
        the store loaded most recently is the one meant."""
        registry = StoreRegistry(documents)
        first = registry.create("One", write_store(tmp_path / "a", ["AAA"]))
        second = registry.create("Two", write_store(tmp_path / "b", ["BBB"]))
        registry.set_active(first["id"])
        registry.mark_loaded(first["id"])
        record = registry.get(second["id"])
        record["active"] = True
        record["last_loaded_at"] = "2999-01-01T00:00:00+00:00"
        registry._write(record)

        assert registry.active_id() == second["id"]

    def test_the_canonical_sample_data_loads_as_a_store(self,
                                                        client,
                                                        tmp_path):
        """The real dataset, not only the small ones above."""
        folder = data_store.save(dataset.data_fetcher(), tmp_path / "canon")
        register(client, "Canonical", folder)

        job = activate(client, "canonical")

        assert job["status"] == "succeeded", job
        assert served(client)["identifiers"] == len(dataset.data_fetcher().identifiers)
