# tests/test_server_startup_load.py
"""BN-335: the engine announces its port first and loads its store after.

On the default synthetic store the load took 31 s, and a client waiting for
`BEACON_PORT` could not tell loading from hung. The registered store is now
loaded by the same job a store switch uses, with `/health` reporting
`loading` until it is served.
"""
import threading

import pytest
from fastapi.testclient import TestClient

from beacon.data import store as data_store
from beacon.server import ServerConfig, create_app
from beacon.server.data_stores import StoreRegistry, resolve_startup
from beacon.server.routers import stores
from test_server_data_stores import HEADERS, TOKEN, activate, register, write_store


@pytest.fixture
def documents(tmp_path):
    return tmp_path / "documents"


def registered(documents,
               tmp_path,
               names: list[str]) -> None:
    """A store named Mine holding *names*, active for the next start."""
    app = create_app(ServerConfig(auth_token=TOKEN, storage_root=documents))

    with TestClient(app) as client:
        register(client, "Mine", write_store(tmp_path / "a", names))
        activate(client, "mine")


def started(documents) -> TestClient:
    """A client over an engine started as the launcher starts it."""
    startup = resolve_startup(None, documents, load_now=False)

    return TestClient(create_app(ServerConfig(auth_token=TOKEN, storage_root=documents,
                                              startup_store=startup.pending)))


class TestResolvingWithoutLoading:

    def test_leaves_the_active_store_pending(self,
                                             documents,
                                             tmp_path):
        registered(documents, tmp_path, ["AAA", "BBB"])

        startup = resolve_startup(None, documents, load_now=False)

        assert startup.fetcher is None
        assert startup.pending is not None and startup.pending["id"] == "mine"
        assert "loading once the engine is serving" in startup.origin

    def test_still_loads_data_named_on_the_command_line(self,
                                                        documents,
                                                        tmp_path):
        """So a path that cannot be read still stops the engine at once."""
        explicit = write_store(tmp_path / "b", ["XXX"])

        startup = resolve_startup(explicit, documents, load_now=False)

        assert startup.fetcher is not None and startup.pending is None


class TestTheEngine:

    def test_reports_loading_and_then_serves_the_store(self,
                                                       documents,
                                                       tmp_path,
                                                       monkeypatch):
        registered(documents, tmp_path, ["AAA", "BBB"])
        load = stores.load
        asked = threading.Event()

        # Held until /health has been asked, as a store of any size would be.
        def held(record):
            asked.wait(10.0)
            return load(record)

        monkeypatch.setattr(stores, "load", held)

        with started(documents) as client:
            first = client.get("/health", headers=HEADERS).json()["data_source"]
            asked.set()
            client.portal.call(client.app.state.jobs.drain)
            served = client.get("/health", headers=HEADERS).json()["data_source"]

        assert first["loading"] is True and first["configured"] is False
        assert served["loading"] is False
        assert served["store_id"] == "mine" and served["identifiers"] == 2

    def test_answers_data_requests_with_no_data_until_then(self,
                                                           documents,
                                                           tmp_path):
        registered(documents, tmp_path, ["AAA"])

        with started(documents) as client:
            response = client.get("/data/identifiers", headers=HEADERS)
            client.portal.call(client.app.state.jobs.drain)

        assert response.status_code == 200

    def test_keeps_running_when_the_store_cannot_be_read(self,
                                                         documents,
                                                         tmp_path):
        registered(documents, tmp_path, ["AAA"])
        record = StoreRegistry(documents).get("mine")
        assert record is not None
        (tmp_path / "a" / data_store.MARKET_FILE).write_bytes(b"not gzip")

        with started(documents) as client:
            client.portal.call(client.app.state.jobs.drain)
            health = client.get("/health", headers=HEADERS)

        assert health.status_code == 200
        assert health.json()["data_source"]["loading"] is False
        assert health.json()["data_source"]["configured"] is False
