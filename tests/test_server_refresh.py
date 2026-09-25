# tests/test_server_refresh.py
"""BN-240: refreshing a store from its own source.

Sync used to download from Yahoo Finance into memory, whatever the data was.
Now a refresh brings a store up to date from wherever it came from, saves the
result, and, if the store is being served, serves it. Yahoo Finance is an
option a folder store can choose, never the default.
"""
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data import store as data_store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import MARKET_DATASET, DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, write
from beacon.synthetic import state as synthetic_state

TOKEN = "refresh-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

DAYS = pd.bdate_range("2024-01-02", periods=5)
NEW_DAYS = pd.bdate_range("2024-01-09", periods=3, tz="UTC")
SYNTHETIC = SyntheticConfig(assets=12, start="2024-01-02", end="2024-03-28",
                            features=False)


def write_store(path: Path,
                names: list[str],
                source: str = data_store.SOURCE_LOCAL) -> Path:
    market = pd.DataFrame([{"IDENTIFIER": name, "DATE": day, "CLOSE": 100.0}
                           for name in names for day in DAYS])
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"} for name in names])
    fetcher = DataFetcher(MarketData.from_dataframe(market),
                          ReferenceData.from_dataframe(reference))

    return data_store.save(fetcher, path, source=source)


def downloader(identifier: str,
               start: str | None = None,
               end: str | None = None) -> pd.DataFrame:
    """Stands in for Yahoo Finance: three days after the stores above end."""
    return pd.DataFrame({"Open": 1.0, "High": 2.0, "Low": 0.5, "Close": 101.0,
                         "Volume": 10}, index=NEW_DAYS)


@pytest.fixture(scope="module")
def synthetic_template(tmp_path_factory) -> Path:
    return write(SYNTHETIC, tmp_path_factory.mktemp("synthetic") / "store")


@pytest.fixture
def synthetic(synthetic_template,
              tmp_path) -> Path:
    return Path(shutil.copytree(synthetic_template, tmp_path / "synthetic"))


@pytest.fixture
def client(tmp_path):
    app = create_app(ServerConfig(auth_token=TOKEN,
                                  storage_root=tmp_path / "documents",
                                  market_downloader=downloader))

    with TestClient(app, raise_server_exceptions=False) as started:
        yield started


def register(client,
             path: Path,
             name: str = "My data",
             **extra):
    response = client.post("/data/stores", headers=HEADERS,
                           json={"name": name, "path": str(path), **extra})
    assert response.status_code == 201, response.json()

    return response.json()


def run(client,
        response) -> dict:
    """Wait for the job a response started, and return it finished."""
    assert response.status_code == 202, response.json()
    client.portal.call(client.app.state.jobs.drain)

    return client.get(f"/jobs/{response.json()['job_id']}", headers=HEADERS).json()


def activate(client,
             store_id: str) -> None:
    job = run(client, client.post(f"/data/stores/{store_id}/activate",
                                  headers=HEADERS))
    assert job["status"] == "succeeded", job


def refresh(client,
            store_id: str,
            **body):
    return client.post(f"/data/stores/{store_id}/refresh", headers=HEADERS,
                       json=body or None)


def served(client) -> dict:
    return client.get("/health", headers=HEADERS).json()["data_source"]


class TestWhatARefreshWouldDo:

    def test_each_kind_of_store_says(self,
                                     client,
                                     tmp_path,
                                     synthetic):
        folder = register(client, write_store(tmp_path / "a", ["AAA"]), "A")
        imported = register(client, write_store(tmp_path / "b", ["BBB"],
                                                data_store.SOURCE_IMPORTED), "B")
        yahoo = register(client, write_store(tmp_path / "c", ["CCC"]), "C",
                         refresh_from="yfinance")
        generated = register(client, synthetic, "D")

        assert folder["refresh"] == "reread"
        assert imported["refresh"] is None
        assert yahoo["refresh"] == "download"
        assert yahoo["refresh_from"] == "yfinance"
        assert generated["refresh"] == "extend"

    def test_synthetic_data_too_old_to_extend_says_so(self,
                                                      client,
                                                      synthetic):
        shutil.rmtree(synthetic / synthetic_state.STATE_DIRECTORY)
        store = register(client, synthetic)

        response = refresh(client, store["id"])

        assert store["refresh"] is None
        assert response.status_code == 409
        assert "before py-beacon 0.1.2" in response.json()["error"]["message"]

    def test_imported_files_have_nothing_to_refresh(self,
                                                    client,
                                                    tmp_path):
        store = register(client, write_store(tmp_path / "b", ["BBB"],
                                             data_store.SOURCE_IMPORTED))

        response = refresh(client, store["id"])

        assert response.status_code == 409
        assert "Import them again" in response.json()["error"]["message"]


class TestRereading:

    def test_a_served_folder_picks_up_changes_made_outside(self,
                                                           client,
                                                           tmp_path):
        path = write_store(tmp_path / "a", ["AAA"])
        store = register(client, path)
        activate(client, store["id"])
        before = served(client)["data_version"]

        write_store(path, ["AAA", "BBB"])
        job = run(client, refresh(client, store["id"]))

        assert job["status"] == "succeeded", job
        assert job["kind"] == f"refresh:{store['id']}"
        assert job["result"]["action"] == "reread"
        assert job["result"]["served"] is True
        assert served(client)["identifiers"] == 2
        assert served(client)["data_version"] != before

    def test_a_folder_not_being_served_has_nothing_to_reread(self,
                                                             client,
                                                             tmp_path):
        store = register(client, write_store(tmp_path / "a", ["AAA"]))

        response = refresh(client, store["id"])

        assert response.status_code == 409
        assert "read afresh" in response.json()["error"]["message"]


class TestExtending:

    def test_served_synthetic_data_is_extended_and_served(self,
                                                          client,
                                                          synthetic):
        store = register(client, synthetic)
        activate(client, store["id"])
        before = served(client)["data_version"]

        job = run(client, refresh(client, store["id"], end="2024-04-30"))

        assert job["status"] == "succeeded", job
        assert job["result"]["action"] == "extend"
        assert job["result"]["end"] == "2024-04-30"
        assert job["result"]["served"] is True
        assert served(client)["data_version"] != before

    def test_a_store_not_served_is_extended_on_disk(self,
                                                    client,
                                                    synthetic):
        store = register(client, synthetic)

        job = run(client, refresh(client, store["id"], end="2024-04-30"))
        settings, _ = synthetic_state.load(synthetic)

        assert job["status"] == "succeeded", job
        assert job["result"]["served"] is False
        assert job["result"]["end"] == "2024-04-30"
        assert settings.extensions[-1]["to"] == "2024-04-30"


class TestYahooFinance:

    def test_a_folder_that_chose_it_downloads_and_saves(self,
                                                        client,
                                                        tmp_path):
        path = write_store(tmp_path / "a", ["AAA", "BBB"])
        store = register(client, path, refresh_from="yfinance")

        job = run(client, refresh(client, store["id"]))
        saved = data_store.load(path)

        assert job["status"] == "succeeded", job
        assert job["result"]["action"] == "download"
        assert job["result"]["rows_added"] == 6
        assert job["result"]["end"] == "2024-01-11"
        assert saved.date_range[1] == pd.Timestamp("2024-01-11")

    def test_it_can_be_chosen_later(self,
                                    client,
                                    tmp_path):
        store = register(client, write_store(tmp_path / "a", ["AAA"]))

        updated = client.patch(f"/data/stores/{store['id']}", headers=HEADERS,
                               json={"refresh_from": "yfinance"}).json()

        assert updated["refresh_from"] == "yfinance"
        assert updated["refresh"] == "download"
        assert updated["name"] == "My data"

    def test_synthetic_data_cannot_choose_it(self,
                                             client,
                                             synthetic):
        store = register(client, synthetic)

        response = client.patch(f"/data/stores/{store['id']}", headers=HEADERS,
                                json={"refresh_from": "yfinance"})

        assert response.status_code == 422
        assert response.json()["error"]["detail"]["findings"][0]["code"] == (
            "REFRESH_SOURCE_UNSUITABLE")

    def test_a_database_cannot_choose_it(self,
                                         client):
        response = client.post("/data/stores", headers=HEADERS, json={
            "name": "db", "kind": "postgres", "refresh_from": "yfinance",
            "connection": {"host": "h", "database": "d", "user": "u"}})

        assert response.status_code == 422


class TestOneAtATime:

    def test_a_refreshing_store_cannot_be_loaded_forgotten_or_refreshed(
            self,
            client,
            synthetic):
        store = register(client, synthetic)
        client.app.state.refreshing.add(store["id"])

        paths = [("post", f"/data/stores/{store['id']}/activate"),
                 ("delete", f"/data/stores/{store['id']}"),
                 ("post", f"/data/stores/{store['id']}/refresh")]

        for method, path in paths:
            assert client.request(method, path,
                                  headers=HEADERS).status_code == 409, path

    def test_the_mark_clears_when_the_refresh_ends(self,
                                                   client,
                                                   synthetic):
        store = register(client, synthetic)

        run(client, refresh(client, store["id"], end="2024-04-05"))

        assert store["id"] not in client.app.state.refreshing


class TestTheDeprecatedSync:

    def test_it_refreshes_the_store_being_served(self,
                                                 client,
                                                 tmp_path):
        store = register(client, write_store(tmp_path / "a", ["AAA"]),
                         refresh_from="yfinance")
        activate(client, store["id"])

        job = run(client, client.post("/data/coverage/market/sync",
                                      headers=HEADERS,
                                      json={"identifiers": ["IGNORED"]}))

        assert job["status"] == "succeeded", job
        assert job["kind"] == f"refresh:{store['id']}"
        assert job["result"]["rows_added"] == 3

    def test_it_publishes_the_freshness_event(self,
                                              client,
                                              tmp_path):
        store = register(client, write_store(tmp_path / "a", ["AAA"]))
        activate(client, store["id"])
        registry = client.app.state.jobs
        queue = client.portal.call(_subscribe, registry)

        run(client, client.post("/data/coverage/market/sync", headers=HEADERS))

        events = client.portal.call(_drain_queue, queue)
        kinds = [event["type"] for event in events]

        assert "data.loaded" in kinds
        assert "data.freshness" in kinds

    def test_it_resets_the_reported_age(self,
                                        client,
                                        tmp_path):
        store = register(client, write_store(tmp_path / "a", ["AAA"]))
        activate(client, store["id"])
        client.app.state.active_data.fetcher.record_refresh(
            MARKET_DATASET, pd.Timestamp("2020-01-01", tz="UTC").to_pydatetime())

        run(client, client.post("/data/coverage/market/sync", headers=HEADERS))
        datasets = client.get("/data/coverage", headers=HEADERS).json()["datasets"]
        market = next(d for d in datasets if d["dataset"] == "market")

        assert market["cache_age"] < 60

    def test_data_named_at_startup_cannot_be_synced(self):
        market = MarketData.from_dataframe(pd.DataFrame(
            {"IDENTIFIER": "AAA", "DATE": DAYS, "CLOSE": 1.0}))
        app = create_app(ServerConfig(auth_token=TOKEN,
                                      data_fetcher=DataFetcher(market)))

        with TestClient(app, raise_server_exceptions=False) as started:
            response = started.post("/data/coverage/market/sync",
                                    headers=HEADERS)

        assert response.status_code == 409
        assert "Register it" in response.json()["error"]["message"]

    def test_it_is_marked_deprecated(self,
                                     client):
        spec = client.get("/openapi.json", headers=HEADERS).json()

        assert spec["paths"]["/data/coverage/{dataset}/sync"]["post"][
            "deprecated"] is True


async def _subscribe(registry):
    return registry.subscribe()


async def _drain_queue(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    return events
