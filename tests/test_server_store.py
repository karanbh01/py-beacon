# tests/test_server_store.py
"""Unit tests for DocumentStore, the watchlist endpoints, and data coverage."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import ConfigurationError
from beacon.server import ServerConfig, create_app
from beacon.server import store as store_module
from beacon.server.store import CURRENT_SCHEMA_VERSION, SCHEMA_VERSION_KEY, DocumentStore

TOKEN = "test-token-value"
ASSETS = ["AAA", "BBB", "CCC"]
DATES = pd.bdate_range("2025-01-02", periods=15)


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data."""
    market = pd.DataFrame([
        {"IDENTIFIER": asset, "DATE": date, "CLOSE": 100.0 + index}
        for asset in ASSETS
        for index, date in enumerate(DATES)
    ])
    reference = pd.DataFrame([
        {"IDENTIFIER": asset, "DATE_FROM": "2020-01-01", "NAME": asset}
        for asset in ASSETS[:2]
    ])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


@pytest.fixture
def store(tmp_path) -> DocumentStore:
    """A store rooted in a temporary directory."""
    return DocumentStore("widgets", root=tmp_path)


@pytest.fixture
def client(tmp_path) -> TestClient:
    """Client whose storage and data both exist."""
    config = ServerConfig(auth_token=TOKEN,
                          data_fetcher=build_fetcher(),
                          storage_root=tmp_path)
    return TestClient(create_app(config), raise_server_exceptions=False)


class TestDocumentStore:

    def test_read_missing_returns_none(self,
                                       store):
        assert store.read("absent") is None

    def test_write_then_read_round_trips(self,
                                         store):
        store.write("one", {"id": "one", "value": 42})

        assert store.read("one")["value"] == 42

    def test_write_stamps_the_schema_version(self,
                                             store):
        stored = store.write("one", {"id": "one"})

        assert stored[SCHEMA_VERSION_KEY] == CURRENT_SCHEMA_VERSION
        assert store.read("one")[SCHEMA_VERSION_KEY] == CURRENT_SCHEMA_VERSION

    def test_write_is_atomic_leaving_no_temporary_files(self,
                                                        store):
        store.write("one", {"id": "one"})

        assert [p.name for p in store.directory.iterdir()] == ["one.json"]

    def test_overwrite_replaces_content(self,
                                        store):
        store.write("one", {"id": "one", "value": 1})
        store.write("one", {"id": "one", "value": 2})

        assert store.read("one")["value"] == 2

    def test_delete_reports_whether_it_removed_anything(self,
                                                        store):
        store.write("one", {"id": "one"})

        assert store.delete("one") is True
        assert store.delete("one") is False

    def test_list_ids_is_sorted(self,
                                store):
        for name in ("b", "a", "c"):
            store.write(name, {"id": name})

        assert store.list_ids() == ["a", "b", "c"]

    def test_read_all_returns_every_document(self,
                                             store):
        store.write("a", {"id": "a"})
        store.write("b", {"id": "b"})

        assert {doc["id"] for doc in store.read_all()} == {"a", "b"}

    def test_collections_do_not_collide(self,
                                        tmp_path):
        first = DocumentStore("alpha", root=tmp_path)
        second = DocumentStore("beta", root=tmp_path)

        first.write("shared", {"id": "shared", "owner": "alpha"})
        second.write("shared", {"id": "shared", "owner": "beta"})

        assert first.read("shared")["owner"] == "alpha"
        assert second.read("shared")["owner"] == "beta"

    @pytest.mark.parametrize("bad_id", ["../escape", "nested/child", ""])
    def test_rejects_path_like_ids(self,
                                   store,
                                   bad_id):
        """Ids arrive from a URL, so one must not write outside the store."""
        with pytest.raises((ConfigurationError, ValueError)):
            store.write(bad_id, {"id": bad_id})

    def test_corrupt_json_is_a_configuration_error(self,
                                                   store):
        (store.directory / "broken.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="not valid JSON"):
            store.read("broken")

    def test_future_schema_version_is_refused(self,
                                              store):
        """A document from a newer build must not be silently misread."""
        payload = {"id": "one", SCHEMA_VERSION_KEY: CURRENT_SCHEMA_VERSION + 5}
        (store.directory / "one.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="newer than this"):
            store.read("one")

    def test_migration_chain_runs_forward(self,
                                          store,
                                          monkeypatch):
        """An older document is migrated before it reaches the caller."""
        monkeypatch.setattr(store_module, "CURRENT_SCHEMA_VERSION", 2)
        monkeypatch.setattr(store_module, "MIGRATIONS",
                            {1: lambda doc: {**doc, "added_in_v2": True}})

        payload = {"id": "one", SCHEMA_VERSION_KEY: 1}
        (store.directory / "one.json").write_text(json.dumps(payload), encoding="utf-8")

        document = store.read("one")

        assert document["added_in_v2"] is True
        assert document[SCHEMA_VERSION_KEY] == 2

    def test_missing_migration_is_refused(self,
                                          store,
                                          monkeypatch):
        monkeypatch.setattr(store_module, "CURRENT_SCHEMA_VERSION", 2)
        monkeypatch.setattr(store_module, "MIGRATIONS", {})

        payload = {"id": "one", SCHEMA_VERSION_KEY: 1}
        (store.directory / "one.json").write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(ConfigurationError, match="no migration registered"):
            store.read("one")


class TestWatchlists:

    def test_starts_empty(self,
                          client):
        assert client.get("/data/watchlists", headers=auth()).json() == {"watchlists": []}

    def test_put_then_get(self,
                          client):
        client.put("/data/watchlists/tech",
                   json={"name": "Tech", "identifiers": ["AAA", "BBB"]},
                   headers=auth())

        body = client.get("/data/watchlists/tech", headers=auth()).json()

        assert body == {"id": "tech", "name": "Tech", "identifiers": ["AAA", "BBB"]}

    def test_identifier_order_is_preserved(self,
                                           client):
        client.put("/data/watchlists/tech",
                   json={"name": "Tech", "identifiers": ["CCC", "AAA", "BBB"]},
                   headers=auth())

        body = client.get("/data/watchlists/tech", headers=auth()).json()

        assert body["identifiers"] == ["CCC", "AAA", "BBB"]

    def test_put_is_an_upsert(self,
                              client):
        client.put("/data/watchlists/tech", json={"name": "Tech", "identifiers": []},
                   headers=auth())
        client.put("/data/watchlists/tech", json={"name": "Renamed", "identifiers": ["AAA"]},
                   headers=auth())

        body = client.get("/data/watchlists/tech", headers=auth()).json()

        assert body["name"] == "Renamed"
        assert body["identifiers"] == ["AAA"]

    def test_listing_includes_every_watchlist(self,
                                              client):
        for name in ("one", "two"):
            client.put(f"/data/watchlists/{name}", json={"name": name, "identifiers": []},
                       headers=auth())

        body = client.get("/data/watchlists", headers=auth()).json()

        assert {w["id"] for w in body["watchlists"]} == {"one", "two"}

    def test_unknown_watchlist_is_404_envelope(self,
                                               client):
        response = client.get("/data/watchlists/absent", headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_delete_then_gone(self,
                              client):
        client.put("/data/watchlists/tech", json={"name": "Tech", "identifiers": []},
                   headers=auth())

        assert client.delete("/data/watchlists/tech", headers=auth()).status_code == 204
        assert client.get("/data/watchlists/tech", headers=auth()).status_code == 404

    def test_delete_unknown_is_404(self,
                                   client):
        assert client.delete("/data/watchlists/absent", headers=auth()).status_code == 404

    def test_empty_name_is_rejected(self,
                                    client):
        response = client.put("/data/watchlists/tech",
                              json={"name": "", "identifiers": []},
                              headers=auth())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_requires_authentication(self,
                                     client):
        assert client.get("/data/watchlists").status_code == 401

    def test_survives_a_process_restart(self,
                                        tmp_path):
        """The acceptance criterion: a watchlist outlives the process.

        Two independent apps over the same storage root stand in for a
        restart — nothing is shared between them but the directory on disk.
        """
        first = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                   storage_root=tmp_path)))
        first.put("/data/watchlists/tech",
                  json={"name": "Tech", "identifiers": ["AAA", "BBB"]},
                  headers=auth())

        second = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                    storage_root=tmp_path)))
        body = second.get("/data/watchlists/tech", headers=auth()).json()

        assert body["identifiers"] == ["AAA", "BBB"]


class TestCoverage:

    def test_reports_every_dataset(self,
                                   client):
        """Three since BN-119: corporate actions are reported even when the
        fixture holds none, because "we hold no actions" is a fact the pane
        should state rather than a row it should omit."""
        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]

        assert {d["dataset"] for d in datasets} == {"market", "reference",
                                                    "corporate_actions"}

    def test_market_coverage_reflects_the_loaded_data(self,
                                                      client):
        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]
        market = next(d for d in datasets if d["dataset"] == "market")

        assert market["configured"] is True
        assert market["identifiers"] == len(ASSETS)
        assert market["start"] == DATES[0].isoformat()
        assert market["end"] == DATES[-1].isoformat()

    def test_reference_coverage_counts_its_own_identifiers(self,
                                                           client):
        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]
        reference = next(d for d in datasets if d["dataset"] == "reference")

        assert reference["configured"] is True
        assert reference["identifiers"] == 2

    def test_cache_age_is_real(self,
                               client):
        """BN-66's criterion, amended by BN-99 rather than dropped.

        It used to assert null, correctly: nothing tracked when data was
        loaded. It is tracked now, so a loaded dataset reports a real age.
        """
        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]
        loaded = [d for d in datasets if d["configured"]]

        assert loaded, "nothing was loaded, so this asserts nothing"

        for dataset in loaded:
            assert dataset["cache_age"] is not None
            assert dataset["cache_age"] >= 0.0
            assert dataset["last_refreshed"] is not None

        # An absent dataset reports null, which is a different statement from
        # "loaded and never refreshed" and must not be collapsed into it.
        for dataset in datasets:
            if not dataset["configured"]:
                assert dataset["cache_age"] is None

    def test_cache_age_is_null_when_a_dataset_is_absent(self,
                                                        tmp_path):
        """Not loaded is a different statement from loaded and never refreshed."""
        market = MarketData.from_dataframe(pd.DataFrame({
            "IDENTIFIER": ["AAA"], "DATE": ["2024-01-01"], "CLOSE": [1.0]}))
        config = ServerConfig(auth_token=TOKEN,
                              data_fetcher=DataFetcher(market),
                              storage_root=tmp_path)
        client = TestClient(create_app(config))

        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]
        reference = next(d for d in datasets if d["dataset"] == "reference")

        assert reference["cache_age"] is None
        assert reference["last_refreshed"] is None

    def test_without_a_data_source(self,
                                   tmp_path):
        client = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                    storage_root=tmp_path)))
        datasets = client.get("/data/coverage", headers=auth()).json()["datasets"]

        assert all(d["configured"] is False for d in datasets)
        assert all(d["identifiers"] == 0 for d in datasets)

    def test_sync_without_a_data_source_reports_500(self,
                                                    tmp_path):
        """Used to be a 501: no ingestion path existed at all.

        BN-100 built one, so the honest failure here is now about this server
        having nothing to sync *into*, not about the feature being absent. The
        sync itself is covered in tests/test_ingest.py, which injects a
        downloader so the path runs without a network.
        """
        bare = TestClient(create_app(ServerConfig(auth_token=TOKEN,
                                                  storage_root=tmp_path)),
                          raise_server_exceptions=False)

        response = bare.post("/data/coverage/market/sync", headers=auth())

        assert response.status_code == 500

    def test_sync_unknown_dataset_is_404(self,
                                         client):
        response = client.post("/data/coverage/nonsense/sync", headers=auth())

        assert response.status_code == 404

    def test_requires_authentication(self,
                                     client):
        assert client.get("/data/coverage").status_code == 401
