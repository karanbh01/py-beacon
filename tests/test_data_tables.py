# tests/test_data_tables.py
"""BN-147: `GET /data/tables/{dataset}` — whole-dataset browsing.

From beacon-ui BU-102. Paged because the default store is 11.8M market rows,
and most of what is worth testing is that paging is honest: no repeats, no
gaps, and a stable order across requests.
"""
import logging
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, generate

TOKEN = "tables-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def panel():
    logging.disable(logging.ERROR)

    try:
        return generate(SyntheticConfig(assets=12, start="2024-01-02",
                                        end="2024-12-31", seed=1))
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def client(panel):
    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=panel.fetcher(),
        storage_root=Path(tempfile.mkdtemp()))))


class TestTableBrowsing:
    """BU-102: `GET /data/tables/{dataset}`."""

    @pytest.mark.parametrize("dataset", ["market", "reference",
                                         "corporate_actions", "features"])
    def test_every_dataset_pages(self,
                                 client,
                                 dataset):
        body = client.get(f"/data/tables/{dataset}", headers=HEADERS,
                          params={"limit": 3}).json()

        assert body["dataset"] == dataset
        assert len(body["rows"]["data"]) == 3
        assert body["total"] > 3

    def test_the_keys_arrive_as_columns(self,
                                        client):
        """A database view wants the row as stored. A MultiIndex would
        serialise as tuples the client has to unpack."""
        body = client.get("/data/tables/market", headers=HEADERS,
                          params={"limit": 1}).json()

        assert body["rows"]["columns"][:2] == ["IDENTIFIER", "DATE"]

    def test_corporate_actions_do_not_duplicate_their_key(self,
                                                          client):
        """`CorporateActions` sets its index with drop=False, so a plain
        `reset_index()` raises "cannot insert EX_DATE, already exists". It
        did."""
        body = client.get("/data/tables/corporate_actions", headers=HEADERS,
                          params={"limit": 2}).json()
        columns = body["rows"]["columns"]

        assert columns.count("EX_DATE") == 1

    def test_paging_does_not_repeat_or_skip(self,
                                            client):
        """The classic failure of paging an unordered frame is showing a row
        twice and never showing another."""
        first = client.get("/data/tables/market", headers=HEADERS,
                           params={"offset": 0, "limit": 50}).json()
        second = client.get("/data/tables/market", headers=HEADERS,
                            params={"offset": 50, "limit": 50}).json()

        rows = [tuple(row) for row in first["rows"]["data"]]
        rows += [tuple(row) for row in second["rows"]["data"]]

        assert len(rows) == len(set(rows)) == 100

    def test_the_order_is_stable_across_requests(self,
                                                 client):
        params = {"offset": 10, "limit": 20}
        first = client.get("/data/tables/market", headers=HEADERS,
                           params=params).json()
        again = client.get("/data/tables/market", headers=HEADERS,
                           params=params).json()

        assert first["rows"]["data"] == again["rows"]["data"]

    def test_past_the_end_is_an_empty_page(self,
                                           client):
        """Not a 404: a client paging to the end should not have to treat its
        last request as a failure."""
        body = client.get("/data/tables/market", headers=HEADERS,
                          params={"offset": 10_000_000, "limit": 5}).json()

        assert body["rows"]["data"] == []
        assert body["total"] > 0

    def test_the_total_is_the_whole_dataset(self,
                                            client,
                                            panel):
        body = client.get("/data/tables/market", headers=HEADERS,
                          params={"limit": 1}).json()

        assert body["total"] == len(panel.market.data)

    def test_an_unbounded_request_is_refused(self,
                                             client):
        """The engine should not assemble 11.8M rows because a client asked
        without thinking."""
        assert client.get("/data/tables/market", headers=HEADERS,
                          params={"limit": 99_999}).status_code == 422

    def test_an_unknown_dataset_names_the_real_ones(self,
                                                    client):
        response = client.get("/data/tables/nonsense", headers=HEADERS)

        assert response.status_code == 404
        assert "market" in response.json()["error"]["message"]

    def test_it_requires_authentication(self,
                                        client):
        assert client.get("/data/tables/market").status_code == 401

    def test_it_is_in_the_spec(self,
                               client):
        paths = client.get("/openapi.json").json()["paths"]

        assert "/data/tables/{dataset}" in paths
