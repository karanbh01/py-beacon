# tests/test_data_tables.py
"""BN-147/150: `GET /data/tables/{dataset}` — browsing, and narrowing.

From beacon-ui BU-102 and the feature-history block. Paged because the default
store is 11.8M market rows, so most of what is worth testing is that paging is
honest — no repeats, no gaps, a stable order — and that the filter composes
with it rather than fighting it.
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


@pytest.fixture(scope="module")
def covered(panel):
    """An instrument the feature table actually carries rows for.

    Derived rather than hardcoded: feature coverage is deliberately incomplete,
    so naming a specific instrument makes the test depend on the draw. The
    first version hardcoded CMPA and failed because this fixture's one-year
    panel does not cover it.
    """
    rows = panel.features.data.reset_index()

    return str(rows["IDENTIFIER"].iloc[0])


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


class TestNarrowingToInstruments:
    """BN-150: the filter that unblocks one instrument's history."""

    def rows(self,
             client,
             dataset: str,
             **params) -> dict:
        return client.get(f"/data/tables/{dataset}", headers=HEADERS,
                          params=params).json()

    def test_it_returns_only_the_named_instrument(self,
                                                  client,
                                                  covered):
        body = self.rows(client, "features", identifiers=covered, limit=1000)
        names = {row[0] for row in body["rows"]["data"]}

        assert names == {covered}

    def test_the_total_counts_the_filtered_set(self,
                                               client,
                                               covered):
        """Not the whole dataset. A total that ignored the filter would size a
        client's scrollbar for rows it will never be shown."""
        everything = self.rows(client, "features", limit=1)["total"]
        filtered = self.rows(client, "features", identifiers=covered,
                             limit=1)["total"]

        assert 0 < filtered < everything

    def test_paging_walks_the_filtered_set(self,
                                           client,
                                           covered):
        """Filtered before paging. The other order makes the second page of a
        filtered request a slice of a different query."""
        total = self.rows(client, "features", identifiers=covered,
                          limit=1)["total"]
        first = self.rows(client, "features", identifiers=covered, offset=0,
                          limit=10)
        second = self.rows(client, "features", identifiers=covered, offset=10,
                           limit=10)

        seen = [tuple(row) for row in first["rows"]["data"]]
        seen += [tuple(row) for row in second["rows"]["data"]]

        assert len(seen) == len(set(seen)) == min(20, total)
        assert {row[0] for row in seen} == {covered}

    def test_both_spellings_are_the_same_request(self,
                                                 client):
        """A client should not have to know which form this server prefers."""
        comma = self.rows(client, "market", identifiers="CMPA,CMPB",
                          limit=1)["total"]
        repeated = client.get("/data/tables/market", headers=HEADERS,
                              params=[("identifiers", "CMPA"),
                                      ("identifiers", "CMPB"),
                                      ("limit", 1)]).json()["total"]

        assert comma == repeated
        assert comma > 0

    def test_two_names_return_more_than_one(self,
                                            client,
                                            panel):
        names = list(dict.fromkeys(
            panel.features.data.reset_index()["IDENTIFIER"]))[:2]
        one = self.rows(client, "features", identifiers=names[0],
                        limit=1)["total"]
        two = self.rows(client, "features", identifiers=",".join(names),
                        limit=1)["total"]

        assert two > one

    @pytest.mark.parametrize("dataset", ["market", "reference",
                                         "corporate_actions", "features"])
    def test_it_works_on_every_dataset(self,
                                       client,
                                       dataset,
                                       covered):
        """The containers differ in whether they consumed the key into the
        index, so the filter has to look in both places."""
        body = self.rows(client, dataset, identifiers=covered, limit=1000)

        assert body["total"] > 0
        assert {row[0] for row in body["rows"]["data"]} == {covered}

    def test_an_unknown_name_yields_no_rows_rather_than_an_error(self,
                                                                 client):
        """A filter matching nothing is an ordinary answer, and one unknown
        name should not fail a page that would have answered for the rest."""
        body = self.rows(client, "features", identifiers="NOSUCH", limit=5)

        assert body["total"] == 0
        assert body["rows"]["data"] == []

    def test_a_known_name_survives_an_unknown_companion(self,
                                                        client,
                                                        covered):
        alone = self.rows(client, "features", identifiers=covered,
                          limit=1)["total"]
        together = self.rows(client, "features",
                             identifiers=f"{covered},NOSUCH",
                             limit=1)["total"]

        assert together == alone
        assert alone > 0

    def test_omitting_it_is_unchanged(self,
                                      client):
        """The behaviour every existing caller depends on."""
        filtered = self.rows(client, "features", limit=1)["total"]
        plain = client.get("/data/tables/features", headers=HEADERS,
                           params={"limit": 1}).json()["total"]

        assert filtered == plain

    def test_too_many_names_is_refused(self,
                                       client):
        """Bounded like the reference endpoints: a malformed client cannot ask
        the server to assemble an unbounded filter."""
        response = client.get("/data/tables/features", headers=HEADERS,
                              params={"identifiers": ",".join(
                                  f"N{n}" for n in range(2000))})

        assert response.status_code == 422

    def test_the_parameter_is_in_the_spec(self,
                                          client):
        spec = client.get("/openapi.json").json()
        params = spec["paths"]["/data/tables/{dataset}"]["get"]["parameters"]

        assert "identifiers" in {entry["name"] for entry in params}


class TestFeatureHistory:
    """The case this unblocks, end to end."""

    def test_a_whole_feature_history_is_reachable(self,
                                                  client,
                                                  panel,
                                                  covered):
        """Previously this meant paging the entire feature table and filtering
        client-side -- 964k rows at the default store size to find a few
        dozen."""
        body = client.get("/data/tables/features", headers=HEADERS,
                          params={"identifiers": covered,
                                  "limit": 1000}).json()

        stored = panel.features.data.reset_index()
        expected = len(stored[stored["IDENTIFIER"] == covered])

        assert body["total"] == expected
        assert len(body["rows"]["data"]) == expected

    def test_the_history_carries_its_dates_and_detail(self,
                                                      client,
                                                      covered):
        """What makes it a history rather than a snapshot: when each value
        became knowable, and which period it describes."""
        body = client.get("/data/tables/features", headers=HEADERS,
                          params={"identifiers": covered, "limit": 5}).json()

        assert set(body["rows"]["columns"]) >= {"IDENTIFIER", "DATE", "TYPE",
                                                "FIELD", "VALUE", "DETAIL"}

    def test_it_spans_more_than_one_date(self,
                                         client,
                                         covered):
        body = client.get("/data/tables/features", headers=HEADERS,
                          params={"identifiers": covered, "limit": 1000}).json()
        dates = {row[1] for row in body["rows"]["data"]}

        assert len(dates) > 1
