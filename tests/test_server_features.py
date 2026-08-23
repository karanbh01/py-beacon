# tests/test_server_features.py
"""BN-137: the features API.

Read, batch, catalogue and import. The catalogue is what beacon-ui populates
its controls from — derived from the loaded data rather than a fixed
vocabulary, so a dataset somebody loads tomorrow becomes a filter with no
client change.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "feature-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


def rows(*records) -> pd.DataFrame:
    base = {"IDENTIFIER": "AAA", "DATE": "2024-05-15",
            "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 1000.0,
            "DETAIL": None}

    return pd.DataFrame([{**base, **record} for record in records])


@pytest.fixture
def client():
    """A server holding a publication and a restatement."""
    source = dataset.data_fetcher()
    table = FeatureData.from_dataframe(rows(
        {"DATE": "2024-05-15", "VALUE": 1000.0, "DETAIL": "FY24Q1"},
        {"DATE": "2024-08-15", "VALUE": 1050.0, "DETAIL": "FY24Q1 restated"}))

    fetcher = DataFetcher(source.market, source.reference,
                          source.corporate_actions, table)

    return TestClient(create_app(ServerConfig(
        auth_token=TOKEN, data_fetcher=fetcher,
        storage_root=Path(tempfile.mkdtemp()))), raise_server_exceptions=False)


def features_of(response) -> list[dict]:
    return response.json()["features"]


class TestReadingOne:
    """`GET /data/features/{identifier}`."""

    def test_it_resolves_point_in_time(self,
                                       client):
        """The endpoint is a thin surface over the accessor, so the guarantee
        that matters is that it stands where it is told to."""
        june = client.get("/data/features/AAA", headers=HEADERS,
                          params={"date": "2024-06-01"})
        september = client.get("/data/features/AAA", headers=HEADERS,
                               params={"date": "2024-09-01"})

        assert features_of(june)[0]["value"] == 1000.0
        assert features_of(september)[0]["value"] == 1050.0

    def test_it_reports_when_the_value_is_from(self,
                                               client):
        """For a fundamental, *when* a number was published is most of what
        makes it interpretable — a client showing revenue without its date is
        showing a number nobody can date."""
        june = client.get("/data/features/AAA", headers=HEADERS,
                          params={"date": "2024-06-01"})
        entry = features_of(june)[0]

        assert entry["date"] == "2024-05-15"
        assert entry["detail"] == "FY24Q1"

    def test_nothing_published_yet_is_null_not_absent(self,
                                                      client):
        april = client.get("/data/features/AAA", headers=HEADERS,
                           params={"date": "2024-04-01"})
        entry = features_of(april)[0]

        assert entry["field"] == "revenue"
        assert entry["value"] is None

    def test_an_uncovered_instrument_answers_rather_than_404s(self,
                                                              client):
        """No coverage is an ordinary answer. A 404 would make "we hold
        nothing for this name" indistinguishable from "no such name"."""
        response = client.get("/data/features/BBB", headers=HEADERS,
                              params={"date": "2024-06-01"})

        assert response.status_code == 200
        assert features_of(response)[0]["value"] is None


class TestTheBatchForm:
    """`GET /data/features`."""

    def test_it_answers_every_identifier(self,
                                         client):
        response = client.get("/data/features", headers=HEADERS,
                              params={"identifiers": "AAA,BBB",
                                      "date": "2024-06-01"})
        entries = response.json()["entries"]

        assert [entry["identifier"] for entry in entries] == ["AAA", "BBB"]
        assert entries[1]["features"][0]["value"] is None

    def test_it_splits_a_comma_separated_list(self,
                                              client):
        """Both forms, on the terms the reference batch endpoint set — a
        client should not have to know which this server prefers."""
        response = client.get("/data/features", headers=HEADERS,
                              params={"identifiers": "AAA", "fields": "revenue"})

        assert response.status_code == 200

    def test_naming_nothing_is_refused(self,
                                       client):
        assert client.get("/data/features",
                          headers=HEADERS).status_code == 404

    def test_the_batch_is_bounded(self,
                                  client):
        """A malformed client cannot ask the server to assemble an unbounded
        response."""
        response = client.get("/data/features", headers=HEADERS,
                              params={"identifiers": ",".join(
                                  f"N{n}" for n in range(2000))})

        assert response.status_code == 404


class TestTheCatalogue:
    """`GET /data/features/catalogue`, which the client builds controls from."""

    def test_it_lists_types_with_coverage(self,
                                          client):
        body = client.get("/data/features/catalogue", headers=HEADERS).json()

        assert body["types"][0]["type"] == "fundamentals"
        assert body["types"][0]["fields"] == ["revenue"]
        assert body["types"][0]["rows"] == 2

    def test_the_literal_route_wins_over_the_identifier_one(self,
                                                            client):
        """FastAPI matches in declaration order and "catalogue" is a perfectly
        good identifier as far as the path is concerned. Declared after the
        by-identifier route, this would answer as though somebody had asked
        for an instrument called "catalogue"."""
        body = client.get("/data/features/catalogue", headers=HEADERS).json()

        assert "types" in body
        assert "identifier" not in body


class TestImport:
    """`POST /data/features` — the derived-data case."""

    def test_it_accepts_and_is_immediately_readable(self,
                                                    client):
        created = client.post("/data/features", headers=HEADERS, json={
            "rows": [{"identifier": "BBB", "date": "2024-06-01",
                      "type": "derived", "field": "score", "value": 7.0}]})

        assert created.status_code == 201
        assert created.json()["accepted"] == 1

        read = client.get("/data/features/BBB", headers=HEADERS,
                          params={"date": "2024-07-01", "fields": "score"})

        assert features_of(read)[0]["value"] == 7.0

    def test_it_merges_rather_than_replaces(self,
                                            client):
        """A feature table is append-only by nature, and a second upload of a
        different dataset should not discard the first."""
        client.post("/data/features", headers=HEADERS, json={
            "rows": [{"identifier": "AAA", "date": "2024-06-01",
                      "type": "derived", "field": "score", "value": 7.0}]})

        catalogue = client.get("/data/features/catalogue",
                               headers=HEADERS).json()

        assert {entry["type"] for entry in catalogue["types"]} == {
            "fundamentals", "derived"}

    def test_it_is_invisible_before_its_date(self,
                                             client):
        """An imported value obeys the same rule as a loaded one. A user
        importing a derived signal dated next month must not be able to screen
        on it today."""
        client.post("/data/features", headers=HEADERS, json={
            "rows": [{"identifier": "AAA", "date": "2024-09-01",
                      "type": "derived", "field": "score", "value": 7.0}]})

        early = client.get("/data/features/AAA", headers=HEADERS,
                           params={"date": "2024-08-01", "fields": "score"})

        assert features_of(early)[0]["value"] is None

    def test_an_unknown_identifier_is_named(self,
                                            client):
        """Findings rather than a bare 422: telling somebody a thousand-row
        upload is wrong without saying which row is not an error message."""
        response = client.post("/data/features", headers=HEADERS, json={
            "rows": [{"identifier": "NOSUCH", "date": "2024-06-01",
                      "type": "derived", "field": "score", "value": 1.0}]})

        assert response.status_code == 422

        findings = response.json()["error"]["detail"]["findings"]

        assert any("NOSUCH" in finding["message"] for finding in findings)

    def test_an_empty_import_is_refused(self,
                                        client):
        response = client.post("/data/features", headers=HEADERS,
                               json={"rows": []})

        assert response.status_code == 422

    def test_a_malformed_date_is_refused(self,
                                         client):
        """The IsoDate type from BN-131, so this is refused at the edge with a
        field path rather than deep in the container."""
        response = client.post("/data/features", headers=HEADERS, json={
            "rows": [{"identifier": "AAA", "date": "not-a-date",
                      "type": "derived", "field": "score", "value": 1.0}]})

        assert response.status_code == 422


class TestTheSurface:
    """What the client generates from."""

    def test_the_routes_are_in_the_spec(self,
                                        client):
        paths = client.get("/openapi.json").json()["paths"]

        assert "/data/features/{identifier}" in paths
        assert "/data/features" in paths
        assert "post" in paths["/data/features"]
        assert "/data/features/catalogue" in paths
