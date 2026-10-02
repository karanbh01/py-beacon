# tests/test_server_preview_caps.py
"""BN-278: the preview returns each constituent's market caps.

beacon-ui fetched them from `/data/reference` after every preview, which was
most of the Constituent Preview's wait. They are computed by the same code,
so the two must agree field for field. The universe and index are
`test_server_preview`'s: six names of 1,000 shares each, priced 500 down to
10, with the two smallest excluded by a market-cap rule.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app
from test_server_preview import (
    PRICES,
    SHARES,
    TOKEN,
    auth,
    build_fetcher,
    definition_document,
    preview,
)

CAP_FIELDS = ["market_cap", "free_float_market_cap", "market_cap_local",
              "free_float_market_cap_local", "market_cap_currency",
              "local_currency", "priced_from", "price_is_stale"]


def client_on(fetcher: DataFetcher,
              storage) -> TestClient:
    config = ServerConfig(auth_token=TOKEN, data_fetcher=fetcher,
                          storage_root=storage)
    client = TestClient(create_app(config), raise_server_exceptions=False)
    created = client.post("/indices", json=definition_document(), headers=auth())
    assert created.status_code == 200, created.json()

    return client


def with_free_float(fetcher: DataFetcher) -> DataFetcher:
    """The same six names, half of each freely floated."""
    market = fetcher.fetch_market_data(list(PRICES)).reset_index()
    market["FREE_FLOAT"] = 0.5
    reference = fetcher.fetch_reference_data(list(PRICES)).reset_index()
    reference["DATE_FROM"] = "2020-01-01"

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


@pytest.fixture
def client(tmp_path) -> TestClient:
    return client_on(with_free_float(build_fetcher()), tmp_path)


def rows(payload: dict) -> dict[str, dict]:
    return {row["identifier"]: row for row in payload["assets"]}


class TestEveryNameHasItsCaps:

    def test_included_and_excluded_names_alike(self,
                                               client):
        """A cap is often why a name was excluded, so excluded names carry
        theirs too."""
        assets = rows(preview(client))

        assert not assets["FFF"]["included"]
        assert {name: row["market_cap"] for name, row in assets.items()} == (
            pytest.approx({name: price * SHARES
                           for name, price in PRICES.items()}))

    def test_the_free_float_cap_scales_the_cap(self,
                                               client):
        row = rows(preview(client))["AAA"]

        assert row["free_float_market_cap"] == pytest.approx(0.5 * 500 * SHARES)
        assert row["free_float_market_cap_local"] == pytest.approx(
            0.5 * 500 * SHARES)

    def test_they_are_dated_and_in_the_index_currency(self,
                                                      client):
        payload = preview(client)
        row = rows(payload)["AAA"]

        assert row["market_cap_currency"] == "USD"
        assert row["local_currency"] == "USD"
        assert row["priced_from"] == payload["resolved_date"]
        assert row["price_is_stale"] is False


class TestTheyAgreeWithTheReferenceEndpoint:

    def test_field_for_field(self,
                             client):
        payload = preview(client)
        response = client.get("/data/reference", headers=auth(), params={
            "identifiers": ",".join(PRICES),
            "fields": "market_cap,free_float_market_cap",
            "date": payload["resolved_date"],
            "currency": "USD"})
        assert response.status_code == 200, response.text

        reference = {entry["identifier"]: entry["fields"]
                     for entry in response.json()["entries"]}

        for name, row in rows(payload).items():
            assert {field: row[field] for field in CAP_FIELDS} == {
                field: reference[name][field] for field in CAP_FIELDS}


class TestWhatTheDataCannotSupply:

    def test_without_free_float_only_the_free_float_caps_are_null(self,
                                                                  tmp_path):
        row = rows(preview(client_on(build_fetcher(), tmp_path)))["AAA"]

        assert row["market_cap"] == pytest.approx(500 * SHARES)
        assert row["free_float_market_cap"] is None
        assert row["free_float_market_cap_local"] is None

    def test_a_name_with_no_price_has_null_caps(self,
                                                tmp_path):
        """GHOST is in the universe and the reference data but never
        traded."""
        fetcher = build_fetcher()
        reference = fetcher.fetch_reference_data(list(PRICES)).reset_index()
        reference["DATE_FROM"] = "2020-01-01"
        reference = pd.concat([reference, pd.DataFrame([{
            "IDENTIFIER": "GHOST", "DATE_FROM": "2020-01-01", "NAME": "Ghost",
            "CURRENCY": "USD", "EXCHANGE": "NYSE"}])])
        market = fetcher.fetch_market_data(list(PRICES)).reset_index()
        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, storage_root=tmp_path,
            data_fetcher=DataFetcher(MarketData.from_dataframe(market),
                                     ReferenceData.from_dataframe(reference)))),
            raise_server_exceptions=False)
        document = definition_document()
        document["universe"]["identifiers"] = [*PRICES, "GHOST"]
        assert client.post("/indices", json=document,
                           headers=auth()).status_code == 200

        ghost = rows(preview(client))["GHOST"]

        assert all(ghost[field] is None for field in CAP_FIELDS
                   if field != "market_cap_currency")
