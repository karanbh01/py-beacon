# tests/test_server_data_router.py
"""Contract tests for the /data router (prices and reference)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.server import ServerConfig, create_app

TOKEN = "test-token-value"
ASSETS = ["AAA", "BBB"]
DATES = pd.bdate_range("2025-01-02", periods=20)


def build_fetcher() -> DataFetcher:
    """Synthetic market and reference data for two assets over 20 business days."""
    market = pd.DataFrame([
        {"IDENTIFIER": asset,
         "DATE": date,
         "CLOSE": 100.0 + index,
         "VOLUME": 1_000 + index}
        for asset in ASSETS
        for index, date in enumerate(DATES)
    ])
    reference = pd.DataFrame([
        {"IDENTIFIER": "AAA",
         "DATE_FROM": "2020-01-01",
         "NAME": "Alpha Corp",
         "CURRENCY": "USD",
         "EXCHANGE": "NYSE"},
        {"IDENTIFIER": "BBB",
         "DATE_FROM": "2020-01-01",
         "NAME": "Beta Ltd",
         "CURRENCY": "GBP",
         "EXCHANGE": "LSE"},
    ])

    return DataFetcher(MarketData.from_dataframe(market),
                       ReferenceData.from_dataframe(reference))


def auth() -> dict[str, str]:
    """Valid Authorization header."""
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client() -> TestClient:
    """Client for a server with a data source."""
    config = ServerConfig(auth_token=TOKEN, data_fetcher=build_fetcher())
    return TestClient(create_app(config), raise_server_exceptions=False)


@pytest.fixture
def client_without_data() -> TestClient:
    """Client for a server started without a data source."""
    return TestClient(create_app(ServerConfig(auth_token=TOKEN)),
                      raise_server_exceptions=False)


class TestPrices:

    def test_returns_the_full_series(self,
                                     client):
        body = client.get("/data/prices/AAA", headers=auth()).json()

        assert body["identifier"] == "AAA"
        assert body["interval"] == "native"
        assert len(body["prices"]["data"]) == len(DATES)
        assert set(body["prices"]["columns"]) == {"CLOSE", "VOLUME"}

    def test_index_is_iso_dates(self,
                                client):
        body = client.get("/data/prices/AAA", headers=auth()).json()

        assert body["prices"]["index"][0] == DATES[0].isoformat()

    def test_date_range_filters(self,
                                client):
        start = DATES[5].strftime("%Y-%m-%d")
        end = DATES[9].strftime("%Y-%m-%d")

        body = client.get(f"/data/prices/AAA?start={start}&end={end}",
                          headers=auth()).json()

        assert len(body["prices"]["data"]) == 5

    def test_column_subset(self,
                           client):
        body = client.get("/data/prices/AAA?columns=CLOSE", headers=auth()).json()

        assert body["prices"]["columns"] == ["CLOSE"]

    @pytest.mark.parametrize("interval", ["weekly", "monthly"])
    def test_resampling_reduces_the_row_count(self,
                                              client,
                                              interval):
        body = client.get(f"/data/prices/AAA?interval={interval}",
                          headers=auth()).json()

        assert body["interval"] == interval
        assert 0 < len(body["prices"]["data"]) < len(DATES)

    def test_resampling_takes_the_period_close(self,
                                               client):
        """Weekly resampling must carry the last observation of each week."""
        native = client.get("/data/prices/AAA", headers=auth()).json()
        weekly = client.get("/data/prices/AAA?interval=weekly", headers=auth()).json()

        native_by_date = dict(zip(native["prices"]["index"],
                                  native["prices"]["data"], strict=True))
        close_index = native["prices"]["columns"].index("CLOSE")

        # Every weekly close must equal some native close, and the final one
        # must be the last native observation.
        weekly_closes = [row[weekly["prices"]["columns"].index("CLOSE")]
                         for row in weekly["prices"]["data"]]
        native_closes = [row[close_index] for row in native_by_date.values()]

        assert set(weekly_closes) <= set(native_closes)
        assert weekly_closes[-1] == native_closes[-1]

    def test_unknown_identifier_is_404_envelope(self,
                                                client):
        response = client.get("/data/prices/NOPE", headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_unsupported_interval_is_404_envelope(self,
                                                  client):
        response = client.get("/data/prices/AAA?interval=hourly", headers=auth())

        assert response.status_code == 404
        assert "hourly" in response.json()["error"]["message"]

    def test_requires_authentication(self,
                                     client):
        assert client.get("/data/prices/AAA").status_code == 401

    def test_no_data_source_is_a_server_configuration_error(self,
                                                            client_without_data):
        """The request was fine; the process is not configured to answer it."""
        response = client_without_data.get("/data/prices/AAA", headers=auth())

        assert response.status_code == 500
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"

    def test_adjusted_parameter_is_not_accepted(self,
                                                client):
        """No price-adjustment logic exists, so the parameter is not offered.

        FastAPI ignores unknown query parameters, so this asserts the schema
        does not advertise one rather than that the request fails.
        """
        schema = client.app.openapi()
        names = {p["name"]
                 for p in schema["paths"]["/data/prices/{identifier}"]["get"]["parameters"]}

        assert "adjusted" not in names


class TestReference:

    def test_returns_the_fields(self,
                                client):
        body = client.get("/data/reference/AAA", headers=auth()).json()

        assert body["identifier"] == "AAA"
        assert body["fields"]["NAME"] == "Alpha Corp"
        assert body["fields"]["CURRENCY"] == "USD"
        assert body["fields"]["EXCHANGE"] == "NYSE"

    def test_distinguishes_identifiers(self,
                                       client):
        body = client.get("/data/reference/BBB", headers=auth()).json()

        assert body["fields"]["EXCHANGE"] == "LSE"

    def test_unknown_identifier_is_404_envelope(self,
                                                client):
        response = client.get("/data/reference/NOPE", headers=auth())

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "DATA_NOT_FOUND"

    def test_requires_authentication(self,
                                     client):
        assert client.get("/data/reference/AAA").status_code == 401


class TestOpenApiContract:

    def test_both_endpoints_are_documented(self,
                                           client):
        paths = client.app.openapi()["paths"]

        assert "/data/prices/{identifier}" in paths
        assert "/data/reference/{identifier}" in paths

    def test_dropped_endpoints_are_absent(self,
                                          client):
        """Fundamentals is superseded by a future features endpoint;
        corporate actions has no data source. Neither is stubbed out."""
        paths = client.app.openapi()["paths"]

        assert not [p for p in paths if "fundamentals" in p]
        assert not [p for p in paths if "corporate-actions" in p]

    def test_error_envelope_documented_on_data_routes(self,
                                                      client):
        responses = client.app.openapi()[
            "paths"]["/data/prices/{identifier}"]["get"]["responses"]

        for code in ("401", "404", "500"):
            assert code in responses
