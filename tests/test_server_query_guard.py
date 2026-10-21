# tests/test_server_query_guard.py
"""BN-331: a query parameter that takes one value is refused when given twice.

Before it, the last value quietly won, which is how the nightly fuzz run's
`?offset=-5&offset=0.0` was answered as if it had asked for offset 0.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "query-guard-token"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=dataset.data_fetcher(),
                          storage_root=tmp_path_factory.mktemp("query-guard"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        yield entered


class TestARepeatedSingleValue:

    def test_is_refused(self,
                        client):
        response = client.get("/data/identifiers?offset=5&offset=0", headers=auth())

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "INVALID_ARGUMENT"
        assert "`offset`" in response.json()["error"]["message"]

    def test_names_every_one_repeated(self,
                                      client):
        response = client.get("/data/identifiers?offset=0&offset=0&limit=1&limit=2",
                              headers=auth())

        assert "`limit`, `offset`" in response.json()["error"]["message"]

    def test_given_once_it_is_read(self,
                                   client):
        response = client.get("/data/identifiers?offset=1&limit=2", headers=auth())

        assert response.status_code == 200, response.text

    def test_authentication_comes_first(self,
                                        client):
        response = client.get("/data/identifiers?offset=5&offset=0")

        assert response.status_code == 401


class TestAListParameter:

    def test_may_repeat(self,
                        client):
        response = client.get("/data/reference?identifiers=AAA&identifiers=BBB",
                              headers=auth())

        assert response.status_code == 200, response.text
        assert len(response.json()["entries"]) == 2

    def test_on_a_path_with_parameters_too(self,
                                           client):
        response = client.get("/data/prices/AAA?columns=CLOSE&columns=VOLUME",
                              headers=auth())

        assert response.status_code == 200, response.text
        assert set(response.json()["prices"]["columns"]) == {"CLOSE", "VOLUME"}

    def test_a_single_value_on_that_path_may_not(self,
                                                 client):
        response = client.get("/data/prices/AAA?interval=weekly&interval=monthly",
                              headers=auth())

        assert response.status_code == 422
