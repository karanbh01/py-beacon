# tests/test_server_request_numbers.py
"""BN-331: a number in a request body must arrive as a JSON number.

Pydantic's lax mode read `true` as 1 and `"5"` as 5, so a check-only route
answered 200 for a body the schema refuses. A whole-valued float such as 30.0
is still an integer, as JSON Schema counts one.
"""
import pytest
from fastapi.testclient import TestClient

from beacon.server import ServerConfig, create_app
from beacon.testing import dataset

TOKEN = "request-numbers-token"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=dataset.data_fetcher(),
                          storage_root=tmp_path_factory.mktemp("request-numbers"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        yield entered


def validate(client,
             strategy: dict | None = None,
             implementation: dict | None = None):
    body = {"id": "S", "name": "S", "index_id": "X",
            "strategy": {"type": "index", **(strategy or {})},
            "implementation": implementation or {}}

    return client.post("/strategies/validate", json=body, headers=auth())


def refused_as_schema(response) -> bool:
    return (response.status_code == 422
            and response.json()["error"]["code"] == "VALIDATION_ERROR")


class TestANumber:

    def test_a_boolean_is_refused(self,
                                  client):
        assert refused_as_schema(validate(client, implementation={"cash_buffer": True}))

    def test_a_string_is_refused(self,
                                 client):
        assert refused_as_schema(validate(client, implementation={"cash_buffer": "0.1"}))

    def test_an_integer_is_a_number(self,
                                    client):
        assert validate(client, implementation={"cash_buffer": 0}).status_code == 200

    def test_inside_a_settings_object_too(self,
                                          client):
        assert refused_as_schema(validate(client,
                                          implementation={"equitise": {"roll_days": False}}))


class TestAWholeNumber:

    def test_a_whole_valued_float_is_accepted(self,
                                              client):
        response = validate(client, strategy={"minimum_observations": 30.0})

        assert response.status_code == 200, response.text

    @pytest.mark.parametrize("value", [30.5, True, "30"])
    def test_anything_else_is_refused(self,
                                      client,
                                      value):
        assert refused_as_schema(validate(client, strategy={"minimum_observations": value}))
