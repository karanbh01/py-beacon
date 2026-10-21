# tests/test_server_request_numbers.py
"""BN-331: a number in a request body must arrive as a JSON number.

Pydantic's lax mode read `true` as 1 and `"5"` as 5, so a check-only route
answered 200 for a body the schema refuses. A whole-valued float such as 30.0
is still an integer, as JSON Schema counts one, and a whole number's bounds stay
in the published schema.
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

    def test_a_fraction_is_refused(self,
                                   client):
        assert refused_as_schema(validate(client, strategy={"minimum_observations": 30.5}))

    def test_its_bound_is_published(self,
                                    client):
        """A validator wrapped around `int` once turned `minimum` into a bare
        `ge` key, and the fuzz run sent 0 as valid."""
        schema = client.get("/openapi.json").json()["components"]["schemas"]["StrategySpec"]
        arms = schema["properties"]["minimum_observations"]["anyOf"]

        assert {"type": "integer", "minimum": 2} in arms


def bare_bounds(node,
                where: str = "") -> list[str]:
    """Every schema in *node* that carries pydantic's bound names, which JSON
    Schema does not read, in place of `minimum` and the rest."""
    if isinstance(node, list):
        return [found for position, item in enumerate(node)
                for found in bare_bounds(item, f"{where}[{position}]")]

    if not isinstance(node, dict):
        return []

    found = [where] if {"ge", "gt", "le", "lt"} & set(node) and "type" in node else []

    for key, value in node.items():
        if key == "properties":
            found += [hit for name, child in value.items()
                      for hit in bare_bounds(child, f"{where}.{name}")]
        else:
            found += bare_bounds(value, f"{where}/{key}")

    return found


class TestThePublishedSchema:

    def test_states_every_bound_as_json_schema_does(self,
                                                    client):
        assert bare_bounds(client.get("/openapi.json").json()) == []
