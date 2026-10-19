# tests/test_server_parameter_schemas.py
"""BN-330: a structured backtest parameter names the shape its editor renders.

Each JSON parameter in `GET /backtest/options` carries a `ref` that resolves in
the OpenAPI document, a list of reference values names its `column`, and a
parameter whose entries are configurable types names the family they come
from. The shapes are checked against what the settings builder accepts, so a
value written to the published shape builds.
"""
import pytest
from fastapi.testclient import TestClient

from beacon import catalogue
from beacon.server import ServerConfig, create_app
from beacon.server.backtest_settings import Findings, typed
from beacon.server.schemas import TypedSpec
from beacon.strategy.signals import FieldSignal
from beacon.testing import dataset

TOKEN = "parameter-schemas-token"

# Every JSON parameter of a backtest family, and the schema it names.
REFS = {
    ("screens", "ExclusionScreen", "identifiers"): "IdentifierList",
    ("screens", "ExclusionScreen", "sectors"): "ReferenceValueList",
    ("screens", "ExclusionScreen", "regions"): "ReferenceValueList",
    ("screens", "ExpressionScreen", "expression"): "ExpressionNode",
    ("flows", "DatedFlows", "amounts"): "DatedAmounts",
    ("replications", "OptimisedReplication", "constraints"): "ConstraintRows",
    ("replications", "SyntheticReplication", "basket"): "IdentifierValues",
    ("signals", "FieldSignal", "field"): "FieldNode",
    ("constructions", "FactorTilts", "factors"): "FactorSignals",
    ("constructions", "FactorTilts", "targets"): "FactorTargets",
    ("constructions", "RiskParity", "budgets"): "IdentifierValues",
    ("replication_constraints", "ExpectedReturnTarget", "expected_returns"):
        "IdentifierValues",
    ("replication_constraints", "GroupBounds", "members"): "IdentifierList",
    ("replication_constraints", "PositionBounds", "assets"): "IdentifierList",
    ("replication_constraints", "TurnoverBudget", "current_weights"): "IdentifierValues",
}


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    config = ServerConfig(auth_token=TOKEN, data_fetcher=dataset.data_fetcher(),
                          storage_root=tmp_path_factory.mktemp("parameter-schemas"))

    with TestClient(create_app(config), raise_server_exceptions=False) as entered:
        yield entered


@pytest.fixture(scope="module")
def options(client) -> dict:
    response = client.get("/backtest/options", headers=auth())
    assert response.status_code == 200, response.text

    return response.json()


@pytest.fixture(scope="module")
def schemas(client) -> dict:
    return client.get("/openapi.json").json()["components"]["schemas"]


def parameter(options: dict,
              family: str,
              type_name: str,
              name: str) -> dict:
    spec = next(spec for spec in options[family] if spec["name"] == type_name)

    return next(item for item in spec["parameters"] if item["name"] == name)


def families(options: dict) -> dict[str, list[dict]]:
    return {name: value for name, value in options.items()
            if isinstance(value, list) and value and "parameters" in value[0]}


class TestEveryJsonParameterNamesItsShape:

    def test_each_names_the_schema_it_takes(self,
                                            options):
        served = {(family, spec["name"], item["name"]): item["ref"]
                  for family, specs in families(options).items() for spec in specs
                  for item in spec["parameters"] if item["type"] == "json"}

        assert served == REFS

    def test_the_vehicle_settings_do_too(self,
                                         options):
        served = {item["name"]: (item["ref"], item["options"])
                  for item in options["vehicle_settings"] if item["type"] == "json"}

        assert served == {"pricing": ("TypedSpec", "pricing"),
                          "limits": ("TypedSpecs", "limits"),
                          "withholding_rates": ("CountryRates", None),
                          # Described field by field in `market_settings`.
                          "market": (None, None)}

    def test_a_scalar_names_none(self,
                                 options):
        """Guards the tests above: a ref on everything would pass them."""
        assert parameter(options, "replications", "OptimisedReplication",
                         "holdings")["ref"] is None

    def test_every_ref_resolves(self,
                                options,
                                schemas):
        items = [item for specs in families(options).values() for spec in specs
                 for item in spec["parameters"]] + options["vehicle_settings"]

        for item in items:
            if item["ref"] is not None:
                assert item["ref"] in schemas, item

    def test_the_shapes_say_what_they_hold(self,
                                           schemas):
        assert schemas["IdentifierList"]["items"] == {"type": "string"}
        assert schemas["DatedAmounts"]["propertyNames"] == {"format": "date"}
        assert schemas["DatedAmounts"]["additionalProperties"] == {"type": "number"}
        assert schemas["ConstraintRows"]["items"] == {"$ref": "#/components/schemas/ConstraintRow"}
        assert schemas["FactorSignals"]["additionalProperties"] == {
            "$ref": "#/components/schemas/TypedSpec"}

    def test_publishing_them_changes_no_route_schema(self,
                                                     client):
        """`ConstraintRow` is published by the optimiser's routes, and stays as
        they publish it."""
        from beacon.server.schemas import ConstraintRow

        published = client.get("/openapi.json").json()["components"]["schemas"]

        assert published["ConstraintRow"]["properties"].keys() == \
            ConstraintRow.model_json_schema()["properties"].keys()


class TestWhereTheValuesComeFrom:

    def test_reference_lists_name_their_column(self,
                                               options):
        assert parameter(options, "screens", "ExclusionScreen", "sectors")["column"] == "SECTOR"
        assert parameter(options, "screens", "ExclusionScreen", "regions")["column"] == "REGION"
        assert parameter(options, "screens", "ExclusionScreen", "identifiers")["column"] is None

    def test_typed_entries_name_their_family(self,
                                             options):
        assert parameter(options, "constructions", "FactorTilts",
                         "factors")["options"] == "signals"
        assert parameter(options, "replications", "OptimisedReplication",
                         "constraints")["options"] == "replication_constraints"

    def test_every_family_named_is_served(self,
                                          options):
        named = {item["options"] for specs in families(options).values() for spec in specs
                 for item in spec["parameters"] if item["options"] is not None}

        assert named and named <= set(options)

    def test_replication_constraints_are_the_optimisers(self,
                                                        client,
                                                        options):
        served = client.get("/optimise/constraint-types", headers=auth()).json()["specs"]

        assert [spec["name"] for spec in options["replication_constraints"]] == \
            [spec["name"] for spec in served]

    def test_a_dated_amount_is_money(self,
                                     options):
        assert parameter(options, "flows", "DatedFlows", "amounts")["unit"] == "money"


class TestTheShapesBuild:
    """A value written to the published shape is one the builder accepts."""

    def build(self,
              kind: str,
              type_name: str,
              params: dict):
        findings = Findings()
        built = typed(kind, TypedSpec(type=type_name, params=params), "path", findings)

        assert not findings, findings.found
        return built

    def test_constraint_rows_carry_their_ids(self):
        rows = [{"id": "tech", "type": "GroupBounds",
                 "params": {"name": "Tech", "members": ["AAA", "BBB"], "maximum": 0.3}},
                {"id": "caps", "type": "PositionBounds", "params": {"maximum": 0.1}}]

        built = self.build(catalogue.REPLICATION, "OptimisedReplication", {"constraints": rows})

        assert [type(rule).__name__ for rule in built.constraints] == ["GroupBounds",
                                                                     "PositionBounds"]

    def test_factor_signals_take_a_field_node(self):
        factors = {"value": {"type": "FieldSignal",
                             "params": {"field": {"node": "field", "namespace": "market",
                                                  "name": "close"}}}}

        built = self.build(catalogue.CONSTRUCTION, "FactorTilts",
                           {"factors": factors, "targets": {"value": 0.3}})

        assert isinstance(built.factors["value"], FieldSignal)

    def test_dated_amounts_take_dates(self):
        built = self.build(catalogue.FLOW, "DatedFlows",
                           {"amounts": {"2024-03-01": 1000.0, "2024-06-03": -500.0}})

        assert sorted(built.amounts.values()) == [-500.0, 1000.0]


class TestReferenceValues:

    def test_lists_a_columns_values(self,
                                    client):
        response = client.get("/data/reference-values", params={"column": "SECTOR"},
                              headers=auth())

        assert response.status_code == 200, response.text
        assert response.json() == {"column": "SECTOR",
                                   "values": sorted(dataset.data_fetcher().reference
                                                    .data["SECTOR"].dropna().unique())}

    def test_the_column_is_matched_without_case(self,
                                                client):
        response = client.get("/data/reference-values", params={"column": "sector"},
                              headers=auth())

        assert response.json()["column"] == "SECTOR"

    def test_an_unknown_column_is_not_found(self,
                                            client):
        response = client.get("/data/reference-values", params={"column": "REGION"},
                              headers=auth())

        assert response.status_code == 404

    def test_needs_the_token(self,
                             client):
        response = client.get("/data/reference-values", params={"column": "SECTOR"})

        assert response.status_code == 401
