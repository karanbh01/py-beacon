# tests/test_expression_schema.py
"""BN-175: the published expression grammar, and a parameter that names a schema.

The grammar did not change to be published — `expressions/core.py` is untouched
— so nothing here tests evaluation. What it tests is the *description* of a
shape that already crossed the wire, and the two ways such a description goes
wrong:

**It can drift from the library.** `ExpressionNode` is unusual among the wire
models: the others *are* the contract, so `required` cannot disagree with what
the server demands, whereas this one mirrors classes that live elsewhere. A
field the model demands and the constructor defaults rejects documents the
library would accept; a field the model defaults and the constructor requires
accepts documents the server then refuses. The second is not hypothetical — the
spike had it on `value`, and a client found it by generating types and trying to
construct one. `TestTheMirrorDoesNotDrift` checks both directions.

**It can be incomplete.** A sixth node kind, or a ninth comparison, added to the
library and not published leaves a client with a grammar that is quietly wrong
rather than loudly broken. Both sets are derived from the library here, so
adding one without publishing it fails the build.
"""
import inspect
import tempfile
from pathlib import Path
from typing import Any, get_args

import pytest
from fastapi.testclient import TestClient
from pydantic_core import PydanticUndefined

from beacon import universe
from beacon.expressions import core, data
from beacon.expressions.core import COMPARISONS, from_dict
from beacon.server import ServerConfig, create_app
from beacon.server.schemas import (
    AllNode,
    AnyNode,
    ComparisonNode,
    ExpressionNode,
    FieldNode,
    NotNode,
)
from beacon.testing import dataset

TOKEN = "expression-schema-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

# Wire model -> the library class it mirrors. The one hand-written table in this
# file, and the test below proves it covers every node the library has: a new
# `Expression` subclass that is not in here fails, which is the point.
MIRROR: dict[type, type] = {FieldNode: core.Field,
                            ComparisonNode: core.Comparison,
                            AllNode: core.All,
                            AnyNode: core.Any_,
                            NotNode: core.Not}

# `node` is the only field a wire model carries that its library class does not:
# `to_dict` writes it so a serialised tree can be read back, and a constructor
# has no use for a tag naming the class being constructed. Excluded from the
# mirror comparison for that reason and no other.
WIRE_ONLY = {"node"}

SAMPLE_FIELD = core.Field("reference", "sector")

# One instance of each library node, so its serialised `node` string can be read
# from the library rather than restated here.
SAMPLES: dict[type, core.Expression] = {
    core.Field: SAMPLE_FIELD,
    core.Comparison: core.Comparison(SAMPLE_FIELD, "eq", "Technology"),
    core.All: core.All([SAMPLE_FIELD]),
    core.Any_: core.Any_([SAMPLE_FIELD]),
    core.Not: core.Not(SAMPLE_FIELD)}


def library_nodes() -> set[type]:
    """Every concrete `Expression` subclass, found rather than listed.

    Walked from the base class so a sixth node kind is discovered the moment it
    is defined. Names beginning with an underscore are the shared bases (`_Group`
    behind `All` and `Any_`), which serialise nothing of their own.
    """
    found: set[type] = set()
    pending = list(core.Expression.__subclasses__())

    while pending:
        cls = pending.pop()
        pending.extend(cls.__subclasses__())

        if not cls.__name__.startswith("_"):
            found.add(cls)

    return found


def required_arguments(cls: type) -> set[str]:
    """The constructor arguments that have no default."""
    return {name for name, parameter in inspect.signature(cls).parameters.items()
            if parameter.default is inspect.Parameter.empty}


def defaulted_arguments(cls: type) -> set[str]:
    """The constructor arguments that may be omitted."""
    return {name for name, parameter in inspect.signature(cls).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def required_fields(model: type) -> set[str]:
    """The wire model's required fields."""
    return {name for name, field in model.model_fields.items()  # type: ignore[attr-defined]
            if field.is_required()}


def optional_fields(model: type) -> set[str]:
    """The wire model's fields that may be omitted."""
    return {name for name, field in model.model_fields.items()  # type: ignore[attr-defined]
            if not field.is_required()}


def comparison_of(field: dict[str, Any],
                  comparison: str,
                  **extra: Any) -> dict[str, Any]:
    """A comparison node, with `value` supplied or deliberately missing."""
    return {"node": "comparison", "field": field,
            "comparison": comparison, **extra}


@pytest.fixture(scope="module")
def fetcher():
    return dataset.data_fetcher()


@pytest.fixture(scope="module")
def client(fetcher):
    app = create_app(ServerConfig(auth_token=TOKEN,
                                 data_fetcher=fetcher,
                                 storage_root=Path(tempfile.mkdtemp())))

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def spec(client):
    return client.get("/openapi.json").json()


@pytest.fixture(scope="module")
def schemas(spec):
    return spec["components"]["schemas"]


@pytest.fixture(scope="module")
def rule_types(client):
    return client.get("/indices/rule-types", headers=HEADERS).json()


@pytest.fixture(scope="module")
def expression():
    """A screen over two namespaces, so the round trip carries a nest."""
    return (data.reference.sector == "Technology") | (data.market.close > 1.0)


def selection_rule(rule_types: dict[str, Any],
                   name: str) -> dict[str, Any]:
    """One published selection rule's spec."""
    return next(spec for spec in rule_types["selection"]
                if spec["name"] == name)


def parameter_of(rule: dict[str, Any],
                 name: str) -> dict[str, Any]:
    """One parameter of a published spec."""
    return next(item for item in rule["parameters"] if item["name"] == name)


class TestTheUnionIsComplete:
    """What a client generates its editor from, checked against the library."""

    def test_every_node_kind_has_an_arm(self):
        """The guard against publishing a grammar missing a kind."""
        assert set(MIRROR.values()) == library_nodes()

    def test_each_arm_pins_the_kind_the_library_serialises(self):
        """The `Literal` on each arm, against what `to_dict` actually writes."""
        for model, cls in MIRROR.items():
            published = get_args(model.model_fields["node"].annotation)[0]

            assert published == SAMPLES[cls].to_dict()["node"]

    def test_the_discriminator_maps_every_arm(self,
                                              schemas):
        """A real `discriminator`, unlike the job-kind union: `node` is a plain
        literal, so the mapping is complete and true rather than a claim that
        the value names a schema when it names an index."""
        published = schemas["ExpressionNode"]["discriminator"]

        assert published["propertyName"] == "node"
        assert set(published["mapping"]) == {
            SAMPLES[cls].to_dict()["node"] for cls in MIRROR.values()}

    def test_the_mapping_points_at_the_published_arms(self,
                                                      schemas):
        for kind, reference in schemas["ExpressionNode"]["discriminator"][
                "mapping"].items():
            name = reference.rsplit("/", 1)[-1]

            assert name in schemas
            assert schemas[name]["properties"]["node"]["const"] == kind

    def test_the_comparison_enum_is_the_library_tuple(self,
                                                      schemas):
        """Sourced from `COMPARISONS`, so a ninth comparison is published by
        adding it there and nowhere else. 'in' versus 'between' is exactly what
        a client gets wrong once and then ships."""
        published = schemas["ComparisonNode"]["properties"]["comparison"]

        assert published["enum"] == list(COMPARISONS)

    def test_an_unknown_comparison_is_refused(self):
        with pytest.raises(ValueError, match="not a comparison"):
            ExpressionNode.model_validate(
                comparison_of(SAMPLE_FIELD.to_dict(), "approximately",
                              value=1))

    def test_the_filter_fields_reference_it(self,
                                           schemas):
        """Both of them: the document a client reads and the body it writes."""
        for model in ("Universe", "UniverseCreate"):
            published = schemas[model]["properties"]["filter"]

            assert {"$ref": "#/components/schemas/ExpressionNode"} in published[
                "anyOf"]

    def test_the_recursion_terminates_in_the_named_union(self,
                                                        schemas):
        """`operands` and `operand` point back at `ExpressionNode` rather than
        at an untyped object, which is what lets a generated client nest."""
        reference = {"$ref": "#/components/schemas/ExpressionNode"}

        assert schemas["AllNode"]["properties"]["operands"]["items"] == reference
        assert schemas["AnyNode"]["properties"]["operands"]["items"] == reference
        assert schemas["NotNode"]["properties"]["operand"] == reference


class TestTheMirrorDoesNotDrift:
    """Required fields, against the constructors they mirror, both directions.

    A mirror that drifts is a 422 the types called fine. `required` is the
    highest-value thing in the document to get right for exactly that reason:
    every other kind of schema wrongness surfaces as a type error somewhere,
    and this one surfaces in production.
    """

    @pytest.mark.parametrize("model", list(MIRROR))
    def test_it_demands_what_the_constructor_demands(self,
                                                     model):
        """A field the constructor defaults and the model requires would
        reject documents the library accepts."""
        assert required_fields(model) - WIRE_ONLY == required_arguments(
            MIRROR[model])

    @pytest.mark.parametrize("model", list(MIRROR))
    def test_it_defaults_what_the_constructor_defaults(self,
                                                       model):
        """The other direction, which is the one the spike got wrong: a field
        the model defaults and the constructor requires accepts documents the
        server then refuses."""
        assert optional_fields(model) == defaulted_arguments(MIRROR[model])

    @pytest.mark.parametrize("model", list(MIRROR))
    def test_the_discriminator_is_required_with_no_default(self,
                                                          model):
        """Both halves matter. A default makes `node` optional on input and
        required on output, which splits the model into an `-Input`/`-Output`
        pair for a reason unrelated to the recursion."""
        field = model.model_fields["node"]

        assert field.is_required()
        assert field.get_default() is PydanticUndefined

    def test_a_valueless_comparison_is_refused_at_the_edge(self,
                                                           client):
        """The defect a client found by generating types: `value` omitted
        type-checked, and then 422'd on submit."""
        response = client.post("/universes", headers=HEADERS,
                               json={"name": "Valueless",
                                     "filter": comparison_of(
                                         data.market.close.to_dict(), "gt")})

        assert response.status_code == 422

    def test_the_library_would_not_have_caught_it(self):
        """Why the edge is the right place to refuse it, and why this is an
        improvement rather than a relocation: `from_dict` reads `value`
        positionally, so a missing one raised `KeyError` — not an
        `ExpressionError` — and reached the client as an unlabelled 500."""
        with pytest.raises(KeyError):
            from_dict(comparison_of(SAMPLE_FIELD.to_dict(), "gt"))


class TestNothingIsPublishedTwice:
    """`separate_input_output_schemas=False` on the app factory.

    FastAPI splits a model into an `-Input`/`-Output` pair when it cannot prove
    the validation and serialisation schemas match, and recursion defeats that
    proof. The pair differs only in which siblings it references, so it is
    duplication rather than disagreement — and a client generator turns it into
    two parallel type trees.
    """

    def test_no_schema_name_is_split(self,
                                     schemas):
        assert [name for name in schemas
                if name.endswith(("-Input", "-Output"))] == []

    def test_the_grammar_is_published_once_each(self,
                                                schemas):
        """Guards the test above, which would also pass if the models were
        missing entirely."""
        for model in MIRROR:
            assert model.__name__ in schemas


class TestAParameterCanNameItsSchema:
    """`ParameterSpec.ref`, from `GET /indices/rule-types`."""

    def test_the_expression_parameter_names_the_grammar(self,
                                                        rule_types):
        """Without this a client meeting `ExpressionRule` learns only that a
        parameter called `expression` exists and is not a number, and its only
        recourse is to special-case the name."""
        parameter = parameter_of(selection_rule(rule_types, "ExpressionRule"),
                                 "expression")

        assert parameter["ref"] == "ExpressionNode"

    def test_it_comes_from_the_class(self,
                                     rule_types):
        """Declared the way `Constraint.UNIT` is, not inferred from the
        parameter's name — an inferred ref is the same special-casing, moved
        server-side."""
        from beacon.index.expression_rules import ExpressionRule

        rule = selection_rule(rule_types, "ExpressionRule")
        served = {item["name"]: item["ref"] for item in rule["parameters"]
                  if item["ref"] is not None}

        assert served == ExpressionRule.PARAM_SCHEMAS

    def test_the_coarse_render_hint_is_unchanged(self,
                                                 rule_types):
        """`type` stays `json`, so a client that ignores `ref` still renders
        something usable."""
        parameter = parameter_of(selection_rule(rule_types, "ExpressionRule"),
                                 "expression")

        assert parameter["type"] == "json"

    def test_a_scalar_parameter_has_none(self,
                                         rule_types):
        """Guards the tests above: a ref on everything would pass them and
        mean nothing."""
        parameter = parameter_of(selection_rule(rule_types, "ExpressionRule"),
                                 "on_missing")

        assert parameter["ref"] is None

    def test_no_ref_dangles(self,
                            rule_types,
                            schemas):
        """Every ref names a schema in this document, so a client can always
        resolve one it is handed."""
        for spec in rule_types["selection"] + rule_types["weighting"]:
            for parameter in spec["parameters"]:
                if parameter["ref"] is not None:
                    assert parameter["ref"] in schemas

    def test_constraint_specs_resolve_too(self,
                                          client,
                                          schemas):
        """The same adapter serves the optimiser's editor, so the same
        guarantee has to hold there."""
        response = client.get("/optimise/constraint-types", headers=HEADERS)

        for spec in response.json()["specs"]:
            for parameter in spec["parameters"]:
                if parameter["ref"] is not None:
                    assert parameter["ref"] in schemas


class TestPublishingChangedNothing:
    """The round trip, which is the whole claim: a tree that goes through the
    API is the same tree, byte for byte, and resolves to the same members.

    A definition built in the editor and one built in a notebook have to be the
    same document on disk, or a backtest is not reproducible from either.
    """

    def test_the_document_echoes_the_tree_unchanged(self,
                                                   client,
                                                   expression):
        created = client.post("/universes", headers=HEADERS,
                              json={"name": "Echoed",
                                    "filter": expression.to_dict()})

        assert created.status_code == 201
        assert created.json()["filter"] == expression.to_dict()

    def test_the_stored_document_is_the_same_tree(self,
                                                 client,
                                                 expression):
        """Read back through the store rather than from the response, because
        the stored bytes are what a later run rebuilds from."""
        created = client.post("/universes", headers=HEADERS,
                              json={"name": "Stored",
                                    "filter": expression.to_dict()})
        fetched = client.get(f"/universes/{created.json()['id']}",
                             headers=HEADERS)

        assert fetched.json()["filter"] == expression.to_dict()

    def test_a_field_without_a_dataset_keeps_its_shape(self):
        """`Field.to_dict` omits a null `dataset` and a plain `model_dump`
        would write one. Invisible to `from_dict`, very visible on disk."""
        assert FieldNode.model_validate(
            SAMPLE_FIELD.to_dict()).model_dump() == SAMPLE_FIELD.to_dict()

    def test_a_field_with_a_dataset_keeps_it(self):
        """Guards the test above, which dropping `dataset` outright would
        also pass."""
        field = core.Field("features", "revenue", "fundamentals")

        assert FieldNode.model_validate(
            field.to_dict()).model_dump() == field.to_dict()

    def test_the_rebuilt_expression_resolves_identically(self,
                                                         client,
                                                         fetcher,
                                                         expression):
        """What the publication is for: the tree a client posts rebuilds via
        `from_dict` into an expression that screens the same names."""
        created = client.post("/universes", headers=HEADERS,
                              json={"name": "Resolved",
                                    "filter": expression.to_dict()})
        rebuilt = from_dict(created.json()["filter"])

        assert (universe.where(rebuilt, fetcher)
                == universe.where(expression, fetcher))

    def test_the_membership_matches_the_filter(self,
                                              client,
                                              fetcher,
                                              expression):
        """And the server resolved it to the same set, so the round trip is
        end to end rather than only in the document."""
        created = client.post("/universes", headers=HEADERS,
                              json={"name": "Members",
                                    "filter": expression.to_dict()})

        assert (created.json()["identifiers"]
                == universe.where(expression, fetcher))
