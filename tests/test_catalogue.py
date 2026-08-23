# tests/test_catalogue.py
"""BN-117: the type catalogue, and the endpoints that serve it.

The catalogue's whole claim is that it *cannot* drift from the classes, so most
of this file is about that claim rather than about the values. Two kinds of
test do the work: introspection tests, which pin what is read off a signature,
and the completeness test, which fails if a rule class exists without a
catalogue entry.

That last one matters more than it looks. A missed registration breaks nothing
at runtime — the rule works, the index runs, and the only symptom is that the
editor never offers it. That is exactly the failure a test has to catch,
because nobody is going to notice it.
"""
import gc
import inspect

import pytest
from fastapi.testclient import TestClient

from beacon import catalogue
from beacon.index.methodology import (
    EligibilityRuleBase,
    EqualWeighted,
    LiquidityRule,
    MarketCapRule,
    MarketCapWeighted,
    WeightingSchemeBase,
)
from beacon.optimise.constraints import Constraint
from beacon.server import ServerConfig, create_app
from beacon.server.constraints import constraint_params
from beacon.server.definitions import selection_rules, weighting_schemes

TOKEN = "test-token-value"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A server with nothing configured: the catalogue needs no data."""
    return TestClient(create_app(ServerConfig(auth_token=TOKEN)))


def subclasses_of(base: type,
                  library_only: bool = True) -> set[type]:
    """Every subclass of a base, transitively.

    Args:
        base: The abstract base to walk.
        library_only: Restrict to classes defined under `beacon.`. The
            completeness check wants this: a subclass defined inside a test is
            not part of the product and must not make the real check fail if it
            outlives the test that made it.
    """
    found: set[type] = set()

    for cls in base.__subclasses__():
        if not library_only or cls.__module__.startswith("beacon."):
            found.add(cls)
        found |= subclasses_of(cls, library_only)

    return found


class TestDisplayType:
    """Annotations to the control a client should render."""

    def test_scalars(self):
        assert catalogue.display_type(float) == (catalogue.NUMBER, False)
        assert catalogue.display_type(int) == (catalogue.INTEGER, False)
        assert catalogue.display_type(str) == (catalogue.STRING, False)

    def test_bool_is_not_an_integer(self):
        """`bool` subclasses `int`, so an isinstance-ordered check would render
        every checkbox as a number field."""
        assert catalogue.display_type(bool) == (catalogue.BOOLEAN, False)

    def test_optional_unwraps_to_the_inner_type(self):
        assert catalogue.display_type(float | None) == (catalogue.NUMBER, True)
        assert catalogue.display_type(int | None) == (catalogue.INTEGER, True)

    def test_a_collection_falls_back_to_json(self):
        """A client cannot render `dict[str, float]` as a scalar control, and
        pretending otherwise would produce a text box that silently fails."""
        assert catalogue.display_type(dict[str, float])[0] == catalogue.JSON
        assert catalogue.display_type(list[str])[0] == catalogue.JSON

    def test_a_union_of_two_real_types_is_json(self):
        assert catalogue.display_type(int | str)[0] == catalogue.JSON

    def test_an_unannotated_parameter_is_json(self):
        assert catalogue.display_type(inspect.Parameter.empty) == (catalogue.JSON,
                                                                   False)


class TestIntrospection:
    """What is read off a constructor."""

    def test_it_reads_names_types_and_defaults(self):
        parameters = {p.name: p for p in catalogue.parameters_of(LiquidityRule)}

        assert parameters["lookback_days"].type == catalogue.INTEGER
        assert parameters["lookback_days"].default == 60
        assert parameters["min_avg_daily_value"].type == catalogue.NUMBER

    def test_a_parameter_with_a_default_is_not_required(self):
        parameters = {p.name: p for p in catalogue.parameters_of(MarketCapRule)}

        assert not parameters["min_market_cap"].required

    def test_a_parameter_without_a_default_is_required(self):
        from beacon.optimise.constraints import GroupBounds

        parameters = {p.name: p for p in catalogue.parameters_of(GroupBounds)}

        assert parameters["name"].required
        assert parameters["members"].required
        assert not parameters["minimum"].required

    def test_self_is_not_a_parameter(self):
        assert "self" not in {p.name for p in catalogue.parameters_of(MarketCapRule)}

    def test_a_constructor_taking_nothing_yields_nothing(self):
        assert catalogue.parameters_of(EqualWeighted) == ()

    def test_an_undeclared_label_is_derived_from_the_name(self):
        parameters = {p.name: p for p in catalogue.parameters_of(MarketCapRule)}

        assert parameters["min_market_cap"].label == "Min market cap"

    def test_a_declared_label_wins(self):
        entry = catalogue.entry_for(catalogue.SELECTION, "MarketCapRule")
        parameters = {p.name: p for p in entry.parameters}

        assert parameters["min_market_cap"].label == "Minimum market cap"

    def test_declared_order_beats_signature_order(self):
        """A form has a designed reading order; a signature's is whatever was
        convenient to write."""
        entry = catalogue.entry_for(catalogue.SELECTION, "LiquidityRule")

        assert [p.name for p in entry.parameters] == [
            "min_avg_daily_volume", "min_avg_daily_value", "lookback_days"]
        assert [p.order for p in entry.parameters] == [1, 2, 3]

    def test_the_summary_comes_from_the_docstring(self):
        entry = catalogue.entry_for(catalogue.WEIGHTING, "MarketCapWeighted")

        assert entry.summary.startswith("Market capitalization weighting")

    def test_a_class_with_no_docstring_has_an_empty_summary(self):
        """Rather than None or a placeholder — the field is always a string, so
        a client renders nothing instead of the word "None"."""
        class Undocumented:
            def __init__(self,
                         value: int = 1):
                self.value = value

        assert catalogue._summary_of(Undocumented) == ""

    def test_var_args_are_not_fields(self):
        """`*args` and `**kwargs` name no field a form could render."""
        class Loose:
            """Takes anything."""

            def __init__(self,
                         real: int = 1,
                         *args: object,
                         **kwargs: object):
                self.real = real

        assert [p.name for p in catalogue.parameters_of(Loose)] == ["real"]

    def test_an_optional_annotation_without_a_default_is_still_required(self):
        """`x: int | None` with no default means the caller must pass
        something — even if that something is None."""
        class Explicit:
            """Wants None spelled out."""

            def __init__(self,
                         value: int | None):
                self.value = value

        parameter = catalogue.parameters_of(Explicit)[0]

        assert parameter.required is True
        assert parameter.type == catalogue.INTEGER


class TestRegistry:
    """Lookup, and what registration does not change."""

    def test_registered_names_cover_the_library(self):
        """Literal, deliberately.

        Registration is a side effect that never changes behaviour — a rule
        works identically registered or not — so forgetting it shows up
        nowhere except here. Deriving this list from the module would make the
        test agree with whatever the code happens to do, which is the one
        thing it must not do.

        FeatureRule joined in BN-136.
        """
        assert catalogue.registered_names(catalogue.SELECTION) == {
            "MarketCapRule", "LiquidityRule", "FeatureRule"}
        assert catalogue.registered_names(catalogue.WEIGHTING) == {
            "EqualWeighted", "MarketCapWeighted"}

    def test_classes_returns_the_constructible_class(self):
        assert catalogue.classes(catalogue.SELECTION)["MarketCapRule"] is MarketCapRule
        assert (catalogue.classes(catalogue.WEIGHTING)["MarketCapWeighted"]
                is MarketCapWeighted)

    def test_an_unknown_name_is_none_rather_than_an_error(self):
        assert catalogue.entry_for(catalogue.SELECTION, "NoSuchRule") is None
        assert catalogue.parameter_names(catalogue.SELECTION, "NoSuchRule") == set()

    def test_an_unknown_kind_is_empty(self):
        assert catalogue.entries("nonsense") == []

    def test_registration_does_not_alter_the_class(self):
        """The decorator's only effect is the registry entry, which is why a
        forgotten registration cannot surface at runtime."""
        rule = MarketCapRule(min_market_cap=1.0, max_market_cap=2.0)

        assert rule.min_market_cap == 1.0
        assert isinstance(rule, EligibilityRuleBase)

    def test_entries_are_name_ordered(self):
        names = [entry.name for entry in catalogue.entries(catalogue.CONSTRAINT)]

        assert names == sorted(names)


class TestCompleteness:
    """A class that exists but is not registered is the failure to catch."""

    @pytest.mark.parametrize(("base", "kind"), [
        (EligibilityRuleBase, catalogue.SELECTION),
        (WeightingSchemeBase, catalogue.WEIGHTING),
        (Constraint, catalogue.CONSTRAINT),
    ])
    def test_every_library_class_is_registered(self, base, kind):
        missing = {cls.__name__ for cls in subclasses_of(base)
                   if not inspect.isabstract(cls)} - catalogue.registered_names(kind)

        assert not missing, (
            f"{', '.join(sorted(missing))} exist but are not in the catalogue, "
            f"so the editor cannot offer them and nothing else will complain")

    def test_an_unregistered_rule_is_detected(self):
        """The check has to actually fail on a rule that skipped the decorator,
        or it is asserting that nothing is wrong with a list of nothing."""
        class UnregisteredRule(MarketCapRule):
            """A rule nobody registered."""

        try:
            everything = {cls.__name__
                          for cls in subclasses_of(EligibilityRuleBase,
                                                   library_only=False)}

            assert "UnregisteredRule" in everything
            assert "UnregisteredRule" not in catalogue.registered_names(
                catalogue.SELECTION)
        finally:
            # Dropped so it cannot outlive this test and appear in the real
            # completeness check above. __subclasses__ holds weak references,
            # so collecting is what actually removes it.
            del UnregisteredRule
            gc.collect()

    def test_the_library_check_ignores_classes_defined_in_tests(self):
        class AlsoUnregistered(MarketCapRule):
            """Defined outside the beacon package."""

        try:
            names = {cls.__name__ for cls in subclasses_of(EligibilityRuleBase)}

            assert "AlsoUnregistered" not in names
        finally:
            del AlsoUnregistered
            gc.collect()


class TestValidationReadsTheCatalogue:
    """The tables the catalogue replaced were the ones validation used."""

    def test_selection_rules_match_the_registry(self):
        assert set(selection_rules()) == catalogue.registered_names(
            catalogue.SELECTION)

    def test_weighting_schemes_match_the_registry(self):
        assert set(weighting_schemes()) == catalogue.registered_names(
            catalogue.WEIGHTING)

    def test_parameters_match_the_constructors(self):
        """The hand-kept table could disagree with the signature and nothing
        failed; the validator simply rejected a parameter the rule accepted."""
        for name, parameters in selection_rules().items():
            cls = catalogue.classes(catalogue.SELECTION)[name]
            expected = set(inspect.signature(cls).parameters)

            assert parameters == expected, name

    def test_constraint_params_match_the_constructors(self):
        for name, parameters in constraint_params().items():
            cls = catalogue.classes(catalogue.CONSTRAINT)[name]

            assert parameters == set(inspect.signature(cls).parameters), name


class TestRuleTypesEndpoint:
    """`GET /indices/rule-types`."""

    def test_it_lists_selection_and_weighting(self, client):
        body = client.get("/indices/rule-types", headers=auth()).json()

        assert {entry["name"] for entry in body["selection"]} == {
            "MarketCapRule", "LiquidityRule", "FeatureRule"}
        assert {entry["name"] for entry in body["weighting"]} == {
            "EqualWeighted", "MarketCapWeighted"}

    def test_it_answers_without_a_data_source(self, client):
        """It describes what the library can do, not what this server holds."""
        assert client.get("/indices/rule-types",
                          headers=auth()).status_code == 200

    def test_it_requires_authentication(self, client):
        assert client.get("/indices/rule-types").status_code == 401

    def test_a_type_carries_everything_a_form_needs(self, client):
        body = client.get("/indices/rule-types", headers=auth()).json()
        rule = next(entry for entry in body["selection"]
                    if entry["name"] == "LiquidityRule")

        assert rule["label"] == "Liquidity"
        assert rule["summary"]

        lookback = next(p for p in rule["parameters"] if p["name"] == "lookback_days")
        assert lookback["type"] == "integer"
        assert lookback["default"] == 60
        assert lookback["required"] is False
        assert lookback["label"] == "Lookback"
        assert lookback["help"]

    def test_parameters_arrive_in_form_order(self, client):
        body = client.get("/indices/rule-types", headers=auth()).json()

        for entry in body["selection"] + body["weighting"]:
            orders = [p["order"] for p in entry["parameters"]]

            assert orders == sorted(orders)

    def test_the_route_is_not_read_as_an_index_id(self, client):
        """`/indices/{index_id}` would happily match "rule-types" and 404."""
        response = client.get("/indices/rule-types", headers=auth())

        assert response.status_code == 200
        assert "selection" in response.json()

    def test_a_named_type_can_be_submitted_as_is(self, client):
        """The `name` field must be what `RuleSpec.type` expects — a catalogue
        naming types the API then rejects would be worse than none."""
        body = client.get("/indices/rule-types", headers=auth()).json()

        for entry in body["selection"]:
            assert entry["name"] in selection_rules()


class TestConstraintTypesUpgrade:
    """The richer shape, added without breaking the original."""

    def test_the_original_field_is_unchanged(self, client):
        body = client.get("/optimise/constraint-types", headers=auth()).json()

        assert set(body["types"]) == set(constraint_params())
        assert body["types"]["FullInvestment"] == ["target"]

    def test_specs_carry_the_same_types(self, client):
        body = client.get("/optimise/constraint-types", headers=auth()).json()

        assert ({spec["name"] for spec in body["specs"]} == set(body["types"]))

    def test_specs_and_types_agree_on_parameters(self, client):
        """Two views of one registry; if they can disagree, one is a copy."""
        body = client.get("/optimise/constraint-types", headers=auth()).json()

        for spec in body["specs"]:
            assert sorted(p["name"] for p in spec["parameters"]) == sorted(
                body["types"][spec["name"]])

    def test_a_required_constraint_parameter_is_marked(self, client):
        body = client.get("/optimise/constraint-types", headers=auth()).json()
        group = next(s for s in body["specs"] if s["name"] == "GroupBounds")
        parameters = {p["name"]: p for p in group["parameters"]}

        assert parameters["name"]["required"] is True
        assert parameters["minimum"]["required"] is False
        assert parameters["minimum"]["default"] == 0.0

    def test_both_editors_get_the_same_shape(self, client):
        """The reason the registry is shared: one client component renders
        both forms."""
        rules = client.get("/indices/rule-types", headers=auth()).json()
        constraints = client.get("/optimise/constraint-types",
                                 headers=auth()).json()

        rule_keys = set(rules["selection"][0])
        constraint_keys = set(constraints["specs"][0])

        assert rule_keys == constraint_keys
