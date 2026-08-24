# tests/test_expressions.py
"""BN-139: the expression core.

Most of this file is about the two hazards of building a deferred-comparison
DSL, because both fail *silently* in ordinary use and neither shows up in a
test that only checks the happy path.
"""
import pytest

from beacon.exceptions import ExpressionError
from beacon.expressions import (
    Field,
    distinct_fields_in,
    fields_in,
    from_dict,
)
from beacon.expressions.core import All, Any_, Not

SECTOR = Field("reference", "sector")
ADV = Field("market", "adv_3m")
REVENUE = Field("features", "revenue", dataset="fundamentals")


class TestBuildingComparisons:
    """The operators, and the methods they are sugar for."""

    def test_an_operator_builds_a_tree_rather_than_answering(self):
        built = SECTOR == "Financials"

        assert built.to_dict() == {
            "node": "comparison",
            "field": {"node": "field", "namespace": "reference",
                      "name": "sector"},
            "comparison": "eq", "value": "Financials"}

    @pytest.mark.parametrize("built,expected", [
        (ADV > 1, "gt"), (ADV >= 1, "ge"), (ADV < 1, "lt"), (ADV <= 1, "le"),
        (ADV == 1, "eq"), (ADV != 1, "ne")])
    def test_every_operator_maps_to_its_word(self,
                                             built,
                                             expected):
        """Words, not symbols: these are stored in a JSON document and read
        back by a client, and `>=` survives that where an operator object
        does not."""
        assert built.comparison == expected

    def test_the_methods_and_the_operators_agree(self):
        """The operators are sugar over the methods, so a user who finds
        operator overloading surprising has a plain way through — and it has
        to be the same path, not a second one that can drift."""
        assert (ADV > 1_000).to_dict() == ADV.gt(1_000).to_dict()
        assert (SECTOR == "X").to_dict() == SECTOR.eq("X").to_dict()

    def test_between_is_inclusive_and_ordered(self):
        assert REVENUE.between(1, 10).to_dict()["value"] == [1, 10]

    def test_a_backwards_between_is_refused(self):
        """An empty range is a typo, not a screen that matches nothing."""
        with pytest.raises(ExpressionError, match="empty"):
            REVENUE.between(10, 1)

    def test_is_in_takes_a_set_of_values(self):
        built = SECTOR.is_in(["Energy", "Utilities"])

        assert built.comparison == "in"
        assert built.value == ["Energy", "Utilities"]

    def test_an_unknown_comparison_is_refused(self):
        from beacon.expressions.core import Comparison

        with pytest.raises(ExpressionError, match="unknown comparison"):
            Comparison(ADV, "approximately", 1)


class TestTheTruthValueHazard:
    """`and`/`or` cannot be overloaded, so `__bool__` has to raise.

    Python evaluates `a and b` by taking `bool(a)` and returning one operand
    or the other. There is no hook. If `__bool__` returned anything,
    `(a == 1) and (b > 2)` would discard half the expression and screen on the
    second half alone — silently, and plausibly enough that nobody checks.
    """

    def test_bool_raises(self):
        with pytest.raises(ExpressionError):
            bool(ADV > 1)

    def test_the_message_names_the_fix(self):
        """The error is the only place a user learns this, so it has to say
        what to type instead rather than merely that something is wrong."""
        with pytest.raises(ExpressionError) as raised:
            bool(ADV > 1)

        message = str(raised.value)

        assert "&" in message and "|" in message
        assert "parenthesise" in message

    def test_and_raises_rather_than_discarding_half(self):
        with pytest.raises(ExpressionError):
            (ADV > 1) and (ADV < 2)  # noqa: B018

    def test_or_raises(self):
        with pytest.raises(ExpressionError):
            (ADV > 1) or (ADV < 2)  # noqa: B018

    def test_if_raises(self):
        with pytest.raises(ExpressionError):
            if ADV > 1:
                pass

    def test_a_bare_field_has_no_truth_value_either(self):
        """`if data.market.adv_3m:` is as meaningless as the comparison, and
        the guard lives on the base class so no node type can forget it."""
        with pytest.raises(ExpressionError):
            bool(ADV)


class TestTheEqualityHazard:
    """`__eq__` returning a tree breaks three things Python assumes."""

    def test_an_assert_on_an_expression_does_not_pass_silently(self):
        """The one that matters most.

        `assert expr == x` is the natural thing to write in a test. With
        `__eq__` building a tree and `__bool__` returning True, it would
        assert nothing at all and pass — so the test suite for anything built
        on expressions would be quietly worthless. It raises instead.
        """
        with pytest.raises(ExpressionError):
            assert SECTOR == "Financials"

    def test_membership_of_the_same_object_short_circuits_on_identity(self):
        """CPython checks `is` before `==`, so this never reaches `__eq__`.

        Worth pinning rather than assuming: the obvious expectation is that
        `in` raises here, and it does not. It is the *distinct but equal* case
        below that raises, and knowing which is which is the difference
        between trusting a containment check and being surprised by one.
        """
        assert SECTOR in [SECTOR, ADV]

    def test_membership_of_an_equal_object_raises(self):
        """No identity to short-circuit on, so `__eq__` runs and returns a
        tree, which `in` then takes a truth value of."""
        with pytest.raises(ExpressionError):
            Field("reference", "sector") in [SECTOR]  # noqa: B015

    def test_a_field_is_still_hashable(self):
        """Defining `__eq__` sets `__hash__` to None unless it is declared,
        which would make a field unhashable outright."""
        assert hash(SECTOR) == hash(Field("reference", "sector"))
        assert {SECTOR: 1}[SECTOR] == 1

    def test_a_set_of_distinct_but_equal_fields_raises(self):
        """The limit of what declaring `__hash__` buys.

        Equal hashes send the set to `__eq__` to resolve the collision, which
        returns a tree. So a `Field` is a safe dict key only when the same
        instance is reused — hence `Field.key`, and hence
        `distinct_fields_in` deduplicating by that rather than by a set.
        """
        with pytest.raises(ExpressionError):
            {SECTOR, Field("reference", "sector")}

    def test_key_is_the_safe_dict_key(self):
        assert SECTOR.key == Field("reference", "sector").key
        assert len({SECTOR.key, ADV.key, Field("reference", "sector").key}) == 2

    def test_the_dataset_is_part_of_the_identity(self):
        """Two vendors may both ship `revenue`; they are not the same field."""
        other = Field("features", "revenue", dataset="alternative")

        assert hash(REVENUE) != hash(other)
        assert not REVENUE.same_as(other)

    def test_same_as_is_the_plain_comparison(self):
        assert REVENUE.same_as(Field("features", "revenue",
                                     dataset="fundamentals"))
        assert not SECTOR.same_as("reference.sector")


class TestComposition:
    """`&`, `|` and `~`."""

    def test_and_composes(self):
        built = (ADV > 1) & (SECTOR == "Energy")

        assert built.to_dict()["node"] == "all"
        assert len(built.to_dict()["operands"]) == 2

    def test_or_composes(self):
        assert ((ADV > 1) | (ADV < 0)).to_dict()["node"] == "any"

    def test_not_composes(self):
        assert (~(ADV > 1)).to_dict()["node"] == "not"

    def test_a_chain_flattens_into_one_group(self):
        """`(a & b) & c` is one group of three, not a nest of two, so that
        two ways of writing the same screen store identically."""
        built = (ADV > 1) & (ADV < 9) & (SECTOR == "Energy")

        assert len(built.to_dict()["operands"]) == 3

    def test_mixed_operators_do_not_flatten(self):
        """`(a & b) | c` must keep its shape — flattening it would change
        what it means."""
        built = ((ADV > 1) & (ADV < 9)) | (SECTOR == "Energy")
        node = built.to_dict()

        assert node["node"] == "any"
        assert node["operands"][0]["node"] == "all"

    def test_an_empty_group_is_refused(self):
        with pytest.raises(ExpressionError, match="at least one"):
            All([])


class TestRoundTripping:
    """A saved definition must screen identically when reloaded."""

    @pytest.mark.parametrize("built", [
        SECTOR == "Financials",
        ADV > 1_000_000,
        REVENUE.between(1, 10),
        SECTOR.is_in(["Energy", "Utilities"]),
        (ADV > 1) & (SECTOR == "Energy"),
        ((ADV > 1) & (ADV < 9)) | ~(SECTOR == "Energy"),
    ])
    def test_it_survives_a_round_trip(self,
                                      built):
        assert from_dict(built.to_dict()).to_dict() == built.to_dict()

    def test_it_survives_json(self):
        """The envelope is stored as JSON, so surviving `to_dict` is not
        enough on its own — a tuple, a set or a numpy scalar would pass the
        dict comparison and fail here."""
        import json

        built = ((ADV > 1) & (SECTOR.is_in(["Energy"]))) | ~(REVENUE < 5)
        restored = from_dict(json.loads(json.dumps(built.to_dict())))

        assert restored.to_dict() == built.to_dict()

    def test_the_dataset_survives(self):
        assert from_dict(REVENUE.to_dict()).dataset == "fundamentals"

    def test_a_malformed_node_is_refused(self):
        with pytest.raises(ExpressionError, match="cannot rebuild"):
            from_dict({"comparison": "gt"})

    def test_an_unknown_node_kind_is_refused(self):
        with pytest.raises(ExpressionError, match="unknown expression node"):
            from_dict({"node": "sometimes"})

    def test_a_comparison_needing_a_field_gets_one(self):
        with pytest.raises(ExpressionError, match="expected a field node"):
            from_dict({"node": "comparison", "field": {"node": "all"},
                       "comparison": "gt", "value": 1})


class TestIntrospection:
    """What a caller needs before running a screen."""

    def test_it_lists_the_fields_an_expression_mentions(self):
        built = ((ADV > 1) & (SECTOR == "Energy")) | ~(REVENUE < 5)
        paths = [field.path for field in fields_in(built)]

        assert paths == ["market.adv_3m", "reference.sector",
                         "features.fundamentals.revenue"]

    def test_it_deduplicates_by_datapoint(self):
        """A screen naming the same field twice depends on one datapoint, and
        the caller checking coverage should be told that once."""
        built = (ADV > 1) & (Field("market", "adv_3m") < 9) & (SECTOR == "X")
        paths = [field.path for field in distinct_fields_in(built)]

        assert paths == ["market.adv_3m", "reference.sector"]

    def test_the_path_reads_the_way_it_was_written(self):
        assert REVENUE.path == "features.fundamentals.revenue"
        assert SECTOR.path == "reference.sector"

    def test_repr_is_readable(self):
        """A user debugging a screen sees this, and a tree of angle brackets
        and object addresses would tell them nothing."""
        assert repr(ADV > 1) == "(market.adv_3m gt 1)"
        assert repr(~(ADV > 1)) == "~(market.adv_3m gt 1)"


class TestNodesAreComplete:
    """A new node type must serialise and round-trip.

    The failure mode of forgetting is silent: the node works in Python and
    only fails when somebody saves a definition and reloads it, which is long
    after it was written.
    """

    def test_every_node_type_round_trips(self):
        from beacon.expressions.core import Expression

        subclasses = {cls for cls in _descendants(Expression)
                      if not cls.__name__.startswith("_")}
        covered = {Field, type(ADV > 1), All, Any_, Not}

        assert subclasses == covered, (
            f"unhandled node type(s): {subclasses - covered}; add it to "
            "`from_dict` and to this test")


def _descendants(cls) -> set:
    found = set()

    for subclass in cls.__subclasses__():
        found.add(subclass)
        found |= _descendants(subclass)

    return found
