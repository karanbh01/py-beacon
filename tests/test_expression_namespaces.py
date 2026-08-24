# tests/test_expression_namespaces.py
"""BN-140: the data namespaces.

`data` is a description, not a dataset — so most of what is worth testing here
is that it resolves the right *names*, that the open half stays open, and that
the attribute protocol is not quietly broken by the dynamic lookup.
"""
import pytest

from beacon import data
from beacon.exceptions import ExpressionError
from beacon.expressions.namespaces import (
    DERIVED_COLUMNS,
    NAMESPACES,
    REFERENCE_COLUMNS,
    column_for,
)


class TestTheAcceptanceCases:
    """The four paths the issue names."""

    @pytest.mark.parametrize("field,namespace,name,dataset", [
        (data.reference.sector, "reference", "sector", None),
        (data.market.close, "market", "close", None),
        (data.market.market_cap, "market", "market_cap", None),
        (data.features.fundamentals.revenue, "features", "revenue",
         "fundamentals"),
    ])
    def test_it_names_its_dataset_type_and_column(self,
                                                  field,
                                                  namespace,
                                                  name,
                                                  dataset):
        assert (field.namespace, field.name, field.dataset) == (
            namespace, name, dataset)

    def test_two_feature_types_sharing_a_field_name_stay_distinct(self):
        """The reason features nest by type rather than flattening.

        `revenue` from a vendor and `revenue` from a user's own model are
        different series, and the API has to be able to say which it meant at
        the moment the user is choosing between them.
        """
        vendor = data.features.fundamentals.revenue
        own = data.features.derived.revenue

        assert not vendor.same_as(own)
        assert vendor.path != own.path


class TestDeclaredAndOpen:
    """The split that makes autocomplete possible without a build step."""

    def test_dir_lists_the_declared_reference_columns(self):
        assert dir(data.reference) == sorted(REFERENCE_COLUMNS)

    def test_dir_includes_the_derived_market_fields(self):
        """A caller should not have to know which side of the stored/derived
        line a datapoint falls on, so both complete together."""
        listed = dir(data.market)

        assert "close" in listed
        for derived in DERIVED_COLUMNS:
            assert derived in listed

    def test_a_column_nobody_declared_still_resolves(self):
        """Declared means *known in advance*, not allowed. A store carrying an
        extra reference column must work without a code change here."""
        field = data.reference.custom_flag

        assert field.namespace == "reference"
        assert field.name == "custom_flag"

    def test_any_feature_type_resolves(self):
        """Somebody loads `satellite_imagery` tomorrow and it works."""
        field = data.features.satellite_imagery.parking_lot_fullness

        assert field.dataset == "satellite_imagery"
        assert field.path == "features.satellite_imagery.parking_lot_fullness"

    def test_the_feature_namespace_offers_no_completions(self):
        """Which types exist is a property of the loaded data, not of this
        module. A plausible-looking fixed list would complete names that do
        not resolve, which is worse than completing nothing."""
        assert dir(data.features) == []

    def test_derived_fields_are_flagged(self):
        assert data.market.is_derived("market_cap")
        assert not data.market.is_derived("close")


class TestTheDatasetsAreClosed:
    """Unlike the fields inside them."""

    def test_an_unknown_dataset_is_refused(self):
        with pytest.raises(ExpressionError, match="no 'refrence' dataset"):
            data.refrence  # noqa: B018

    def test_the_message_lists_what_does_exist(self):
        with pytest.raises(ExpressionError) as raised:
            data.nonsense  # noqa: B018

        for namespace in NAMESPACES:
            assert namespace in str(raised.value)

    def test_dir_lists_the_namespaces(self):
        """`dir()` sorts whatever `__dir__` returns, so the declaration order
        here is not what a user sees."""
        assert dir(data) == sorted(NAMESPACES)


class TestTheAttributeProtocol:
    """What a dynamic `__getattr__` breaks if it is careless."""

    def test_hasattr_answers_rather_than_raising(self):
        """`hasattr` catches `AttributeError` and nothing else, so an unknown
        dataset raising only a `BeaconError` would make `hasattr` blow up
        instead of returning False."""
        assert hasattr(data, "market")
        assert not hasattr(data, "refrence")

    def test_getattr_with_a_default_works(self):
        assert getattr(data, "refrence", "fallback") == "fallback"

    def test_a_dunder_is_not_a_field(self):
        """`__getattr__` runs for every dunder Python looks up on an object it
        is copying, pickling or inspecting. Returning a `Field` for those
        makes the namespace behave bizarrely under a debugger — and `copy`
        and `pickle` both probe for exactly these.
        """
        # Not `__getstate__`: `object` has defined it since Python 3.11, so
        # it resolves normally and never reaches `__getattr__` at all.
        for name in ("__deepcopy__", "__copy__", "__iter__"):
            assert not hasattr(data.market, name)
            assert not hasattr(data.features.fundamentals, name)

    def test_a_namespace_survives_being_copied(self):
        """The concrete consequence of the rule above."""
        import copy

        assert copy.deepcopy(data.market).declared == data.market.declared

    def test_repr_says_what_it_is(self):
        assert repr(data) == "<beacon.data>"
        assert repr(data.market) == "<data.market>"
        assert repr(data.features.fundamentals) == "<data.features.fundamentals>"


class TestColumnMapping:
    """Lower case in the API, upper case in the store."""

    def test_a_reference_field_maps_to_its_stored_column(self):
        assert column_for(data.reference.sector) == "SECTOR"
        assert column_for(data.reference.country_domicile) == "COUNTRY_DOMICILE"

    def test_a_market_field_maps_to_its_stored_column(self):
        assert column_for(data.market.close) == "CLOSE"
        assert column_for(data.market.shares_outstanding) == "SHARES_OUTSTANDING"

    def test_a_feature_field_keeps_its_case(self):
        """A feature's `FIELD` is whatever the loader wrote, so upper-casing
        it would stop matching the data it names."""
        assert column_for(data.features.fundamentals.pe_ratio) == "pe_ratio"


class TestItComposesWithTheCore:
    """The namespaces exist to be compared and stored."""

    def test_a_screen_reads_the_way_the_issue_writes_it(self):
        screen = ((data.reference.sector == "Financials")
                  & (data.market.adv_3m > 1_000_000)
                  & (data.features.fundamentals.revenue > 1e9))

        assert len(screen.to_dict()["operands"]) == 3

    def test_it_round_trips(self):
        from beacon.expressions import from_dict

        screen = (data.market.market_cap > 1e9) | (data.reference.region == "EU")

        assert from_dict(screen.to_dict()).to_dict() == screen.to_dict()

    def test_the_root_is_importable_from_the_package(self):
        """`from beacon import data` is the documented entry point."""
        import beacon

        assert beacon.data is data
