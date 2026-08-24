# tests/test_expression_validation.py
"""BN-141: validating an expression, and generating stubs for it.

The failure this prevents is the quiet one. `data.reference.sectr` that
resolves to nothing produces an empty index and no explanation, which looks
exactly like a legitimate answer — so it is never investigated.
"""
import ast
import logging

import pandas as pd
import pytest

from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.expressions import data
from beacon.expressions.stubs import build_parser, generate
from beacon.expressions.validation import (
    UNKNOWN_FEATURE_TYPE,
    UNKNOWN_FIELD,
    errors_in,
    is_valid,
    validate,
)
from beacon.testing import dataset


def feature_rows(*records) -> pd.DataFrame:
    base = {"IDENTIFIER": "AAA", "DATE": "2024-01-15",
            "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 1.0,
            "DETAIL": None}

    return pd.DataFrame([{**base, **record} for record in records])


@pytest.fixture(scope="module")
def stub(fetcher):
    return generate(fetcher)


@pytest.fixture(scope="module")
def fetcher():
    """A store carrying two feature datasets."""
    source = dataset.data_fetcher()
    features = FeatureData.from_dataframe(feature_rows(
        {"FIELD": "revenue"},
        {"FIELD": "pe_ratio", "VALUE": 15.0},
        {"TYPE": "alternative", "FIELD": "x_sentiment", "VALUE": 0.3}))

    return DataFetcher(source.market, source.reference,
                       source.corporate_actions, features)


class TestTheAcceptanceCase:
    """`data.reference.sectr` returns a finding naming `sector`."""

    def test_a_typo_is_caught(self,
                              fetcher):
        findings = validate(data.reference.sectr == "Financials", fetcher)

        assert len(findings) == 1
        assert findings[0].code == UNKNOWN_FIELD

    def test_the_finding_names_the_field_the_user_meant(self,
                                                        fetcher):
        """The information is already in hand by the time the error is
        written; a validator that has it and does not say so has chosen to be
        unhelpful."""
        findings = validate(data.reference.sectr == "Financials", fetcher)

        assert "sector" in findings[0].message
        assert "Did you mean" in findings[0].message

    def test_the_finding_points_at_the_offending_field(self,
                                                       fetcher):
        """A client needs an anchor, not just a message."""
        findings = validate(data.reference.sectr == "X", fetcher)

        assert findings[0].path == "reference.sectr"


class TestWhatValidates:
    """The cases that must pass."""

    @pytest.mark.parametrize("expression", [
        data.reference.sector == "Financials",
        data.market.close > 1,
        data.market.volume > 1,
        data.features.fundamentals.revenue > 1,
        data.features.alternative.x_sentiment > 0,
    ])
    def test_a_loaded_field_is_valid(self,
                                     expression,
                                     fetcher):
        assert validate(expression, fetcher) == []

    def test_a_derived_field_is_valid(self,
                                      fetcher):
        """`market_cap` is a column nowhere — it is computed per request — so
        checking the loaded frame alone would reject the field most likely to
        be screened on."""
        assert is_valid(data.market.market_cap > 1e9, fetcher)
        assert is_valid(data.market.adv_3m > 1e6, fetcher)

    def test_a_composed_expression_validates_every_branch(self,
                                                          fetcher):
        screen = ((data.reference.sector == "X")
                  & (data.market.clsoe > 1)
                  | ~(data.reference.reigon == "EU"))
        findings = validate(screen, fetcher)

        assert {finding.path for finding in findings} == {
            "market.clsoe", "reference.reigon"}

    def test_every_mistake_is_reported_at_once(self,
                                               fetcher):
        """Findings rather than an exception on the first problem: somebody
        fixing a screen wants the whole list, not one round trip per typo."""
        screen = (data.reference.sectr == "X") & (data.market.clsoe > 1)

        assert len(validate(screen, fetcher)) == 2


class TestFeatureFields:
    """The open half, checked against what is loaded."""

    def test_an_unloaded_feature_type_is_named(self,
                                               fetcher):
        findings = validate(data.features.satellite.parking > 1, fetcher)

        assert findings[0].code == UNKNOWN_FEATURE_TYPE

    def test_it_says_which_types_do_exist(self,
                                          fetcher):
        """A user who has misremembered the dataset name needs the list, not
        confirmation that they were wrong."""
        findings = validate(data.features.fundamental.revenue > 1, fetcher)
        message = findings[0].message

        assert "fundamentals" in message

    def test_a_field_missing_from_a_real_dataset_is_caught(self,
                                                           fetcher):
        findings = validate(data.features.fundamentals.revenu > 1, fetcher)

        assert findings[0].code == UNKNOWN_FIELD
        assert "revenue" in findings[0].message

    def test_the_same_field_in_the_wrong_dataset_is_caught(self,
                                                           fetcher):
        """`revenue` exists, but not in `alternative`. Checking the field name
        alone would pass this, which is the whole reason features are keyed by
        type rather than flattened."""
        findings = validate(data.features.alternative.revenue > 1, fetcher)

        assert len(findings) == 1

    def test_no_feature_data_at_all_says_so(self):
        findings = validate(data.features.fundamentals.revenue > 1,
                            dataset.data_fetcher())

        assert findings[0].code == UNKNOWN_FEATURE_TYPE
        assert "no feature data is loaded" in findings[0].message


class TestSuggestions:
    """"Did you mean" is the difference between useful and frustrating."""

    def test_an_unrelated_name_gets_no_suggestion(self,
                                                  fetcher):
        """A wrong suggestion is worse than none — it sends somebody to check
        the wrong thing."""
        findings = validate(data.reference.xyzzy == "X", fetcher)

        assert "Did you mean" not in findings[0].message

    def test_an_unrelated_name_still_lists_what_exists(self,
                                                       fetcher):
        findings = validate(data.reference.xyzzy == "X", fetcher)

        assert "Available" in findings[0].message


class TestTheFindingShape:
    """What the server returns."""

    def test_it_serialises_to_the_api_shape(self,
                                            fetcher):
        finding = validate(data.reference.sectr == "X", fetcher)[0].as_dict()

        assert set(finding) == {"path", "severity", "code", "message"}
        assert finding["severity"] == "error"

    def test_errors_are_the_blocking_subset(self,
                                            fetcher):
        screen = data.reference.sectr == "X"

        assert len(errors_in(screen, fetcher)) == 1
        assert not is_valid(screen, fetcher)


class TestStubGeneration:
    """Completions for static analysers, which do not run code."""

    def test_it_parses(self,
                       stub):
        """A stub that does not parse is worse than no stub: Pylance reports
        an error in a generated file the user did not write."""
        ast.parse(stub)

    def test_it_declares_a_class_per_loaded_dataset(self,
                                                    stub):
        assert "class _FundamentalsFields:" in stub
        assert "class _AlternativeFields:" in stub

    def test_it_declares_an_attribute_per_field(self,
                                                stub):
        assert "revenue: Field" in stub
        assert "x_sentiment: Field" in stub

    def test_the_features_namespace_names_the_datasets(self,
                                                       stub):
        assert "fundamentals: _FundamentalsFields" in stub
        assert "alternative: _AlternativeFields" in stub

    def test_it_declares_the_core_namespaces_too(self,
                                                 stub):
        assert "market: _MarketFields" in stub
        assert "market_cap: Field" in stub
        assert "data: _Data" in stub

    def test_regenerating_from_different_data_changes_it(self,
                                                         fetcher):
        """The whole reason it is generated rather than shipped."""
        source = dataset.data_fetcher()
        other = DataFetcher(source.market, source.reference,
                            source.corporate_actions,
                            FeatureData.from_dataframe(feature_rows(
                                {"TYPE": "satellite", "FIELD": "parking"})))

        assert "class _SatelliteFields:" in generate(other)
        assert "class _SatelliteFields:" not in generate(fetcher)

    def test_a_field_that_is_not_a_python_name_is_left_out(self):
        """A vendor shipping `p/e ratio` would produce a stub that does not
        parse. It is skipped rather than mangled — a mangled name would
        autocomplete to something that does not resolve."""
        source = dataset.data_fetcher()
        awkward = DataFetcher(source.market, source.reference,
                              source.corporate_actions,
                              FeatureData.from_dataframe(feature_rows(
                                  {"FIELD": "p/e ratio"},
                                  {"FIELD": "clean_name"})))

        logging.disable(logging.ERROR)

        try:
            stub = generate(awkward)
        finally:
            logging.disable(logging.NOTSET)

        ast.parse(stub)

        assert "clean_name: Field" in stub
        assert "p/e ratio" not in stub

    def test_an_awkward_field_still_resolves_at_runtime(self):
        """Left out of the stub is not the same as unusable: the namespace is
        open, so `getattr` reaches it."""
        field = getattr(data.features.fundamentals, "p/e ratio")

        assert field.name == "p/e ratio"


class TestStaleStubsAreSafe:
    """The ordering that makes generation safe to offer at all."""

    def test_a_stale_stub_does_not_authorise_a_screen(self,
                                                      fetcher):
        """A stub kept from an older store completes a field the data no
        longer carries. That must produce a *finding*, not a selection —
        validating against the stub instead of the data would let a stale file
        silently authorise a screen that then matches nothing.
        """
        source = dataset.data_fetcher()
        older = DataFetcher(source.market, source.reference,
                            source.corporate_actions,
                            FeatureData.from_dataframe(feature_rows(
                                {"FIELD": "retired_metric"})))

        assert "retired_metric: Field" in generate(older)
        assert not is_valid(data.features.fundamentals.retired_metric > 1,
                            fetcher)


class TestTheCommandLine:
    def test_the_documented_arguments_parse(self,
                                            tmp_path):
        args = build_parser().parse_args(["--out", str(tmp_path / "d.pyi")])

        assert args.out == tmp_path / "d.pyi"

    def test_help_renders(self):
        """argparse interpolates `%` in help text, so a literal percent sign
        raises at format time and nowhere else."""
        assert "--store" in build_parser().format_help()

    def test_a_missing_store_exits_two(self,
                                       tmp_path,
                                       capsys):
        from beacon.expressions.stubs import main

        code = main(["--store", str(tmp_path / "nothing"),
                     "--out", str(tmp_path / "d.pyi")])

        assert code == 2
        assert "error:" in capsys.readouterr().err


class TestTheFieldCatalogue:
    """`GET /data/fields` — what a client builds a field picker from."""

    @pytest.fixture
    def client(self,
               fetcher):
        import tempfile
        from pathlib import Path

        from fastapi.testclient import TestClient

        from beacon.server import ServerConfig, create_app

        return TestClient(create_app(ServerConfig(
            auth_token="t", data_fetcher=fetcher,
            storage_root=Path(tempfile.mkdtemp()))))

    def body(self,
             client):
        return client.get("/data/fields",
                          headers={"Authorization": "Bearer t"}).json()

    def test_it_lists_every_namespace(self,
                                      client):
        paths = {entry["namespace"] for entry in self.body(client)["fields"]}

        assert paths == {"market", "reference", "actions", "features"}

    def test_a_feature_field_carries_its_dataset(self,
                                                 client):
        """Without it a picker cannot tell two vendors' `revenue` apart, which
        is the distinction the feature table exists to preserve."""
        revenue = [entry for entry in self.body(client)["fields"]
                   if entry["name"] == "revenue"]

        assert revenue[0]["dataset"] == "fundamentals"
        assert revenue[0]["path"] == "features.fundamentals.revenue"

    def test_derived_fields_are_listed_and_flagged(self,
                                                   client):
        """Listed because they are screenable; flagged because a client may
        want to say where the number comes from."""
        derived = {entry["name"] for entry in self.body(client)["fields"]
                   if entry["derived"]}

        assert "market_cap" in derived

    def test_index_columns_are_not_offered(self,
                                           client):
        """`IDENTIFIER` and `DATE` are on every frame and screening on them is
        meaningless — listing them is noise in the picker."""
        names = {entry["name"] for entry in self.body(client)["fields"]}

        assert not names & {"identifier", "date", "date_from", "date_to"}

    def test_the_paths_are_what_the_expression_api_accepts(self,
                                                           client,
                                                           fetcher):
        """The catalogue and the expression API must not disagree about what
        exists — a picker offering a field that fails validation is worse than
        one that omits it."""
        for entry in self.body(client)["fields"]:
            namespace = getattr(data, entry["namespace"])
            reached = (getattr(getattr(namespace, entry["dataset"]),
                               entry["name"])
                       if entry["dataset"] else getattr(namespace,
                                                        entry["name"]))

            assert is_valid(reached > 0, fetcher), entry["path"]

    def test_it_is_in_the_spec(self,
                               client):
        assert "/data/fields" in client.get("/openapi.json").json()["paths"]
