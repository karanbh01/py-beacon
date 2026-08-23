# tests/test_features.py
"""BN-134: the feature table.

A feature is any per-instrument datapoint that is not market data, reference
data, or a corporate action — fundamentals, alternative datasets, macro
series, and values somebody derived and imported. They share one shape, so
they share one table rather than a surface per kind, each with its own schema
and its own point-in-time rules to get subtly wrong.

**Two decisions carry the weight, and both are tested here rather than
described.**

`DATE` holds the date a value became *knowable*, not the period it describes.
A backtest standing on 1 April must not see Q1 revenue published in May. That
is enforced by the accessor (BN-135), but it is only possible because the
column means what it means, so the tests below pin the meaning.

A restatement is not a duplicate. A vendor revising Q1 revenue in August does
not erase what it said in May, and a backtest standing in June must still see
the May figure.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from beacon.data import store
from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.testing import dataset


def rows(*records) -> pd.DataFrame:
    """Build a feature frame from partial records."""
    base = {"IDENTIFIER": "AAA", "DATE": "2024-05-15",
            "TYPE": "fundamentals", "FIELD": "revenue", "VALUE": 1000.0,
            "DETAIL": None}

    return pd.DataFrame([{**base, **record} for record in records])


class TestTheShape:
    """What a feature row must carry."""

    def test_it_loads_field_value_pairs(self):
        table = FeatureData.from_dataframe(
            rows({"FIELD": "revenue", "VALUE": 1000.0},
                 {"FIELD": "total_debt", "VALUE": 250.0}))

        assert len(table.data) == 2
        assert table.fields() == ["revenue", "total_debt"]

    @pytest.mark.parametrize("column", ["IDENTIFIER", "DATE", "TYPE",
                                        "FIELD", "VALUE"])
    def test_a_missing_required_column_is_refused(self,
                                                  column):
        frame = rows({}).drop(columns=[column])

        with pytest.raises(ValueError, match=column):
            FeatureData.from_dataframe(frame)

    def test_detail_is_optional_and_never_defaulted(self):
        """An absent DETAIL means the dataset had no further context, which is
        different from an empty string. Defaulting it would invent a claim."""
        table = FeatureData.from_dataframe(rows({}).drop(columns=["DETAIL"]))

        assert "DETAIL" in table.columns
        assert table.data["DETAIL"].isna().all()

    def test_detail_takes_anything(self):
        """Deliberately unconstrained: a restatement note, a fiscal period, a
        vendor revision, a units note. A fixed vocabulary invented here would
        be invented wrong."""
        table = FeatureData.from_dataframe(
            rows({"DETAIL": "FY2024Q1"},
                 {"FIELD": "x", "DETAIL": "revised 2024-08-01, USD millions"}))

        assert set(table.data["DETAIL"]) == {"FY2024Q1",
                                             "revised 2024-08-01, USD millions"}

    @pytest.mark.parametrize("column", ["TYPE", "FIELD"])
    def test_a_blank_type_or_field_is_refused(self,
                                              column):
        """Both decide what a value *is*. A row that does not say cannot be
        queried for, so it is a row nobody can ever read back."""
        with pytest.raises(ValueError, match=column):
            FeatureData.from_dataframe(rows({column: "   "}))

    def test_the_date_is_parsed_to_a_timestamp(self):
        table = FeatureData.from_dataframe(rows({"DATE": "2024-05-15"}))
        dates = table.data.index.get_level_values("DATE")

        assert dates[0] == pd.Timestamp("2024-05-15")


class TestTypeSeparatesDatasets:
    """Several datasets share one table without colliding."""

    def test_the_same_field_from_two_types_is_two_series(self):
        """`revenue` from a vendor and `revenue` from a user's own model are
        different things, and a query has to be able to say which."""
        table = FeatureData.from_dataframe(
            rows({"TYPE": "fundamentals", "VALUE": 1000.0},
                 {"TYPE": "derived", "VALUE": 1200.0}))

        assert len(table.data) == 2
        assert table.types == ["derived", "fundamentals"]

    def test_fields_can_be_listed_per_type(self):
        table = FeatureData.from_dataframe(
            rows({"TYPE": "fundamentals", "FIELD": "revenue"},
                 {"TYPE": "alternative", "FIELD": "card_spend"}))

        assert table.fields("fundamentals") == ["revenue"]
        assert table.fields("alternative") == ["card_spend"]
        assert table.fields() == ["card_spend", "revenue"]

    def test_a_user_can_name_a_type_the_engine_never_heard_of(self):
        """`TYPE` is a label, not an enum. A closed set here would mean a code
        change every time somebody loaded a new dataset."""
        table = FeatureData.from_dataframe(rows({"TYPE": "satellite_imagery"}))

        assert table.types == ["satellite_imagery"]


class TestRestatementsAreKept:
    """The decision that makes the history recoverable."""

    def test_a_later_revision_does_not_erase_the_earlier_one(self):
        """A vendor revising Q1 revenue in August does not change what it said
        in May, and a backtest standing in June must still see the May
        figure. Overwriting would make the table smaller and the history
        unrecoverable."""
        table = FeatureData.from_dataframe(
            rows({"DATE": "2024-05-15", "VALUE": 1000.0, "DETAIL": "FY24Q1"},
                 {"DATE": "2024-08-15", "VALUE": 1050.0,
                  "DETAIL": "FY24Q1 restated"}))

        assert len(table.data) == 2
        assert sorted(table.data["VALUE"]) == [1000.0, 1050.0]

    def test_an_exact_duplicate_keeps_the_last(self):
        """Same instrument, same date, same type, same field: the same claim
        loaded twice, which is what re-importing a corrected file produces."""
        table = FeatureData.from_dataframe(
            rows({"VALUE": 1000.0}, {"VALUE": 1111.0}))

        assert len(table.data) == 1
        assert table.data["VALUE"].iloc[0] == 1111.0


class TestItPersists:
    """Round-tripping through the data store."""

    def test_it_survives_a_round_trip(self):
        source = dataset.data_fetcher()
        table = FeatureData.from_dataframe(
            rows({"IDENTIFIER": "AAA", "FIELD": "revenue", "VALUE": 1000.0},
                 {"IDENTIFIER": "BBB", "FIELD": "revenue", "VALUE": 2000.0}))

        fetcher = DataFetcher(source.market, source.reference,
                              source.corporate_actions, table)
        path = store.save(fetcher, Path(tempfile.mkdtemp()) / "store")

        restored = store.load(path).features

        pd.testing.assert_frame_equal(restored.data.reset_index(),
                                      table.data.reset_index())

    def test_a_store_without_features_stays_without(self):
        """Written only when present, so a dataset with no features
        round-trips as one with no features rather than one carrying an empty
        file."""
        path = store.save(dataset.data_fetcher(),
                          Path(tempfile.mkdtemp()) / "store")

        assert store.load(path).features.is_empty
        assert not (path / "features.csv.gz").exists()

    def test_an_absent_table_is_empty_not_none(self):
        """A dataset without features is still a dataset: callers ask it what
        it holds without checking for None first."""
        fetcher = dataset.data_fetcher()

        assert fetcher.features.is_empty
        assert fetcher.features.identifiers == []
        assert fetcher.features.types == []


class TestCoverage:
    """What `/data/coverage` reports."""

    def test_it_counts_fields_not_columns(self):
        """A feature table has five columns however many datapoints it
        carries, so a column count would report the same number for every
        store ever loaded."""
        table = FeatureData.from_dataframe(
            rows({"FIELD": "revenue"}, {"FIELD": "total_debt"},
                 {"FIELD": "pe_ratio"}))

        assert table.coverage()["fields"] == 3
        assert len(table.columns) == 4

    def test_an_empty_table_reports_zeroes_rather_than_failing(self):
        """"We hold no features" is a fact the pane should state, not a
        dataset it should omit."""
        assert FeatureData.empty().coverage() == {
            "identifiers": 0, "types": [], "fields": 0, "rows": 0}

    def test_the_endpoint_lists_it_beside_the_others(self):
        from fastapi.testclient import TestClient

        from beacon.server import ServerConfig, create_app

        source = dataset.data_fetcher()
        fetcher = DataFetcher(source.market, source.reference,
                              source.corporate_actions,
                              FeatureData.from_dataframe(rows({})))

        client = TestClient(create_app(ServerConfig(
            auth_token="t", data_fetcher=fetcher,
            storage_root=Path(tempfile.mkdtemp()))))

        body = client.get("/data/coverage",
                          headers={"Authorization": "Bearer t"}).json()
        names = [entry["dataset"] for entry in body["datasets"]]

        assert "features" in names

    def test_its_identifiers_join_the_union(self):
        """A name known only through a feature is still a name the store
        covers."""
        source = dataset.data_fetcher()
        table = FeatureData.from_dataframe(rows({"IDENTIFIER": "AAA"}))

        fetcher = DataFetcher(source.market, source.reference,
                              source.corporate_actions, table)

        assert "AAA" in fetcher.features.identifiers
