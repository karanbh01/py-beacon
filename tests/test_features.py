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


class TestPointInTimeReads:
    """BN-135: the accessor, and the look-ahead guard.

    `DATE` holding the announcement date (BN-134) only matters because of what
    happens here. These tests stand on a series of dates around a publication
    and assert what is visible from each.
    """

    @staticmethod
    def published(*records) -> FeatureData:
        """A table with a publication history."""
        return FeatureData.from_dataframe(rows(*records))

    def test_a_value_is_invisible_before_it_was_published(self):
        """The whole point. A backtest standing on 1 April must not see Q1
        revenue that nobody knew until 15 May, however completely the quarter
        it describes had ended."""
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0,
                                "DETAIL": "FY24Q1"})

        assert table.value_as_of("AAA", "revenue", "2024-04-01") is None

    def test_it_is_invisible_the_day_before(self):
        """Off-by-one on a date boundary is the classic way this goes wrong,
        so the boundary is tested rather than assumed."""
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0})

        assert table.value_as_of("AAA", "revenue", "2024-05-14") is None

    def test_it_is_visible_on_the_day(self):
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0})

        assert table.value_as_of("AAA", "revenue", "2024-05-15") == 1000.0

    def test_it_stays_in_force_until_the_next_one(self):
        """Not the row *for* the date — the latest thing knowable on it.
        Fundamentals are quarterly and a backtest runs daily, so a query on an
        ordinary Tuesday has to resolve to the last published figure."""
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0})

        assert table.value_as_of("AAA", "revenue", "2024-07-01") == 1000.0

    def test_a_restatement_takes_over_only_from_its_own_date(self):
        """Standing in June, the May figure is still what was believed. The
        August restatement had not happened, and a backtest that saw it would
        be trading on information from the future."""
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0},
                               {"DATE": "2024-08-15", "VALUE": 1050.0})

        assert table.value_as_of("AAA", "revenue", "2024-06-01") == 1000.0
        assert table.value_as_of("AAA", "revenue", "2024-08-15") == 1050.0

    def test_the_history_is_recoverable(self):
        """A restatement is kept rather than overwritten, so "what did we
        believe, and when" is answerable."""
        table = self.published({"DATE": "2024-05-15", "VALUE": 1000.0},
                               {"DATE": "2024-08-15", "VALUE": 1050.0})

        assert len(table.history("AAA", "revenue")) == 2
        assert len(table.history("AAA", "revenue", "2024-06-01")) == 1


class TestStaleness:
    """Never looking too far back."""

    def test_a_very_old_value_is_not_reported_as_current(self):
        """Serving a six-year-old fundamental as current is worse than
        serving nothing: nothing is visibly a gap, and a stale number is a
        plausible answer that makes every screen built on it wrong."""
        table = FeatureData.from_dataframe(rows({"DATE": "2018-05-15",
                                                 "VALUE": 1000.0}))

        assert table.value_as_of("AAA", "revenue", "2025-01-01") is None

    def test_the_bound_can_be_lifted(self):
        """A caller wanting no bound takes responsibility for it."""
        table = FeatureData.from_dataframe(rows({"DATE": "2018-05-15",
                                                 "VALUE": 1000.0}))

        assert table.value_as_of("AAA", "revenue", "2025-01-01",
                                 max_age_days=None) == 1000.0

    def test_a_recent_value_is_unaffected(self):
        """Guards the tests above: a bound that rejected everything would
        pass them."""
        table = FeatureData.from_dataframe(rows({"DATE": "2024-05-15",
                                                 "VALUE": 1000.0}))

        assert table.value_as_of("AAA", "revenue", "2024-06-01") == 1000.0


class TestMissingCoverageIsAnOrdinaryAnswer:
    """Most datasets cover most names most of the time, and not all of them
    all of it."""

    def test_an_unknown_instrument_is_none_not_an_error(self):
        table = FeatureData.from_dataframe(rows({}))

        assert table.value_as_of("NOSUCH", "revenue", "2024-06-01") is None

    def test_an_unknown_field_is_none_not_an_error(self):
        table = FeatureData.from_dataframe(rows({}))

        assert table.value_as_of("AAA", "nosuch", "2024-06-01") is None

    def test_an_empty_table_answers_rather_than_raising(self):
        assert FeatureData.empty().value_as_of("AAA", "revenue",
                                               "2024-06-01") is None


class TestTypeDisambiguates:
    """Two datasets carrying the same field name."""

    def test_a_type_selects_between_them(self):
        table = FeatureData.from_dataframe(
            rows({"TYPE": "fundamentals", "VALUE": 1000.0},
                 {"TYPE": "derived", "VALUE": 9999.0}))

        assert table.value_as_of("AAA", "revenue", "2024-06-01",
                                 feature_type="fundamentals") == 1000.0
        assert table.value_as_of("AAA", "revenue", "2024-06-01",
                                 feature_type="derived") == 9999.0


class TestTheFetcherSurface:
    """What callers outside this module use."""

    @staticmethod
    def fetcher(*records) -> DataFetcher:
        source = dataset.data_fetcher()

        return DataFetcher(source.market, source.reference,
                           source.corporate_actions,
                           FeatureData.from_dataframe(rows(*records)))

    def test_one_value(self):
        fetcher = self.fetcher({"DATE": "2024-05-15", "VALUE": 1000.0})

        assert fetcher.fetch_feature("AAA", "revenue", "2024-06-01") == 1000.0

    def test_the_batch_form_answers_every_pair(self):
        """Present-and-null rather than absent, the contract BN-131 set: a
        caller reads a value rather than testing for a key."""
        fetcher = self.fetcher({"IDENTIFIER": "AAA", "VALUE": 1000.0})

        answer = fetcher.fetch_features(["AAA", "NOSUCH"],
                                        ["revenue", "missing"], "2024-06-01")

        assert answer["AAA"]["revenue"] == 1000.0
        assert answer["AAA"]["missing"] is None
        assert answer["NOSUCH"]["revenue"] is None

    def test_discovery_lists_what_is_loaded(self):
        """So a client populates a control without hard-coding a vocabulary."""
        fetcher = self.fetcher({"TYPE": "fundamentals", "FIELD": "revenue"},
                               {"TYPE": "alternative", "FIELD": "card_spend"})

        assert fetcher.feature_types() == ["alternative", "fundamentals"]
        assert fetcher.feature_fields("alternative") == ["card_spend"]

    def test_a_fetcher_without_features_answers_none(self):
        """A server holding no features still responds to the question."""
        assert dataset.data_fetcher().fetch_feature(
            "AAA", "revenue", "2024-06-01") is None
