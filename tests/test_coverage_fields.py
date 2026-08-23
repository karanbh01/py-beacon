# tests/test_coverage_fields.py
"""BN-119: what `/data/coverage` says about the data it holds.

The coverage pane was rendering staleness from thresholds it held itself — 24h
for one dataset, 7d for another — which is a client guessing at a property of
the data. The guess and the engine diverge the moment either changes, and
nothing tells anybody. So the engine publishes the frequency and the duration
it implies, and the client stops deciding.

The other half is arithmetic that has to hold: per-dataset sizes must be
per-dataset, and the identifier count must be a union rather than a sum, or the
pane reports more assets covered than exist.
"""
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from beacon.data import store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import (
    ACTIONS_DATASET,
    DAILY,
    EVENT,
    FREQUENCY_FOR_DATASET,
    MARKET_DATASET,
    REFERENCE_DATASET,
    STALE_AFTER_SECONDS,
    STATIC,
    DataFetcher,
)
from beacon.server import ServerConfig, create_app
from beacon.synthetic import SyntheticConfig, generate

TOKEN = "test-token-value"


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def stored():
    """A generated store on disk, and a client serving it."""
    with tempfile.TemporaryDirectory() as directory:
        dataset = generate(SyntheticConfig(assets=25, start="2023-01-02",
                                           end="2024-06-28", seed=4))
        path = store.save(dataset.fetcher(), Path(directory) / "store",
                          source=store.SOURCE_SYNTHETIC)

        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=store.load(path))))

        yield client, path


@pytest.fixture(scope="module")
def coverage(stored) -> dict:
    """The coverage payload for the stored dataset."""
    client, _ = stored

    return client.get("/data/coverage", headers=auth()).json()


def dataset_of(payload: dict,
               name: str) -> dict:
    """One dataset entry out of a coverage payload."""
    return next(entry for entry in payload["datasets"]
                if entry["dataset"] == name)


class TestFrequency:
    """The engine's definition of what stale means."""

    def test_market_data_is_daily(self, coverage):
        assert dataset_of(coverage, MARKET_DATASET)["frequency"] == DAILY

    def test_reference_data_is_static(self, coverage):
        """Names and sectors change, but not on a schedule worth refreshing
        against. Reference data a month old is not stale."""
        assert dataset_of(coverage, REFERENCE_DATASET)["frequency"] == STATIC

    def test_corporate_actions_are_event_driven(self, coverage):
        """A quiet week is not staleness."""
        assert dataset_of(coverage, ACTIONS_DATASET)["frequency"] == EVENT

    def test_the_threshold_travels_with_the_frequency(self, coverage):
        """So a client renders staleness without encoding the mapping, which
        would be the hardcoded threshold moved rather than deleted."""
        for entry in coverage["datasets"]:
            expected = STALE_AFTER_SECONDS[entry["frequency"]]

            assert entry["stale_after_seconds"] == expected

    def test_static_data_has_no_threshold(self, coverage):
        """Null rather than a very large number: the question does not apply,
        and a big number would eventually be crossed."""
        assert dataset_of(coverage,
                          REFERENCE_DATASET)["stale_after_seconds"] is None

    def test_staleness_is_derivable_from_the_response_alone(self, coverage):
        """The acceptance criterion. A client should need nothing but
        `frequency`, `stale_after_seconds` and `last_refreshed`."""
        market = dataset_of(coverage, MARKET_DATASET)

        assert market["last_refreshed"]
        assert market["cache_age"] is not None
        assert market["cache_age"] < market["stale_after_seconds"]

    def test_every_dataset_has_a_frequency(self):
        from beacon.data.fetcher import DATASETS

        assert set(FREQUENCY_FOR_DATASET) == set(DATASETS)
        assert all(frequency in STALE_AFTER_SECONDS
                   for frequency in FREQUENCY_FOR_DATASET.values())


class TestSource:
    """Where the rows came from."""

    def test_it_comes_from_the_store_manifest(self, coverage):
        """For the datasets the store actually holds.

        An unconfigured one carries no source, which is the same answer the
        empty corporate-action history has always given: there are no rows, so
        there is no provenance to report, and naming the store's would claim
        one for data that is not there. Scoped here because the fixture gained
        a fourth dataset in BN-134 that it does not populate.
        """
        configured = [entry for entry in coverage["datasets"]
                      if entry["configured"]]

        assert configured, "no dataset was configured, so this proves nothing"

        for entry in configured:
            assert entry["source"] == store.SOURCE_SYNTHETIC

    def test_an_unconfigured_dataset_claims_no_source(self, coverage):
        for entry in coverage["datasets"]:
            if not entry["configured"]:
                assert entry["source"] is None

    def test_an_in_process_fetcher_reports_no_source(self):
        """Null rather than "local": a fetcher assembled from frames has no
        provenance, and claiming one would be a guess presented as a fact."""
        fetcher = DataFetcher(MarketData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": "2025-01-02", "CLOSE": 100.0}])))
        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)))

        payload = client.get("/data/coverage", headers=auth()).json()

        assert dataset_of(payload, MARKET_DATASET)["source"] is None
        assert dataset_of(payload, MARKET_DATASET)["cache_size_bytes"] is None


class TestFieldCount:
    """How many attributes each dataset actually holds."""

    def test_market_counts_its_columns(self, coverage):
        """OHLCV, shares outstanding and free float."""
        # Eight since BN-128: the seven equity columns plus RATE, which the
        # FX pairs carry and equity rows leave empty.
        assert dataset_of(coverage, MARKET_DATASET)["field_count"] == 8

    def test_reference_excludes_the_validity_keys(self, coverage, stored):
        """DATE_FROM and DATE_TO are keys, not fields. Counting them would make
        "three fields held" mean one real attribute.

        Derived from the loaded columns rather than pinned to a number. The
        claim is *which* columns are excluded, and a literal count restated
        that as trivia -- it needed an edit for REGION in BN-128 and again for
        the country columns in BN-133, neither of which had anything to do
        with what this is checking.
        """
        _client, path = stored

        entry = dataset_of(coverage, REFERENCE_DATASET)
        loaded = store.load(path).reference
        columns = set(loaded.data.reset_index().columns)
        keys = {"IDENTIFIER", "DATE_FROM", "DATE_TO"}

        assert entry["field_count"] == len(columns - keys)
        assert keys & columns, "the fixture carries no validity keys to exclude"

    def test_actions_exclude_the_identity_keys(self, coverage):
        """TYPE, VALUE, PAY_DATE, STATUS — not IDENTIFIER and EX_DATE."""
        assert dataset_of(coverage, ACTIONS_DATASET)["field_count"] == 4


class TestCacheSize:
    """Bytes on disk, per dataset and in total."""

    def test_each_dataset_reports_its_own_file(self, coverage):
        """Not the store total. Three rows each showing the total would display
        the same figure three times and make any sum of them wrong."""
        sizes = {entry["dataset"]: entry["cache_size_bytes"]
                 for entry in coverage["datasets"]}

        assert len(set(sizes.values())) == len(sizes), sizes
        assert sizes[MARKET_DATASET] > sizes[REFERENCE_DATASET]

    def test_the_parts_do_not_exceed_the_whole(self, coverage):
        total = sum(entry["cache_size_bytes"] or 0 for entry in coverage["datasets"])

        assert total <= coverage["cache_size_bytes"]

    def test_the_total_includes_more_than_the_datasets(self, coverage):
        """The manifest is part of the store but not a dataset, so the total is
        strictly larger than the sum of the parts."""
        total = sum(entry["cache_size_bytes"] or 0 for entry in coverage["datasets"])

        assert coverage["cache_size_bytes"] > total

    def test_it_matches_the_store_on_disk(self, coverage, stored):
        _, path = stored

        assert coverage["cache_size_bytes"] == store.size_on_disk(path)

    def test_an_unmeasurable_path_is_none(self, tmp_path):
        assert store.size_on_disk(tmp_path / "nowhere") is None
        assert store.dataset_size_on_disk(tmp_path, MARKET_DATASET) is None

    def test_an_unknown_dataset_has_no_file(self, stored):
        _, path = stored

        assert store.dataset_size_on_disk(path, "nonsense") is None


class TestIdentifiersUnion:
    """Assets covered, counted once."""

    def test_it_is_not_the_sum_of_the_datasets(self):
        """Every name is in both market and reference data here, so summing
        would report double the universe."""
        assets = 25
        payload_total = sum(entry["identifiers"]
                            for entry in _coverage_for(assets)["datasets"])

        assert payload_total > assets

    def test_it_counts_each_name_once(self, coverage):
        # The twenty-five companies plus one identifier per FX pair: a
        # currency pair is market data, and appears in the union like
        # anything else the store holds.
        from beacon.synthetic import regions

        assert coverage["identifiers_union"] == 25 + len(regions.pairs())

    def test_a_name_in_only_one_dataset_still_counts(self):
        """The union is over all datasets, so a priced name with no reference
        row is still an asset covered."""
        market = MarketData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": name, "DATE": "2025-01-02", "CLOSE": 100.0}
            for name in ("AAA", "BBB")]))
        reference = ReferenceData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE_FROM": "2020-01-01", "NAME": "Alpha"}]))

        client = TestClient(create_app(ServerConfig(
            auth_token=TOKEN, data_fetcher=DataFetcher(market, reference))))
        payload = client.get("/data/coverage", headers=auth()).json()

        assert payload["identifiers_union"] == 2

    def test_a_server_without_data_covers_nothing(self):
        client = TestClient(create_app(ServerConfig(auth_token=TOKEN)))
        payload = client.get("/data/coverage", headers=auth()).json()

        assert payload["identifiers_union"] == 0
        assert payload["cache_size_bytes"] is None
        assert all(not entry["configured"] for entry in payload["datasets"])


class TestActionsDataset:
    """Corporate actions are reported, not just served."""

    def test_they_appear_as_a_dataset(self, coverage):
        entry = dataset_of(coverage, ACTIONS_DATASET)

        assert entry["configured"] is True
        assert entry["identifiers"] > 0

    def test_they_carry_a_date_span(self, coverage):
        entry = dataset_of(coverage, ACTIONS_DATASET)

        assert entry["start"] < entry["end"]

    def test_an_empty_history_is_reported_as_unconfigured(self):
        """"We hold no actions" is a fact the pane should state rather than a
        dataset it should omit."""
        fetcher = DataFetcher(MarketData.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "DATE": "2025-01-02", "CLOSE": 100.0}])))
        client = TestClient(create_app(
            ServerConfig(auth_token=TOKEN, data_fetcher=fetcher)))

        payload = client.get("/data/coverage", headers=auth()).json()

        assert dataset_of(payload, ACTIONS_DATASET)["configured"] is False

    def test_actions_are_not_offered_for_sync(self):
        """Nothing downloads them, and a sync button that always fails is worse
        than no button."""
        from beacon.server.routers.coverage import DATASETS

        assert ACTIONS_DATASET not in DATASETS


def _coverage_for(assets: int) -> dict:
    """Coverage for a freshly generated in-memory dataset."""
    dataset = generate(SyntheticConfig(assets=assets, start="2023-01-02",
                                       end="2023-06-30", seed=1))
    client = TestClient(create_app(
        ServerConfig(auth_token=TOKEN, data_fetcher=dataset.fetcher())))

    return client.get("/data/coverage", headers=auth()).json()
