# tests/test_index_cache.py
"""BN-160: the content-addressed IndexResult cache.

The cache's one safety rule is that it caches only what can be keyed
completely, so most of this file is about the key rather than the storage:
identical inputs must share a fingerprint or the cache never hits, and any
changed input — a rule parameter, the window, the store, the library itself —
must change it or the cache serves stale numbers. There is no invalidation
logic to test, because there is none: a changed input simply never matches.

The results stored here are real calculator output over the canonical dataset,
saved to a disk store first — the data identity half of the key needs a store
path, and an in-memory fetcher is (deliberately) uncacheable.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from beacon.data import store
from beacon.data.fetcher import DataFetcher
from beacon.index import cache as index_cache
from beacon.index.cache import (
    IndexResultCache,
    explain_uncacheable,
    fingerprint,
    key_parts,
)
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EligibilityRuleBase, EqualWeighted, MarketCapRule
from beacon.index.result import IndexResult
from beacon.testing import dataset

# A short window inside the canonical dataset's span: three names, one quarter,
# three monthly rebalances — enough for every panel to be non-trivial, cheap
# enough to run once per module.
UNIVERSE = ["AAA", "BBB", "CCC"]
START = "2024-01-02"
END = "2024-03-28"


def build_definition(min_market_cap: float = 1.0,
                     rules: list[EligibilityRuleBase] | None = None) -> IndexDefinition:
    """The index under test; the rule parameter is the knob the key tests turn."""
    return IndexDefinition(
        index_id="CACHE-IX",
        index_name="Cache Test Index",
        base_date=START,
        base_value=1000.0,
        currency="USD",
        eligibility_rules=(rules if rules is not None
                           else [MarketCapRule(min_market_cap=min_market_cap)]),
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=UNIVERSE)


def saved_fetcher(path: Path) -> DataFetcher:
    """The canonical dataset written to disk and read back, so the fetcher
    carries the store path the data-identity half of the key requires."""
    logging.disable(logging.ERROR)
    try:
        store.save(dataset.data_fetcher(), path)
        return store.load(path)
    finally:
        logging.disable(logging.NOTSET)


def the_key(fetcher: DataFetcher,
            definition: IndexDefinition | None = None,
            start: str = START,
            end: str = END) -> str:
    """A fingerprint that must exist — the tests for None live elsewhere."""
    key = fingerprint(definition or build_definition(), fetcher, start, end)

    assert key is not None
    return key


@pytest.fixture(scope="module")
def disk_fetcher(tmp_path_factory) -> DataFetcher:
    return saved_fetcher(tmp_path_factory.mktemp("data") / "store")


@pytest.fixture(scope="module")
def result(disk_fetcher) -> IndexResult:
    """One real calculation, shared read-only by every round-trip test."""
    logging.disable(logging.ERROR)
    try:
        return IndexCalculator(build_definition(),
                               disk_fetcher).run(start_date=START, end_date=END)
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture
def cache(tmp_path) -> IndexResultCache:
    return IndexResultCache(root=tmp_path / "cache")


class TestFingerprint:
    """Identical inputs share a key; any changed input never matches."""

    def test_identical_inputs_share_a_key(self,
                                          disk_fetcher):
        """Two independently built but identical configurations must key the
        same, or the cache could never hit across sessions."""
        assert the_key(disk_fetcher) == the_key(disk_fetcher)

    def test_the_key_is_a_sha256_hexdigest(self,
                                           disk_fetcher):
        """Content-addressed means the key IS the address: it has to be a
        filesystem-safe fixed-width digest, not the inputs themselves."""
        key = the_key(disk_fetcher)

        assert len(key) == 64
        assert set(key) <= set("0123456789abcdef")

    def test_a_changed_rule_parameter_changes_the_key(self,
                                                      disk_fetcher):
        """The issue's acceptance case: change one rule parameter, miss."""
        changed = build_definition(min_market_cap=2.0)

        assert the_key(disk_fetcher) != the_key(disk_fetcher, changed)

    def test_a_changed_window_changes_the_key(self,
                                              disk_fetcher):
        assert the_key(disk_fetcher) != the_key(disk_fetcher, end="2024-02-29")

    def test_a_different_store_changes_the_key(self,
                                               disk_fetcher,
                                               tmp_path):
        """Byte-identical data in a different directory is a different data
        identity: the path is part of the key, not just the content."""
        other = saved_fetcher(tmp_path / "other-store")

        assert the_key(disk_fetcher) != the_key(other)

    def test_a_regenerated_store_changes_the_key(self,
                                                 tmp_path):
        """The issue's sharpest acceptance case: mutate nothing, regenerate
        the store, miss. The regenerated content is byte-identical, so the
        stamp has to carry the write itself (the manifest's mtime) — the
        conservative direction, since a rewrite can never produce a stale hit,
        at worst a fresh calculation."""
        path = tmp_path / "store"
        before = the_key(saved_fetcher(path))

        regenerated = saved_fetcher(path)

        assert the_key(regenerated) != before

    def test_a_bumped_version_changes_the_key(self,
                                              disk_fetcher,
                                              monkeypatch):
        """BN-153 changed what identical inputs produce, so an upgrade must
        self-invalidate: the library version is the fourth part of the key."""
        before = the_key(disk_fetcher)

        monkeypatch.setattr(index_cache, "__version__", "999.0.0")

        assert the_key(disk_fetcher) != before


class TestUncacheable:
    """Incomplete key means no key — and the reason is available on asking."""

    def test_an_unregistered_rule_is_uncacheable(self,
                                                 disk_fetcher):
        """A rule the catalogue has never heard of cannot contribute its
        parameters to the key, so the whole calculation must refuse to cache
        rather than key on a definition minus one rule."""
        class UnregisteredRule(EligibilityRuleBase):
            def __init__(self) -> None:
                super().__init__(rule_name="Unregistered")

            def is_eligible(self,
                            asset,
                            current_date,
                            market_data_provider,
                            context=None) -> bool:
                return True

        definition = build_definition(rules=[UnregisteredRule()])

        assert fingerprint(definition, disk_fetcher, START, END) is None

        reason = explain_uncacheable(definition, disk_fetcher, START, END)

        assert reason is not None
        assert "UnregisteredRule" in reason
        assert "not registered" in reason

    def test_an_in_memory_fetcher_is_uncacheable(self):
        """No store path, no stable data identity — which is also right:
        tests and mocks should not cache."""
        in_memory = dataset.data_fetcher()

        assert fingerprint(build_definition(), in_memory, START, END) is None

        reason = explain_uncacheable(build_definition(), in_memory, START, END)

        assert reason is not None
        assert "store path" in reason

    def test_a_cacheable_setup_has_nothing_to_explain(self,
                                                      disk_fetcher):
        assert explain_uncacheable(build_definition(),
                                   disk_fetcher, START, END) is None


class TestRoundTrip:
    """A stored result reloads equal on every panel: the second run is a read,
    and it calculates nothing."""

    def cached(self,
               cache: IndexResultCache,
               fetcher: DataFetcher,
               result: IndexResult) -> IndexResult:
        key = the_key(fetcher)
        cache.put(key, result, key_parts(build_definition(), fetcher, START, END))

        reloaded = cache.get(key)

        assert reloaded is not None
        return reloaded

    def test_levels_and_divisor_round_trip_exactly(self,
                                                   cache,
                                                   disk_fetcher,
                                                   result):
        """Exactly, not approximately: floats written as shortest round-trip
        reprs parse back to the same float64, so any tolerance here would only
        hide a serialisation defect."""
        reloaded = self.cached(cache, disk_fetcher, result)

        pd.testing.assert_series_equal(reloaded.index_levels,
                                       result.index_levels,
                                       check_names=False, check_freq=False)
        pd.testing.assert_series_equal(reloaded.divisor_history,
                                       result.divisor_history,
                                       check_names=False, check_freq=False)

    def test_daily_weights_round_trip_in_their_storage_dtypes(self,
                                                              cache,
                                                              disk_fetcher,
                                                              result):
        """BN-153 chose the panel's dtypes for the size of a real run, and a
        reload that quietly widened them would undo that decision downstream."""
        reloaded = self.cached(cache, disk_fetcher, result)
        panel = reloaded.daily_weights

        pd.testing.assert_frame_equal(panel, result.daily_weights)
        assert str(panel.dtypes["DATE"]) == "datetime64[ns]"
        assert str(panel.dtypes["IDENTIFIER"]) == "category"
        assert str(panel.dtypes["AMOUNT"]) == "float32"
        assert str(panel.dtypes["WEIGHT"]) == "float32"

    def test_snapshots_round_trip_with_timestamp_keys(self,
                                                      cache,
                                                      disk_fetcher,
                                                      result):
        """JSON keys are strings, so the reload has to restore pd.Timestamp
        keys or every consumer's date arithmetic breaks on the cached copy."""
        reloaded = self.cached(cache, disk_fetcher, result)

        assert reloaded.index_id == result.index_id
        assert reloaded.constituent_snapshots == result.constituent_snapshots
        assert reloaded.weight_snapshots == result.weight_snapshots
        assert reloaded.announcement_dates == result.announcement_dates
        assert reloaded.cap_reports == result.cap_reports
        assert all(isinstance(date, pd.Timestamp)
                   for date in reloaded.weight_snapshots)

    def test_the_manifest_keeps_the_key_parts_in_the_clear(self,
                                                           cache,
                                                           disk_fetcher,
                                                           result):
        """'Why didn't this hit?' must be answerable by looking at the entry,
        which is the whole reason the manifest exists."""
        parts = key_parts(build_definition(), disk_fetcher, START, END)
        key = the_key(disk_fetcher)
        cache.put(key, result, parts)

        manifest = json.loads(
            (cache.entry_path(key) / "manifest.json").read_text(encoding="utf-8"))

        assert manifest["key"] == key
        assert manifest["key_parts"] == parts
        assert manifest["created_at"] and manifest["last_used"]
        assert set(manifest["sizes"]) == {"levels.csv.gz", "divisor.csv.gz",
                                          "daily_weights.csv.gz", "snapshots.json"}

    def test_an_unknown_key_is_a_miss(self,
                                      cache):
        assert cache.get("0" * 64) is None

    def test_an_invalid_key_is_refused_outright(self,
                                                cache,
                                                result):
        """get() deletes entries it cannot parse, so an arbitrary string must
        never reach the filesystem as an entry path."""
        cache.put("../not-a-fingerprint", result)

        assert cache.get("../not-a-fingerprint") is None
        assert cache.size_on_disk() == 0


class TestCorruption:
    """A corrupt entry is a miss and gets removed — never an error."""

    def stored(self,
               cache: IndexResultCache,
               fetcher: DataFetcher,
               result: IndexResult) -> str:
        key = the_key(fetcher)
        cache.put(key, result)

        return key

    def test_a_truncated_panel_is_a_miss_and_the_entry_goes(self,
                                                            cache,
                                                            disk_fetcher,
                                                            result,
                                                            caplog):
        """Half a gzip stream must read as 'not cached' rather than raise into
        the calculation that asked, and the wreck must not sit there turning
        every future lookup into the same warning."""
        key = self.stored(cache, disk_fetcher, result)
        panel = cache.entry_path(key) / "daily_weights.csv.gz"
        panel.write_bytes(panel.read_bytes()[:20])

        with caplog.at_level(logging.WARNING, logger="beacon.index.cache"):
            assert cache.get(key) is None

        assert not cache.entry_path(key).exists()
        assert any("corrupt" in record.message for record in caplog.records)

    def test_a_missing_file_is_a_miss_and_the_entry_goes(self,
                                                         cache,
                                                         disk_fetcher,
                                                         result):
        key = self.stored(cache, disk_fetcher, result)
        (cache.entry_path(key) / "snapshots.json").unlink()

        assert cache.get(key) is None
        assert not cache.entry_path(key).exists()

    def test_the_slot_is_usable_again_after_the_removal(self,
                                                        cache,
                                                        disk_fetcher,
                                                        result):
        """Removal is what makes the miss honest: the next calculation caches
        into the same key and the entry is whole again."""
        key = self.stored(cache, disk_fetcher, result)
        (cache.entry_path(key) / "levels.csv.gz").unlink()

        assert cache.get(key) is None

        cache.put(key, result)

        assert cache.get(key) is not None


class TestPruning:
    """LRU by last_used, capped by total size, pruned on put."""

    @staticmethod
    def age(cache: IndexResultCache,
            key: str,
            stamp: str) -> None:
        """Backdate an entry by hand: two puts in one test can land on the
        same clock tick, so eviction order is arranged, not raced."""
        manifest_path = cache.entry_path(key) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["last_used"] = stamp
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                 encoding="utf-8")

    def test_the_least_recently_used_entry_is_evicted(self,
                                                      cache,
                                                      result,
                                                      monkeypatch):
        """Under a cap that holds one entry but not two, the second put must
        evict the older entry and keep itself — newest-in is the entry most
        likely to be asked for again."""
        first, second = "1" * 64, "2" * 64
        cache.put(first, result)
        self.age(cache, first, "2000-01-01T00:00:00+00:00")

        # 1.5 entries: room for either entry alone (their manifests differ by
        # a few bytes of timestamp) but never for both.
        monkeypatch.setattr(index_cache, "MAX_CACHE_BYTES",
                            cache.size_on_disk() * 3 // 2)
        cache.put(second, result)

        assert not cache.entry_path(first).exists()
        assert cache.entry_path(second).exists()

    def test_nothing_is_evicted_under_the_cap(self,
                                              cache,
                                              result):
        """The real cap is 512 MB and these entries are kilobytes: an
        unprovoked eviction would make the cache useless in practice."""
        first, second = "1" * 64, "2" * 64
        cache.put(first, result)
        cache.put(second, result)

        assert cache.entry_path(first).exists()
        assert cache.entry_path(second).exists()

    def test_a_hit_refreshes_last_used(self,
                                       cache,
                                       disk_fetcher,
                                       result):
        """The 'recently used' half of LRU: without the refresh, eviction
        order would be creation order and a daily-driven index would age out
        beneath its consumers."""
        key = self.stored_key(cache, disk_fetcher, result)
        aged = "2000-01-01T00:00:00+00:00"
        self.age(cache, key, aged)

        assert cache.get(key) is not None

        manifest = json.loads((cache.entry_path(key) / "manifest.json")
                              .read_text(encoding="utf-8"))

        assert manifest["last_used"] > aged

    @staticmethod
    def stored_key(cache: IndexResultCache,
                   fetcher: DataFetcher,
                   result: IndexResult) -> str:
        key = the_key(fetcher)
        cache.put(key, result)

        return key

    def test_clear_empties_the_cache(self,
                                     cache,
                                     result):
        cache.put("1" * 64, result)
        cache.put("2" * 64, result)

        assert cache.clear() == 2
        assert cache.size_on_disk() == 0
