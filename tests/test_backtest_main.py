# tests/test_backtest_main.py
"""BN-161: `Backtest` — the one-call front door over calculator, cache, engine.

Three things carry the weight here. First, the front door must be a pure
composition: its result equals the hand-composed calculator + engine result
exactly, because it adds orchestration and never arithmetic. Second, the cache
seam: a cost sweep over one definition performs one index calculation, while
an uncacheable (in-memory) source still *works* and simply calculates every
time. Third, the loud empty-universe error — the silent dead-level wart the
issue records — which belongs to the front door, not the calculator.

Data-source resolution mirrors `tests/test_portfolio_asset_view.py`: a bound
provider wins, the ambient source falls back, and no source anywhere raises
the error naming both fixes.
"""
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

import beacon
from beacon import sources
from beacon.backtest.engine import BacktestEngine
from beacon.backtest.main import Backtest, _rejecting_empty
from beacon.data import store
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import (
    CalculationError,
    DataSourceError,
    MissingDependencyError,
)
from beacon.index import cache as index_cache
from beacon.index.cache import IndexResultCache
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.result import IndexResult
from beacon.testing import dataset

# The same short window inside the canonical dataset that the cache tests use:
# three names, one quarter, three monthly rebalances — every panel non-trivial,
# every run cheap.
UNIVERSE = ["AAA", "BBB", "CCC"]
START = "2024-01-02"
END = "2024-03-28"
CAPITAL = 1_000_000.0


def build_definition(universe: list[str] | None = UNIVERSE) -> IndexDefinition:
    """The index under test; the universe is the knob the error tests turn."""
    return IndexDefinition(
        index_id="BT-MAIN-IX",
        index_name="Backtest Front Door Index",
        base_date=START,
        base_value=1000.0,
        currency="USD",
        eligibility_rules=[],
        weighting_scheme=EqualWeighted(),
        rebalancing_frequency="MONTHLY",
        universe_identifiers=universe)


def saved_fetcher(path: Path) -> DataFetcher:
    """The canonical dataset written to disk and read back, so the fetcher
    carries the store path the cache's data identity requires (the same
    helper `tests/test_index_cache.py` uses)."""
    logging.disable(logging.ERROR)
    try:
        store.save(dataset.data_fetcher(), path)
        return store.load(path)
    finally:
        logging.disable(logging.NOTSET)


@pytest.fixture(autouse=True)
def clean_ambient():
    """Every test starts and ends with no process-level source."""
    sources._reset_for_tests()
    yield
    sources._reset_for_tests()


@pytest.fixture(scope="module")
def disk_fetcher(tmp_path_factory) -> DataFetcher:
    return saved_fetcher(tmp_path_factory.mktemp("data") / "store")


@pytest.fixture
def cache(tmp_path) -> IndexResultCache:
    return IndexResultCache(root=tmp_path / "cache")


@pytest.fixture
def counted_runs(monkeypatch):
    """How many times the calculator actually walked the window."""
    calls = {"count": 0}
    original = IndexCalculator.run

    def counting(self,
                 start_date=None,
                 end_date=None):
        calls["count"] += 1
        return original(self, start_date=start_date, end_date=end_date)

    monkeypatch.setattr(IndexCalculator, "run", counting)
    return calls


def quiet_run(backtest: Backtest,
              definition: IndexDefinition | None = None,
              **kwargs):
    """One run with the pipeline's per-day INFO chatter suppressed."""
    logging.disable(logging.ERROR)
    try:
        return backtest.run(definition if definition is not None
                            else build_definition(),
                            start=kwargs.pop("start", START),
                            end=kwargs.pop("end", END),
                            **kwargs)
    finally:
        logging.disable(logging.NOTSET)


class TestConstruction:
    """The assumptions live on the object, mirroring the engine's defaults."""

    def test_defaults_mirror_the_engine(self):
        bt = Backtest(initial_capital=CAPITAL)

        assert bt.initial_capital == CAPITAL
        assert bt.transaction_cost_bps == 0.0
        assert bt.price_column == "CLOSE"
        assert bt.currency == "USD"
        assert bt.modifiers is None
        assert bt.benchmark is None
        assert bt.data_provider is None

    def test_the_default_cache_is_a_real_one(self):
        """With platformdirs installed the default is the shared machine-wide
        cache — the point of BN-160 is reuse across sessions, not per-object."""
        bt = Backtest(initial_capital=CAPITAL)

        assert isinstance(bt.cache, IndexResultCache)

    def test_an_explicit_cache_is_used_as_given(self,
                                                cache):
        assert Backtest(initial_capital=CAPITAL, cache=cache).cache is cache

    def test_the_end_date_is_required(self):
        """Mirrors the calculator's own contract: a window with no end is a
        caller mistake, not a default to invent."""
        with pytest.raises(ValueError, match="end"):
            Backtest(initial_capital=CAPITAL).run(build_definition(),
                                                  start=START)

    def test_without_platformdirs_caching_degrades_to_off(self,
                                                          monkeypatch):
        """A missing optional package costs the convenience, never the run:
        the front door must work on the core install exactly as on a full
        one."""
        def unavailable():
            raise MissingDependencyError("platformdirs",
                                         "The default index-cache location",
                                         "server")

        monkeypatch.setattr(index_cache, "default_root", unavailable)

        bt = Backtest(initial_capital=CAPITAL,
                      data_provider=dataset.data_fetcher())

        assert bt.cache is None

        result = quiet_run(bt)

        assert not result.trading_nav.empty


class TestResolutionOrder:
    """Bound wins; ambient falls back; the error names both fixes."""

    def test_a_bound_provider_wins_over_the_ambient_one(self):
        """A run's result must read the data the run used, however the
        process source has moved since."""
        bound = dataset.data_fetcher()
        beacon.use(dataset.data_fetcher())

        result = quiet_run(Backtest(initial_capital=CAPITAL,
                                    data_provider=bound))

        assert result._data_fetcher is bound

    def test_the_ambient_source_answers_when_nothing_is_bound(self):
        """Resolution happens per run, not at construction: `beacon.use()`
        after the Backtest is built is still honoured."""
        bt = Backtest(initial_capital=CAPITAL)

        fetcher = dataset.data_fetcher()
        beacon.use(fetcher)

        result = quiet_run(bt)

        assert result._data_fetcher is fetcher

    def test_no_source_anywhere_names_both_fixes(self,
                                                 tmp_path,
                                                 monkeypatch):
        """"No data" deep inside a run is useless unless it says what to do
        about it."""
        monkeypatch.setattr(store, "default_path", lambda: tmp_path / "nothing")

        with pytest.raises(DataSourceError) as raised:
            Backtest(initial_capital=CAPITAL).run(build_definition(),
                                                  start=START, end=END)

        message = str(raised.value)

        assert "beacon.use" in message
        assert "beacon.synthetic" in message


class TestCacheFlows:
    """Miss calculates and stores; hit reuses and rebinds; no key still works."""

    def test_a_second_run_is_a_hit_and_calculates_nothing(self,
                                                          disk_fetcher,
                                                          cache,
                                                          counted_runs):
        bt = Backtest(initial_capital=CAPITAL,
                      data_provider=disk_fetcher,
                      cache=cache)

        first = quiet_run(bt)
        second = quiet_run(bt)

        assert counted_runs["count"] == 1
        pd.testing.assert_series_equal(first.trading_nav, second.trading_nav)
        pd.testing.assert_series_equal(first.index.levels, second.index.levels)

    def test_a_hit_rebinds_the_cached_result_to_the_run_data(self,
                                                             disk_fetcher,
                                                             cache):
        """The cache stores results unbound; the run must rebind them so
        asset-level views read what the simulation read (decision 16)."""
        bt = Backtest(initial_capital=CAPITAL,
                      data_provider=disk_fetcher,
                      cache=cache)
        quiet_run(bt)

        reused = quiet_run(bt)

        assert reused.index.source._data_fetcher is disk_fetcher
        assert reused.index.source.asset("AAA").asset_id == "AAA"

    def test_an_in_memory_fetcher_still_works_and_calculates_every_time(self,
                                                                        cache,
                                                                        counted_runs):
        """Uncacheable means uncached, never broken: no store path, no key,
        and every run pays for its own calculation."""
        bt = Backtest(initial_capital=CAPITAL,
                      data_provider=dataset.data_fetcher(),
                      cache=cache)

        first = quiet_run(bt)
        second = quiet_run(bt)

        assert counted_runs["count"] == 2
        assert cache.size_on_disk() == 0
        pd.testing.assert_series_equal(first.trading_nav, second.trading_nav)

    def test_the_disposition_is_logged_once_per_run(self,
                                                    disk_fetcher,
                                                    cache,
                                                    caplog):
        """One INFO line saying miss or hit — the answer to "why was this
        slow / where did this number come from" without DEBUG spelunking."""
        bt = Backtest(initial_capital=CAPITAL,
                      data_provider=disk_fetcher,
                      cache=cache)

        with caplog.at_level(logging.INFO, logger="beacon.backtest.main"):
            bt.run(build_definition(), start=START, end=END)
            bt.run(build_definition(), start=START, end=END)

        messages = [record.message for record in caplog.records]

        assert sum("cache miss" in message for message in messages) == 1
        assert sum("cache hit" in message for message in messages) == 1

    def test_a_fetcher_the_key_builder_chokes_on_is_just_uncacheable(self):
        """A hand-rolled or mock provider (no real `store_path`) must degrade
        to no caching rather than fail the run — the fund's own tests drive
        the pipeline with exactly such mocks."""
        bt = Backtest(initial_capital=CAPITAL)

        key, parts = bt._cache_key(build_definition(), MagicMock(), START, END)

        assert key is None
        assert parts is None


class TestTheCostSweep:
    """The acceptance case: three costs, one calculation."""

    def test_three_costs_perform_one_calculation(self,
                                                 disk_fetcher,
                                                 cache,
                                                 counted_runs):
        """The assumptions are the model and the definition is the subject:
        sweeping a cost must reprice the trades, not re-walk the index."""
        finals = []
        for cost_bps in (0.0, 5.0, 25.0):
            bt = Backtest(initial_capital=CAPITAL,
                          transaction_cost_bps=cost_bps,
                          data_provider=disk_fetcher,
                          cache=cache)
            result = quiet_run(bt)
            finals.append(float(result.trading_nav.iloc[-1]))

        assert counted_runs["count"] == 1
        # Costs are real money: the free run must end above the costliest.
        assert finals[0] > finals[2]


class TestEquivalence:
    """The front door is composition, not arithmetic: its numbers are the
    hand-built calculator + engine numbers, exactly."""

    def test_run_equals_the_hand_composed_pipeline(self,
                                                   disk_fetcher,
                                                   cache):
        definition = build_definition()

        logging.disable(logging.ERROR)
        try:
            index_result = IndexCalculator(definition, disk_fetcher).run(
                start_date=START, end_date=END)
            by_hand = BacktestEngine(start_date=START,
                                     end_date=END,
                                     initial_capital=CAPITAL,
                                     data_provider=disk_fetcher,
                                     index_result=index_result,
                                     transaction_cost_bps=5.0).run()
        finally:
            logging.disable(logging.NOTSET)

        front_door = quiet_run(Backtest(initial_capital=CAPITAL,
                                        transaction_cost_bps=5.0,
                                        data_provider=disk_fetcher,
                                        cache=cache))

        pd.testing.assert_series_equal(front_door.trading_nav,
                                       by_hand.trading_nav)
        pd.testing.assert_series_equal(front_door.index.levels,
                                       by_hand.index.levels)
        assert (len(front_door.portfolio.transactions)
                == len(by_hand.portfolio.transactions))

    def test_the_benchmark_of_record_lands_on_the_result(self):
        """The constructor's benchmark is a fact about every run this object
        produces, stored as a book on the result."""
        levels = pd.Series([100.0, 101.0, 102.5],
                           index=pd.bdate_range(START, periods=3))

        result = quiet_run(Backtest(initial_capital=CAPITAL,
                                    benchmark=levels,
                                    data_provider=dataset.data_fetcher()))

        assert result.benchmark is not None
        pd.testing.assert_series_equal(result.benchmark.levels, levels)


class TestTheEmptyUniverseError:
    """The silent dead-level wart, made loud where "simulate this" is the
    stated intent."""

    def test_a_definition_without_a_universe_raises(self):
        with pytest.raises(CalculationError, match="universe_identifiers"):
            quiet_run(Backtest(initial_capital=CAPITAL,
                               data_provider=dataset.data_fetcher()),
                      build_definition(universe=None))

    def test_an_unresolvable_universe_raises(self):
        """Identifiers the data source has never heard of resolve to nothing,
        which used to yield a flat level and zero trades with no error."""
        with pytest.raises(CalculationError) as raised:
            quiet_run(Backtest(initial_capital=CAPITAL,
                               data_provider=dataset.data_fetcher()),
                      build_definition(universe=["ZZZ", "YYY"]))

        message = str(raised.value)

        assert "universe_identifiers" in message
        assert "nothing to simulate" in message

    def test_an_empty_level_series_is_rejected_directly(self):
        """The guard itself, on the barest empty result: an empty window is
        as unsimulatable as an empty universe."""
        hollow = IndexResult(index_id="HOLLOW",
                             index_levels=pd.Series(dtype=float),
                             divisor_history=pd.Series(dtype=float),
                             constituent_snapshots={},
                             weight_snapshots={})

        with pytest.raises(CalculationError, match="universe_identifiers"):
            _rejecting_empty(build_definition(), hollow)

    def test_a_healthy_result_passes_through_untouched(self,
                                                       disk_fetcher):
        logging.disable(logging.ERROR)
        try:
            result = IndexCalculator(build_definition(), disk_fetcher).run(
                start_date=START, end_date=END)
        finally:
            logging.disable(logging.NOTSET)

        assert _rejecting_empty(build_definition(), result) is result


class TestOptimisedRuns:
    """`optimise=` tracks the solved weights and books the calculation."""

    def optimised(self,
                  constraints) -> tuple:
        fetcher = dataset.data_fetcher()
        plain = quiet_run(Backtest(initial_capital=CAPITAL,
                                   data_provider=fetcher))
        optimised = quiet_run(Backtest(initial_capital=CAPITAL,
                                       data_provider=fetcher),
                              optimise=constraints)

        return plain, optimised

    def test_the_target_index_book_is_populated(self):
        """The acceptance case: the definition's calculation becomes the
        `target_index` book — the first thing to actually populate decision
        5's book — while the engine tracks the solved schedule."""
        pytest.importorskip("scipy")
        from beacon.optimise import FullInvestment, PositionBounds

        plain, optimised = self.optimised(
            [FullInvestment(),
             PositionBounds(minimum=0.4, maximum=1.0, assets=["AAA"])])

        assert optimised.target_index is not None
        assert optimised.target_index.source is not None
        assert optimised.index is None
        pd.testing.assert_series_equal(optimised.target_index.levels,
                                       plain.index.levels)

    def test_a_binding_constraint_moves_the_money(self):
        """Forcing 40% into one name of an equal-weight three must trade a
        different book than the unconstrained thirds."""
        pytest.importorskip("scipy")
        from beacon.optimise import FullInvestment, PositionBounds

        plain, optimised = self.optimised(
            [FullInvestment(),
             PositionBounds(minimum=0.4, maximum=1.0, assets=["AAA"])])

        assert (float(optimised.trading_nav.iloc[-1])
                != pytest.approx(float(plain.trading_nav.iloc[-1])))
