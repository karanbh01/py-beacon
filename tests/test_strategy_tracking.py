# tests/test_strategy_tracking.py
"""BN-269: index tracking through full, optimised and sampled replication.

Eighteen names in three sectors, with history from 2023 so a year of
returns exists when the 2024 backtest starts. Returns are a market factor, a
sector factor and noise, drawn from a fixed seed, and share counts fall away
so the market-cap index is concentrated in the first names. LATE lists in
2024 with too little history to measure.
"""
import numpy as np
import pandas as pd
import pytest

from beacon.backtest import Backtest, uk_oeic
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.fund import Fund
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted
from beacon.index.schedule import sessions
from beacon.optimise.config import OptimisationConfig
from beacon.strategy import (
    FullReplication,
    IndexTracking,
    OptimisedReplication,
    Replication,
    ReplicationStep,
    SampledReplication,
)
from beacon.strategy.tracking import _quotas

DAYS = sessions(pd.Timestamp("2023-01-03"), pd.Timestamp("2024-12-31"), "XNYS")
SECTORS = {"Tech": 6, "Health": 6, "Energy": 6}
NAMES = [f"{sector[:3].upper()}{i}" for sector, count in SECTORS.items()
         for i in range(count)]
SECTOR_OF = {name: sector for sector, count in SECTORS.items()
             for name in NAMES if name.startswith(sector[:3].upper())}
LATE_START = pd.Timestamp("2024-05-01")
START, END = "2024-01-02", "2024-12-31"


def fetcher() -> DataFetcher:
    random = np.random.default_rng(11)
    market = random.normal(0.0003, 0.010, len(DAYS))
    sector_moves = {sector: random.normal(0.0, 0.006, len(DAYS)) for sector in SECTORS}
    rows = []

    for position, name in enumerate([*NAMES, "LATE"]):
        sector = SECTOR_OF.get(name, "Tech")
        returns = (market * (0.8 + 0.05 * position) + sector_moves[sector]
                   + random.normal(0.0, 0.008, len(DAYS)))
        prices = 50.0 * np.cumprod(1.0 + returns)
        shares = 1e9 / (1 + position) ** 1.3

        for day, price in zip(DAYS, prices, strict=True):
            if name == "LATE" and day < LATE_START:
                continue

            rows.append({"IDENTIFIER": name, "DATE": day, "CLOSE": price,
                         "SHARES_OUTSTANDING": shares})

    reference = pd.DataFrame([
        {"IDENTIFIER": name, "NAME": name,
         "DATE_FROM": str(LATE_START.date()) if name == "LATE" else "2020-01-01",
         "CURRENCY": "USD", "EXCHANGE": "XNYS",
         "SECTOR": SECTOR_OF.get(name, "Tech")} for name in [*NAMES, "LATE"]])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


DATA = fetcher()


def definition(universe: list[str] | None = None) -> IndexDefinition:
    return IndexDefinition(index_id="MCW", index_name="Market cap", base_date=START,
                           base_value=1000.0, currency="USD", eligibility_rules=[],
                           weighting_scheme=MarketCapWeighted(),
                           rebalancing_frequency="QUARTERLY", calendar="XNYS",
                           universe_identifiers=universe or NAMES)


def run(strategy, **settings):
    return Backtest(initial_capital=1e7, data_provider=DATA, cache=False,
                    **settings).run(strategy, start=START, end=END)


class LargestNames(Replication):
    """The naive alternative: the heaviest names, rescaled."""

    def __init__(self,
                 holdings: int):
        self.holdings = holdings

    def replicate(self,
                  target,
                  date,
                  context) -> ReplicationStep:
        chosen = sorted(target, key=lambda name: target[name], reverse=True)[:self.holdings]
        total = sum(target[name] for name in chosen)

        return ReplicationStep(date=date, holdings=len(chosen),
                               weights={name: target[name] / total for name in chosen})


class TestFullReplication:

    def test_it_is_the_plain_run(self):
        plain = run(definition())
        tracked = run(IndexTracking(definition()))

        assert tracked.trading_nav.equals(plain.trading_nav)
        assert tracked.replication[0].holdings == len(NAMES)

    def test_an_index_tracking_strategy_cannot_also_be_optimised(self):
        with pytest.raises(ValueError, match="OptimisedReplication"):
            Backtest(initial_capital=1e7, data_provider=DATA).run(
                IndexTracking(definition()), start=START, end=END, optimised=True,
                optimisation_config=OptimisationConfig())


@pytest.fixture(scope="module")
def optimised():
    return run(IndexTracking(definition(), OptimisedReplication(holdings=8)))


@pytest.fixture(scope="module")
def sampled():
    return run(IndexTracking(definition(), SampledReplication(holdings=9)))


class TestOptimisedReplication:

    def test_it_holds_no_more_than_the_limit(self,
                                             optimised):
        assert all(step.holdings <= 8 for step in optimised.replication)
        assert all(sum(step.weights.values()) == pytest.approx(1.0)
                   for step in optimised.replication)

    def test_it_reports_its_ex_ante_tracking_error(self,
                                                   optimised):
        errors = [step.tracking_error for step in optimised.replication]

        assert all(0.0 < error < 0.2 for error in errors)

    def test_it_tracks_better_than_the_largest_names(self,
                                                     optimised):
        naive = run(IndexTracking(definition(), LargestNames(8)))

        assert optimised.get_tracking_error() < naive.get_tracking_error()

    def test_the_run_is_measured_against_the_index(self,
                                                   optimised):
        assert optimised.index.target is not None
        assert optimised.index.optimised is None

    def test_a_name_too_new_to_measure_is_held_at_its_index_weight(self):
        result = run(IndexTracking(definition([*NAMES, "LATE"]),
                                   OptimisedReplication(holdings=8)))
        late = [step for step in result.replication if step.date > LATE_START]

        assert late and all(step.unmeasured == ("LATE",) for step in late)
        assert all(step.holdings <= 8 for step in late)


class TestSampledReplication:

    def test_it_holds_the_limit(self,
                                sampled):
        assert all(step.holdings == 9 for step in sampled.replication)

    def test_each_sector_keeps_its_index_weight(self,
                                                sampled):
        index = run(definition()).index.target.source

        for step in sampled.replication:
            target = index.weight_snapshots[step.date]

            for sector in SECTORS:
                held = sum(w for name, w in step.weights.items() if SECTOR_OF[name] == sector)
                wanted = sum(w for name, w in target.items() if SECTOR_OF[name] == sector)

                assert held == pytest.approx(wanted)

    def test_a_cell_holds_its_largest_names(self,
                                            sampled):
        index = run(definition()).index.target.source
        step = sampled.replication[0]
        target = index.weight_snapshots[step.date]
        heaviest = max(target, key=lambda name: target[name])

        assert heaviest in step.weights

    def test_a_limit_above_the_index_holds_it_in_full(self):
        result = run(IndexTracking(definition(), SampledReplication(holdings=50)))

        assert result.replication[0].holdings == len(NAMES)


class TestQuotas:

    def test_holdings_follow_cell_weight(self):
        quotas = _quotas({"a": 0.6, "b": 0.3, "c": 0.1}, {"a": 10, "b": 10, "c": 10}, 10)

        assert quotas == {"a": 6, "b": 3, "c": 1}

    def test_a_small_limit_keeps_the_heaviest_cells(self):
        assert _quotas({"a": 0.6, "b": 0.3, "c": 0.1},
                       {"a": 5, "b": 5, "c": 5}, 2) == {"a": 1, "b": 1}

    def test_a_cell_holds_no_more_than_it_has(self):
        quotas = _quotas({"a": 0.9, "b": 0.1}, {"a": 2, "b": 10}, 6)

        assert quotas == {"a": 2, "b": 4}


class TestTheStrategyElsewhere:

    def test_a_fund_can_hold_its_index_through_a_replication(self):
        fund = Fund(name="Sampled", vehicle=uk_oeic(),
                    strategy=IndexTracking(definition(), SampledReplication(holdings=6)))
        result = fund.backtest(end=END, start=START, initial_capital=1e7,
                               data_provider=DATA, cache=False)

        assert all(step.holdings == 6 for step in result.replication)

    @pytest.mark.parametrize("build", [
        lambda: OptimisedReplication(holdings=0),
        lambda: OptimisedReplication(lookback_days=50, minimum_observations=100),
        lambda: SampledReplication(holdings=0),
        lambda: SampledReplication(holdings=5, size_buckets=0),
    ])
    def test_an_impossible_setting_is_refused(self,
                                              build):
        with pytest.raises(ValueError):
            build()

    def test_full_replication_reports_no_tracking_error(self):
        step = FullReplication().replicate({"A": 0.5, "B": 0.5}, pd.Timestamp(START),
                                           context=None)

        assert step.tracking_error == 0.0
