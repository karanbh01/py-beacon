# tests/test_backtest_index_calendar.py
"""BN-486: a backtest built by hand steps onto its index's calendar.

The index keeps its definition's calendar on its result, and the engine uses
it when it is given none, so a run built by hand trades and is valued on the
index's sessions, as one built by `Backtest` is. The data has bars on every
weekday; Martin Luther King Jr. Day, 2024-01-15, is no XNYS session.
"""
import dataclasses

import numpy as np
import pandas as pd

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.index.cache import IndexResultCache
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted

DAYS = pd.bdate_range("2024-01-02", "2024-03-28")
HOLIDAY = pd.Timestamp("2024-01-15")
NAMES = ["AAA", "BBB"]


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": 100.0 * (1.001 + 0.001 * i) ** n,
             "SHARES_OUTSTANDING": 1e6}
            for i, name in enumerate(NAMES) for n, day in enumerate(DAYS)]
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name,
                               "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def index(data: DataFetcher):
    definition = IndexDefinition(index_id="CAL", index_name="Calendar", base_date="2024-01-02",
                                 base_value=100.0, currency="USD", eligibility_rules=[],
                                 weighting_scheme=EqualWeighted(),
                                 rebalancing_frequency="MONTHLY", calendar="XNYS",
                                 universe_identifiers=NAMES)

    return IndexCalculator(definition, data).run(end_date="2024-03-28")


def nav_days(result,
             data: DataFetcher,
             **options) -> pd.DatetimeIndex:
    run = BacktestEngine(start_date="2024-01-02", end_date="2024-03-28",
                         initial_capital=1e6, data_provider=data, index_result=result,
                         **options).run()

    return pd.DatetimeIndex(run.portfolio.nav.index)


class TestTheIndexCalendar:

    def test_the_index_result_keeps_its_definition_s(self):
        assert index(fetcher()).calendar == "XNYS"

    def test_a_run_built_by_hand_steps_onto_its_sessions(self):
        data = fetcher()

        days = nav_days(index(data), data)

        assert HOLIDAY not in days
        assert days.equals(nav_days(index(data), data, calendar="XNYS"))

    def test_a_result_with_none_still_steps_on_every_weekday(self):
        data = fetcher()
        bare = dataclasses.replace(index(data), calendar=None)

        assert HOLIDAY in nav_days(bare, data)

    def test_a_cached_result_keeps_it(self,
                                      tmp_path):
        cache = IndexResultCache(tmp_path)
        key = "a" * 64
        cache.put(key, index(fetcher()))

        assert cache.get(key).calendar == "XNYS"
        assert np.isfinite(cache.get(key).index_levels).all()
