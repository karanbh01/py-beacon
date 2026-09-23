# tests/test_free_float_backfill.py
"""BN-219: the free float carries forward within a global window.

A float-adjusted index read the free float in three places, and a blank cell
got two opposite answers: the weighting and the market values refused, the
special-dividend divisor skipped the adjustment and took the dividend off at
full size. Free float changes on corporate events and reviews, not daily, so
the last known value carries forward -- for up to `free_float_backfill_days`,
default 90, set once on the fetcher -- and beyond that every path refuses.
"""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from beacon.asset.equity import Equity
from beacon.data import store
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.data.free_float import (
    DEFAULT_FREE_FLOAT_BACKFILL_DAYS,
    carried_forward,
    require_free_float,
)
from beacon.exceptions import CalculationError
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import MarketCapWeighted

NAMES = ["AAA", "BBB", "CCC"]
START = "2023-06-01"
BASE = "2024-01-02"
END = "2024-02-28"


def fetcher_with(floats: dict[str, float],
                 **settings: object) -> DataFetcher:
    """Three names on every weekday; FREE_FLOAT only on the dates given.

    `floats` maps a date to the value every name carries on it. Every other
    cell of the column is blank, which is the case this issue is about.
    """
    rows = [{"IDENTIFIER": name, "DATE": date, "CLOSE": 100.0,
             "SHARES_OUTSTANDING": 1e6,
             "FREE_FLOAT": floats.get(date.strftime("%Y-%m-%d"), np.nan)}
            for name in NAMES
            for date in pd.bdate_range(START, END)]
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"}
                              for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference),
                       **settings)  # type: ignore[arg-type]


def float_adjusted_run(fetcher: DataFetcher) -> None:
    definition = IndexDefinition(index_id="FF",
                                 index_name="Float adjusted",
                                 base_date=BASE,
                                 base_value=1000.0,
                                 currency="USD",
                                 eligibility_rules=[],
                                 weighting_scheme=MarketCapWeighted(
                                     use_free_float=True),
                                 rebalancing_frequency="MONTHLY",
                                 calendar="XNYS",
                                 universe_identifiers=NAMES)

    IndexCalculator(definition, fetcher).run(end_date=END)


class TestTheLookup:

    HISTORY = pd.Series([0.8, 0.6],
                        index=pd.to_datetime(["2024-01-02", "2024-03-01"]))

    def test_a_value_within_the_window_carries(self):
        assert carried_forward(self.HISTORY, pd.Timestamp("2024-02-15"),
                               90) == 0.8

    def test_one_older_than_the_window_does_not(self):
        assert carried_forward(self.HISTORY, pd.Timestamp("2024-02-15"),
                               30) is None

    def test_a_value_printed_later_is_never_used(self):
        """Forward only: the 2024-03-01 value exists, and nothing dated before
        it may see it -- the look-ahead BN-208 closed for prices and FX."""
        assert carried_forward(self.HISTORY, pd.Timestamp("2023-12-29"),
                               10_000) is None

    def test_the_boundary_day_counts(self):
        assert carried_forward(self.HISTORY, pd.Timestamp("2024-01-12"),
                               10) == 0.8


class TestTheFetcher:

    def test_the_default_is_ninety_days(self):
        assert DEFAULT_FREE_FLOAT_BACKFILL_DAYS == 90
        assert fetcher_with({}).free_float_backfill_days == 90

    def test_todays_value_wins(self):
        fetcher = fetcher_with({"2024-01-02": 0.7, "2024-01-03": 0.5})

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-03") == 0.5

    def test_a_blank_day_takes_the_last_known_value(self):
        fetcher = fetcher_with({"2023-12-01": 0.7})

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") == 0.7

    def test_beyond_the_window_there_is_none(self):
        fetcher = fetcher_with({"2023-09-01": 0.7})

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") is None

    def test_the_window_is_the_users_to_set(self):
        fetcher = fetcher_with({"2023-09-01": 0.7},
                               free_float_backfill_days=180)

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") == 0.7

    def test_zero_turns_carrying_off(self):
        """How every read behaved before this setting existed."""
        fetcher = fetcher_with({"2024-01-12": 0.7},
                               free_float_backfill_days=0)

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") is None

    @pytest.mark.parametrize("days", [-1, 1.5, None, True])
    def test_a_window_that_is_not_a_day_count_is_refused(self,
                                                         days):
        with pytest.raises(ValueError, match="free_float_backfill_days"):
            fetcher_with({}, free_float_backfill_days=days)

    def test_a_store_without_the_column_answers_none(self):
        """Carrying needs a history to carry from; an absent column is the
        BN-217 up-front check's business, not a blank to fill."""
        rows = pd.DataFrame([{"IDENTIFIER": "AAA", "DATE": "2024-01-02",
                              "CLOSE": 1.0}])
        fetcher = DataFetcher(MarketData.from_dataframe(rows))

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-02") is None

    def test_a_merge_is_seen(self):
        """The history is cached per name, so a merge has to clear it, as it
        clears the FX series and the session panel."""
        fetcher = fetcher_with({"2023-09-01": 0.7})

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") is None

        fetcher.merge_market_data(pd.DataFrame([{
            "IDENTIFIER": "AAA", "DATE": pd.Timestamp("2024-01-10"),
            "CLOSE": 100.0, "SHARES_OUTSTANDING": 1e6, "FREE_FLOAT": 0.4}]))

        assert fetcher.fetch_free_float_factor("AAA", "2024-01-15") == 0.4


class TestEveryPathGivesTheSameAnswer:

    def test_a_run_over_blank_days_uses_the_last_value(self):
        """Was a refusal on the first blank cell, which here is every day."""
        float_adjusted_run(fetcher_with({"2023-12-01": 0.7}))

    def test_a_run_beyond_the_window_refuses_and_says_how_to_fix_it(self):
        with pytest.raises(CalculationError) as raised:
            float_adjusted_run(fetcher_with({"2023-09-01": 0.7}))

        message = str(raised.value)

        assert "within 90 days" in message
        assert "free_float_backfill_days" in message

    def test_the_dividend_path_refuses_rather_than_skipping(self):
        """It did `if ff is not None: reduction *= ff`, so a missing value took
        the dividend off at full size and nothing said so."""
        definition = MagicMock()
        definition.currency = "USD"
        definition.weighting_scheme.use_free_float = True
        data = MagicMock()
        data.fetch_shares_outstanding.return_value = 1000
        data.fetch_free_float_factor.return_value = None
        data.fx_rate_on.return_value = 1.0
        asset = Equity(name="A", currency="USD", ticker="AAA", exchange="XNYS")
        action = {"type": "SPECIAL_DIVIDEND", "asset": asset, "value": 2.0,
                  "ex_date": "2024-01-15"}

        with pytest.raises(CalculationError, match="needs a free float"):
            IndexCalculator(definition, data).handle_corporate_action(
                action, [asset], 100000.0, 10.0)

    def test_one_refusal_serves_all_three(self):
        """The shared function is the rule; this pins that a double gets it
        too, since it reads only `fetch_free_float_factor`."""
        provider = MagicMock()
        provider.fetch_free_float_factor.return_value = 1.5

        with pytest.raises(CalculationError, match="not between 0 and 1"):
            require_free_float(provider, "AAA", "2024-01-02", "Test")


class TestTheSettingIsPublished:

    def test_the_store_threads_it(self,
                                  tmp_path):
        store.save(fetcher_with({"2023-12-01": 0.7}), tmp_path)

        loaded = store.load(tmp_path, free_float_backfill_days=30)

        assert loaded.free_float_backfill_days == 30

    def test_health_reports_it(self):
        from fastapi.testclient import TestClient

        from beacon.server import ServerConfig, create_app

        config = ServerConfig(auth_token="t",
                              data_fetcher=fetcher_with(
                                  {}, free_float_backfill_days=45))
        body = TestClient(create_app(config)).get(
            "/health", headers={"Authorization": "Bearer t"}).json()

        assert body["free_float_backfill_days"] == 45
