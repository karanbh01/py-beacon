# tests/test_fx_pairs.py
"""BN-226: one rule for what a currency pair is.

A pair is a market identifier with `RATE` populated. That is what `fx_pairs`
lists and what a conversion reads. A pair stored without `RATE` still
converts, from its first data column, but says so, since it is not listed.
"""
import logging

import pandas as pd

from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher
from beacon.testing import dataset

DAYS = pd.bdate_range("2024-01-02", periods=5)


def fetcher_with(pair_columns: dict[str, float]) -> DataFetcher:
    rows = [{"IDENTIFIER": "AAA", "DATE": day, "CLOSE": 10.0} for day in DAYS]
    rows += [{"IDENTIFIER": "GBPUSD", "DATE": day, **pair_columns}
             for day in DAYS]

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))


class TestTheCanonicalDataset:

    def test_its_pair_is_listed(self):
        """It was stored without RATE, so it converted and was not listed."""
        assert dataset.data_fetcher().fx_pairs == [dataset.FX_PAIR]

    def test_its_pair_is_not_an_instrument(self):
        assert (dataset.FX_PAIR
                not in dataset.data_fetcher().instrument_identifiers)

    def test_it_still_converts(self):
        rate = dataset.data_fetcher().fx_rate_on("GBP", "USD", "2024-01-03")

        assert rate is not None and rate > 1.0


class TestAPairWithoutRate:

    def test_it_converts_from_its_first_column_and_warns(self,
                                                         caplog):
        fetcher = fetcher_with({"CLOSE": 1.25})

        with caplog.at_level(logging.WARNING):
            rate = fetcher.fx_rate_on("GBP", "USD", DAYS[1])

        assert rate == 1.25
        assert "GBPUSD has no RATE values" in caplog.text

    def test_it_is_not_listed(self):
        assert fetcher_with({"CLOSE": 1.25}).fx_pairs == []

    def test_a_pair_with_rate_is_read_from_it_without_a_warning(self,
                                                                caplog):
        fetcher = fetcher_with({"CLOSE": 9.0, "RATE": 1.25})

        with caplog.at_level(logging.WARNING):
            rate = fetcher.fx_rate_on("GBP", "USD", DAYS[1])

        assert rate == 1.25
        assert fetcher.fx_pairs == ["GBPUSD"]
        assert "no RATE values" not in caplog.text
