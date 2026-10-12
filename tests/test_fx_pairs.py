# tests/test_fx_pairs.py
"""One rule for what a currency pair is.

A pair is the series reference data calls one: ``ASSET_CLASS`` ``Currency``,
``SECURITY_TYPE`` ``Spot``, with its base and quote currencies. That is what
`fx_pairs` lists and what a conversion reads, from ``CLOSE`` like any other
series. A series named like a pair but without such a row is not a pair, and
market data with a ``RATE`` column is refused.
"""
# BN-226 made one rule of it; BN-308 moved the rule from the RATE column to
# reference data.
import pandas as pd
import pytest

from beacon.data import pair_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.testing import dataset

DAYS = pd.bdate_range("2024-01-02", periods=5)

EQUITY = {"IDENTIFIER": "AAA", "NAME": "AAA", "CURRENCY": "USD",
          "EXCHANGE": "XNYS", "DATE_FROM": "1900-01-01"}


def fetcher_with(pair_columns: dict[str, float],
                 reference_rows: list[dict] | None = None) -> DataFetcher:
    rows = [{"IDENTIFIER": "AAA", "DATE": day, "CLOSE": 10.0} for day in DAYS]
    rows += [{"IDENTIFIER": "GBPUSD", "DATE": day, **pair_columns}
             for day in DAYS]
    reference = (ReferenceData.from_dataframe(pd.DataFrame(reference_rows))
                 if reference_rows is not None else None)

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       reference)


class TestTheCanonicalDataset:

    def test_its_pair_is_listed(self):
        """Its reference data carries a row that makes GBPUSD a pair."""
        assert dataset.data_fetcher().fx_pairs == [dataset.FX_PAIR]

    def test_its_pair_is_not_an_instrument(self):
        fetcher = dataset.data_fetcher()

        assert dataset.FX_PAIR not in fetcher.instrument_identifiers
        assert dataset.FX_PAIR not in fetcher.tradable_identifiers
        assert not fetcher.is_tradable(dataset.FX_PAIR)

    def test_it_still_converts(self):
        rate = dataset.data_fetcher().fx_rate_on("GBP", "USD", "2024-01-03")

        assert rate is not None and rate > 1.0


class TestAPairIsWhatReferenceDataSays:

    def test_a_described_pair_is_read_from_close(self):
        fetcher = fetcher_with({"CLOSE": 1.25},
                               [EQUITY, pair_reference("GBPUSD")])

        assert fetcher.fx_rate_on("GBP", "USD", DAYS[1]) == 1.25
        assert fetcher.fx_pairs == ["GBPUSD"]

    def test_the_name_does_not_matter(self):
        """Found by its currencies, not by spelling them."""
        rows = [{"IDENTIFIER": "CABLE", "DATE": day, "CLOSE": 1.25} for day in DAYS]
        reference = pd.DataFrame([pair_reference("CABLE", base="GBP", quote="USD")])
        fetcher = DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                              ReferenceData.from_dataframe(reference))

        assert fetcher.fx_rate_on("GBP", "USD", DAYS[1]) == 1.25
        assert fetcher.fx_pairs == ["CABLE"]

    def test_a_pair_named_series_without_a_row_is_not_a_pair(self):
        """GBPUSD with no reference row is just a priced series: it is not
        listed and it does not convert pounds."""
        fetcher = fetcher_with({"CLOSE": 1.25}, [EQUITY])

        assert fetcher.fx_pairs == []
        assert fetcher.fetch_fx_rates("GBP", "USD").empty
        assert "GBPUSD" in fetcher.instrument_identifiers

    def test_without_reference_data_no_series_is_a_pair(self):
        fetcher = fetcher_with({"CLOSE": 1.25})

        assert fetcher.fx_pairs == []
        assert fetcher.fetch_fx_rates("GBP", "USD").empty


class TestTheRateColumnIsRetired:

    def test_market_data_with_rate_is_refused(self):
        with pytest.raises(ValueError, match="CLOSE"):
            fetcher_with({"CLOSE": 9.0, "RATE": 1.25},
                         [EQUITY, pair_reference("GBPUSD")])

    def test_merging_rows_with_rate_is_refused(self):
        fetcher = fetcher_with({"CLOSE": 1.25},
                               [EQUITY, pair_reference("GBPUSD")])
        rows = pd.DataFrame([{"IDENTIFIER": "GBPUSD", "DATE": "2024-01-09",
                              "CLOSE": 1.26, "RATE": 1.26}])

        with pytest.raises(ValueError, match="RATE"):
            fetcher.merge_market_data(rows)
