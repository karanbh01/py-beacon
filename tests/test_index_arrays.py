# tests/test_index_arrays.py
"""BN-351: the index's daily valuation on arrays.

An asset's hash is worked out once, and is the one `@dataclass` gives. A
day's prices for a book come as one array, with the values the batch read
gives name by name.
"""
import pickle

import numpy as np
import pandas as pd

from beacon.asset.base import Asset
from beacon.asset.bond import Bond
from beacon.asset.equity import Equity
from beacon.data.base import MarketData
from beacon.data.fetcher import DataFetcher

DAYS = pd.bdate_range("2024-01-02", periods=5)


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": 10.0 + i + offset}
            for i, day in enumerate(DAYS) for offset, name in enumerate(("AAA", "BBB", "CCC"))
            if not (name == "BBB" and day == DAYS[2])]
    rows.append({"IDENTIFIER": "CCC", "DATE": DAYS[1], "CLOSE": np.nan})

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))


def equity() -> Equity:
    return Equity(name="A", currency="USD", ticker="AAA", exchange="XNYS", sector="Tech")


class TestTheAssetHash:

    def test_is_the_generated_one(self):
        held = equity()
        fields = (held.name, held.currency, held.asset_id, held.asset_type, held.ticker,
                  held.exchange, held.isin, held.sector, held.country)

        assert hash(held) == hash(fields)
        assert Equity.__hash__ is Asset.__hash__ and Bond.__hash__ is Asset.__hash__

    def test_is_worked_out_again_after_a_pickle(self):
        held = equity()
        copied = pickle.loads(pickle.dumps(held))

        assert copied == held and hash(copied) == hash(held)
        assert "_hash" not in held.__getstate__()


class TestPriceArrays:

    def test_match_the_batch_read(self):
        source = fetcher()
        names = ["AAA", "BBB", "CCC", "ZZZ"]

        for day in DAYS:
            stamp = f"{day:%Y-%m-%d}"
            batch = source.prices_on(names, stamp)
            array = source.price_array_on(names, stamp)

            assert array is not None
            assert [None if np.isnan(value) else value for value in array] == [
                batch[name] for name in names]

    def test_on_a_day_without_rows(self):
        array = fetcher().price_array_on(["AAA"], "2024-01-13")

        assert array is not None and np.isnan(array).all()

    def test_not_for_a_column_that_is_not_numbers(self):
        frame = pd.DataFrame({"IDENTIFIER": ["AAA"], "DATE": [DAYS[0]], "CLOSE": [1.0],
                              "NOTE": ["x"]})
        source = DataFetcher(MarketData.from_dataframe(frame))

        assert source.price_array_on(["AAA"], f"{DAYS[0]:%Y-%m-%d}", "NOTE") is None
