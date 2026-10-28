# tests/test_expression_bulk.py
"""BN-387: a field read for many names at once gives each name's answer.

`numbers_of` reads the canonical dataset's six names (one quoted in pounds)
in one pass; `value_of` reads them one by one. They must agree exactly, on
market columns, market caps converted into a book currency, and features
with their point-in-time rules, and on names the data does not have.
"""
import math

import pandas as pd
import pytest

from beacon.data.features import FeatureData
from beacon.data.fetcher import DataFetcher
from beacon.expressions import data
from beacon.expressions.resolve import numbers_of, value_of
from beacon.testing import dataset

NAMES = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "NOWHERE"]
DAYS = [pd.Timestamp(day) for day in ("2023-01-03", "2023-06-30", "2024-03-15", "2025-12-31",
                                      "2026-01-20")]


def fetcher() -> DataFetcher:
    """The canonical dataset with a feature: published at different times,
    one name's long before it is wanted, one's never."""
    source = dataset.data_fetcher()
    rows = [{"IDENTIFIER": name, "DATE": published, "TYPE": "fundamentals",
             "FIELD": "pb_ratio", "VALUE": value, "DETAIL": ""}
            for name, published, value in (("AAA", "2023-02-01", 1.5), ("BBB", "2023-06-30", 2.0),
                                           ("CCC", "2024-01-10", 0.8), ("DDD", "2020-01-01", 3.0),
                                           ("EEE", "2024-03-16", 1.1))]

    return DataFetcher(source.market, source.reference, source.corporate_actions,
                       FeatureData.from_dataframe(pd.DataFrame(rows)))


SOURCE = fetcher()


def one_by_one(field,
               day: pd.Timestamp,
               currency: str) -> dict[str, float | None]:
    found = {}

    for name in NAMES:
        value = value_of(field, name, day, SOURCE, currency=currency)
        found[name] = None if value is None or not math.isfinite(float(value)) else float(value)

    return found


@pytest.mark.parametrize("day", DAYS)
@pytest.mark.parametrize("currency", ["USD", "GBP"])
@pytest.mark.parametrize("field", [data.market.market_cap, data.market.close,
                                   data.market.shares_outstanding,
                                   data.features.fundamentals.pb_ratio])
def test_all_at_once_is_each_name_s_answer(field,
                                           currency,
                                           day):
    assert numbers_of(field, NAMES, day, SOURCE, currency=currency) == one_by_one(field, day,
                                                                                  currency)


def test_it_covers_every_kind_of_answer():
    """Some names have values and some have none, so the agreement above is
    not two empty answers."""
    caps = numbers_of(data.market.market_cap, NAMES, DAYS[2], SOURCE)
    ratios = numbers_of(data.features.fundamentals.pb_ratio, NAMES, DAYS[2], SOURCE)

    assert caps["NOWHERE"] is None and all(caps[name] for name in NAMES[:6])
    assert ratios == {"AAA": 1.5, "BBB": 2.0, "CCC": 0.8, "DDD": None, "EEE": None,
                      "FFF": None, "NOWHERE": None}
