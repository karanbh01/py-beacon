# tests/test_class_benchmark.py
"""BN-314: what a share class is measured against, the index in its currency.

Three names priced in dollars through 2024, a dollar-to-euro rate that drifts
up about 10% over the year, and short rates of 5% for the dollar and 3% for
the euro. An unhedged euro class must track the index converted into euros; a
hedged one must track the index hedged the same way, so its tracking
difference leaves out the 2% a year its hedge gives up.
"""
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from beacon.backtest import Backtest, ShareClass, Vehicle
from beacon.backtest.class_benchmark import class_tracking, index_in_class_currency
from beacon.data import pair_reference, short_rate_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.index.constructor import IndexDefinition
from beacon.index.methodology import EqualWeighted
from beacon.index.schedule import sessions

DAYS = sessions(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-12-31"), "XNYS")
NAMES = ["A", "B", "C"]
START, END = "2024-01-02", "2024-12-31"
USD_RATE, EUR_RATE = 0.05, 0.03


def fetcher(rates: bool = True,
            constant_fx: bool = False) -> DataFetcher:
    """Dollar names, a dollar-to-euro rate, and the two short rates."""
    random = np.random.default_rng(29)
    market = random.normal(0.0004, 0.009, len(DAYS))
    rows = []

    for name in NAMES:
        prices = 50.0 * np.cumprod(1.0 + market + random.normal(0.0, 0.004, len(DAYS)))
        rows += [{"IDENTIFIER": name, "DATE": day, "CLOSE": price, "SHARES_OUTSTANDING": 1e8}
                 for day, price in zip(DAYS, prices, strict=True)]

    fx = (np.full(len(DAYS), 0.92) if constant_fx
          else 0.92 * np.cumprod(1.0 + 0.0004 + random.normal(0.0, 0.003, len(DAYS))))
    rows += [{"IDENTIFIER": "USDEUR", "DATE": day, "CLOSE": rate}
             for day, rate in zip(DAYS, fx, strict=True)]
    references = [pair_reference("USDEUR")]

    if rates:
        rows += [{"IDENTIFIER": f"{currency} 1M", "DATE": day, "CLOSE": rate}
                 for currency, rate in (("USD", USD_RATE), ("EUR", EUR_RATE))
                 for day in DAYS]
        references += [short_rate_reference(f"{currency} 1M", currency, tenor="1M")
                       for currency in ("USD", "EUR")]

    reference = pd.DataFrame([*({"IDENTIFIER": name, "NAME": name, "DATE_FROM": "2020-01-01",
                                 "CURRENCY": "USD", "EXCHANGE": "XNYS"} for name in NAMES),
                              *references])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


DATA = fetcher()
DEFINITION = IndexDefinition(index_id="EW", index_name="Equal", base_date=START,
                             base_value=100.0, currency="USD", eligibility_rules=[],
                             weighting_scheme=EqualWeighted(),
                             rebalancing_frequency="QUARTERLY", calendar="XNYS",
                             universe_identifiers=NAMES)


def levels(data: DataFetcher = DATA) -> pd.Series:
    """The index's levels in dollars."""
    return Backtest(initial_capital=1.0, data_provider=data,
                    cache=False).calculate(DEFINITION, start=START, end=END).index_levels


def run(share_class: ShareClass):
    """A fund of one class over the year."""
    return Backtest(initial_capital={share_class.name: 1e6}, share_classes=[share_class],
                    data_provider=DATA, cache=False,
                    vehicle=Vehicle()).run(DEFINITION, start=START, end=END)


LEVELS = levels()


class TestConverting:

    def test_the_index_in_its_own_currency_is_itself(self):
        converted = index_in_class_currency(LEVELS, "USD", "USD", DATA)

        pd.testing.assert_series_equal(converted, LEVELS.astype(float))

    def test_an_unhedged_class_takes_each_day_s_rate(self):
        converted = index_in_class_currency(LEVELS, "USD", "EUR", DATA)
        rates = DATA.fx_rates_on("USD", "EUR", LEVELS.index)

        pd.testing.assert_series_equal(converted, LEVELS * rates, check_names=False)

    def test_a_missing_rate_is_refused(self):
        with pytest.raises(CalculationError, match="USD to JPY"):
            index_in_class_currency(LEVELS, "USD", "JPY", DATA)


class TestHedging:

    def test_each_month_earns_the_index_and_the_carry(self):
        """With the rate fixed, a month's hedge leaves the index's return plus
        the forward's premium, struck at covered interest parity."""
        data = fetcher(constant_fx=True)
        dollars = levels(data)
        hedged = index_in_class_currency(dollars, "USD", "EUR", data, hedge_ratio=1.0)
        firsts = dollars.groupby([dollars.index.year, dollars.index.month]).head(1).index

        for start, end in pairwise(firsts):
            expiry = pd.Timestamp(start + pd.offsets.BMonthBegin(1))
            days = (expiry - start).days
            carry = (1 + EUR_RATE * days / 360) / (1 + USD_RATE * days / 360) - 1
            expected = dollars[end] / dollars[start] + carry

            assert hedged[end] / hedged[start] == pytest.approx(expected, rel=1e-12)

    def test_the_hedge_ratio_scales_the_carry(self):
        data = fetcher(constant_fx=True)
        dollars = levels(data)
        full = index_in_class_currency(dollars, "USD", "EUR", data, hedge_ratio=1.0)
        half = index_in_class_currency(dollars, "USD", "EUR", data, hedge_ratio=0.5)
        unhedged = dollars * 0.92

        # With the rate fixed, the hedge's only effect is the carry.
        assert half.iloc[-1] / half.iloc[0] == pytest.approx(
            (full.iloc[-1] / full.iloc[0] + unhedged.iloc[-1] / unhedged.iloc[0]) / 2,
            rel=1e-3)

    def test_it_takes_the_currency_move_out(self):
        """Hedged, a 10% move in the euro barely reaches the level; converted,
        it all does."""
        hedged = index_in_class_currency(LEVELS, "USD", "EUR", DATA, hedge_ratio=1.0)
        converted = index_in_class_currency(LEVELS, "USD", "EUR", DATA)
        own = LEVELS.iloc[-1] / LEVELS.iloc[0]
        carry = (1 + EUR_RATE) / (1 + USD_RATE)

        assert hedged.iloc[-1] / hedged.iloc[0] == pytest.approx(own * carry, rel=0.01)
        assert abs(converted.iloc[-1] / converted.iloc[0] - own) > 0.05


class TestClassTracking:

    def test_a_hedged_class_tracks_the_hedged_index(self):
        """Against the index in dollars it would give up the 2% carry; against
        the index hedged the same way it tracks closely."""
        result = run(ShareClass("EUR H", currency="EUR", hedged=True))
        book = result.share_classes["EUR H"]
        hedged = index_in_class_currency(LEVELS, "USD", "EUR", DATA, hedge_ratio=1.0)

        against_hedged = class_tracking(book, hedged)
        against_dollars = class_tracking(book, LEVELS)

        assert abs(against_hedged["tracking_difference"]) < 0.002
        assert against_dollars["tracking_difference"] < -0.015
        assert against_hedged["tracking_error"] < against_dollars["tracking_error"]

    def test_an_unhedged_class_tracks_the_converted_index(self):
        result = run(ShareClass("EUR", currency="EUR"))
        book = result.share_classes["EUR"]
        converted = index_in_class_currency(LEVELS, "USD", "EUR", DATA)

        tracking = class_tracking(book, converted)

        assert abs(tracking["tracking_difference"]) < 0.002
        assert tracking["total_return"] == pytest.approx(book.total_return())

    def test_without_a_benchmark_only_the_return_is_given(self):
        book = run(ShareClass("Acc")).share_classes["Acc"]

        tracking = class_tracking(book, pd.Series(dtype=float))

        assert tracking == {"total_return": book.total_return(),
                            "tracking_difference": None, "tracking_error": None}
