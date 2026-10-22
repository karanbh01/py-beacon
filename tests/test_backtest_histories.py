# tests/test_backtest_histories.py
"""BN-338: a name's volume and converted closes are read once per run.

Market impact, the liquidity screen and cap, and execution limits used to read
and convert each name's last few months on every day they asked. The answers
from the cached history must be the ones the windowed reads gave.
"""
import math

import numpy as np
import pandas as pd

from beacon.backtest import histories
from beacon.backtest.costs import ExecutionLimit, MarketImpact
from beacon.backtest.screens import ScreenContext, average_traded
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher

DATES = pd.bdate_range("2023-01-02", "2023-12-29")


def fetcher(gap: bool = True) -> DataFetcher:
    """Two names, one quoted in sterling, with noisy volume and a gap."""
    rng = np.random.default_rng(3)
    rows = []
    for name in ("AAA", "BBB"):
        price = 100.0
        for date in DATES:
            price *= 1.0 + rng.normal(0.0, 0.02)
            missing = (gap and name == "AAA"
                       and pd.Timestamp("2023-06-01") <= date <= pd.Timestamp("2023-06-20"))
            rows.append({"IDENTIFIER": name, "DATE": date, "CLOSE": price,
                         "VOLUME": np.nan if missing else float(rng.integers(1e5, 1e6))})
    rows += [{"IDENTIFIER": "GBPUSD", "DATE": date, "CLOSE": 1.25 + 0.01 * np.sin(i / 9)}
             for i, date in enumerate(DATES)]
    common = {"DATE_FROM": "2020-01-01", "ASSET_CLASS": "Equity", "SECURITY_TYPE": "Common Stock"}
    reference = pd.DataFrame([
        {**common, "IDENTIFIER": "AAA", "NAME": "A", "CURRENCY": "USD", "EXCHANGE": "NYSE"},
        {**common, "IDENTIFIER": "BBB", "NAME": "B", "CURRENCY": "GBP", "EXCHANGE": "LSE"},
        {**common, "IDENTIFIER": "GBPUSD", "NAME": "GBP/USD", "CURRENCY": "USD", "EXCHANGE": "FX",
         "ASSET_CLASS": "Currency", "SECURITY_TYPE": "Spot", "BASE_CURRENCY": "GBP",
         "QUOTE_CURRENCY": "USD"}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def windowed_traded(source: DataFetcher,
                    name: str,
                    date: pd.Timestamp,
                    currency: str,
                    days: int = 63) -> float | None:
    """The read average_traded made before BN-338, written out."""
    start = date - pd.Timedelta(days=int(days * 1.6) + 10)
    frame = source.fetch_market_data(name, start.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
    volume = frame["VOLUME"].dropna().tail(days)

    if volume.empty:
        return None

    prices = source.fetch_prices([name], start.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"),
                                 currency=currency)
    traded = (prices[name].dropna().reindex(volume.index) * volume).dropna()

    return None if traded.empty else float(traded.mean())


class TestTheAnswersAreUnchanged:

    def test_average_traded_value_in_the_books_currency(self):
        source = fetcher()
        context = ScreenContext(source, "USD")

        for name in ("AAA", "BBB"):
            for date in DATES[::7]:
                expected = windowed_traded(source, name, date, "USD")

                assert average_traded(name, date, context) == expected

    def test_through_a_gap_in_the_volume(self):
        source = fetcher()
        date = pd.Timestamp("2023-06-15")

        assert average_traded("AAA", date, ScreenContext(source, "USD")) == \
            windowed_traded(source, "AAA", date, "USD")

    def test_market_impact_reads_the_same_volatility(self):
        source = fetcher()
        context = ScreenContext(source, "USD")
        date = pd.Timestamp("2023-09-29")
        start = date - pd.Timedelta(days=int(63 * 1.6) + 10)
        prices = source.fetch_prices(["BBB"], start.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"),
                                     currency="USD")
        volatility = float(prices["BBB"].dropna().pct_change().dropna().tail(63).std())
        expected = volatility * math.sqrt(1e6 / windowed_traded(source, "BBB", date, "USD"))

        assert MarketImpact().rate("BBB", 1e6, date, context) == expected

    def test_execution_volume_backfills_as_before(self):
        source = fetcher()
        limit = ExecutionLimit(participation=0.1)

        # A day in the gap, beyond five days of backfill, takes the average.
        window = source.fetch_market_data("AAA", "2023-02-15", "2023-06-19")["VOLUME"]

        expected = float(window.dropna().tail(63).mean())

        assert limit.volume("AAA", pd.Timestamp("2023-06-19"), source, 5) == expected


class TestTheCache:

    def test_a_name_is_read_once(self,
                                 monkeypatch):
        source = fetcher()
        reads = []
        original = histories._read
        monkeypatch.setattr(histories, "_read",
                            lambda *args: reads.append(args[1]) or original(*args))
        context = ScreenContext(source, "USD")

        for date in DATES[100:140]:
            average_traded("BBB", date, context)

        assert reads == ["BBB"]

    def test_new_market_data_is_read_afresh(self):
        source = fetcher(gap=False)
        date = DATES[-1]
        before = average_traded("AAA", date, ScreenContext(source, "USD"))

        source.merge_market_data(pd.DataFrame({"IDENTIFIER": ["AAA"], "DATE": [date],
                                               "CLOSE": [1.0], "VOLUME": [1e12]}))

        assert average_traded("AAA", date, ScreenContext(source, "USD")) != before


class TestTheLastBar:
    """BN-340: a carried close comes from the cached closes."""

    def test_matches_the_windowed_read_through_a_gap(self):
        source = fetcher()
        start, date = pd.Timestamp("2023-05-01"), pd.Timestamp("2023-06-15")
        closes = histories.history(source, "AAA", None).closes_between(start, date)
        frame = source.fetch_market_data("AAA", "2023-05-01", "2023-06-15")["CLOSE"].dropna()

        assert (closes.index[-1], closes.iloc[-1]) == (frame.index[-1], frame.iloc[-1])


class TestTheArrays:
    """BN-345: market impact and traded value from arrays, to the last bit."""

    def test_the_volatility_is_the_series_standard_deviation(self):
        from beacon.backtest.costs import _sample_std

        rng = np.random.default_rng(5)

        for _ in range(500):
            values = rng.normal(0.0, 0.02, int(rng.integers(2, 64)))

            assert _sample_std(values) == pd.Series(values).std()

    def test_the_returns_are_pct_change_of_the_window(self):
        source = fetcher()
        start, date = pd.Timestamp("2023-05-01"), pd.Timestamp("2023-09-29")
        held = histories.history(source, "BBB", "USD")
        expected = held.closes_between(start, date).pct_change().dropna().tail(63).to_numpy()

        assert np.array_equal(held.returns(start, date, 63), expected)
