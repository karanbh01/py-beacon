# tests/test_data_kinds.py
"""BN-308: reference data says what a series is.

Every market series carries its value in CLOSE. A currency pair is the series
reference data calls ``Currency``/``Spot``, found by its two currencies
whatever it is named; a short rate is ``Fixed Income``/``Short Rate``. Both
are references rather than holdings, so neither is offered to a universe.
"""
import logging

import pandas as pd
import pytest

from beacon.data import importing, pair_reference, short_rate_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.data.kinds import classify

DAYS = pd.bdate_range("2024-01-02", "2024-01-12")
EQUITY = {"IDENTIFIER": "AAA", "NAME": "Alpha", "CURRENCY": "USD", "EXCHANGE": "XNYS",
          "DATE_FROM": "2020-01-01"}


def reference(*rows: dict) -> pd.DataFrame:
    frame = pd.DataFrame([EQUITY, *rows])
    frame["DATE_FROM"] = pd.to_datetime(frame["DATE_FROM"])

    return frame.set_index("IDENTIFIER")


def fetcher(rows: list[dict],
            closes: dict[str, float]) -> DataFetcher:
    market = [{"IDENTIFIER": name, "DATE": day, "CLOSE": close}
              for name, close in {"AAA": 100.0, **closes}.items() for day in DAYS]

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(market)),
                       ReferenceData.from_dataframe(pd.DataFrame([EQUITY, *rows])))


class TestPairs:

    def test_a_pair_is_found_by_its_currencies_not_its_name(self):
        data = fetcher([pair_reference("Cable WMR", base="GBP", quote="USD")],
                       {"Cable WMR": 1.27})

        assert data.fx_pairs == ["Cable WMR"]
        assert data.fx_rate_on("GBP", "USD", DAYS[3]) == pytest.approx(1.27)
        assert data.fx_rate_on("USD", "GBP", DAYS[3]) == pytest.approx(1 / 1.27)

    def test_a_series_named_like_a_pair_is_not_one_without_its_row(self):
        data = fetcher([], {"GBPUSD": 1.27})

        assert data.fx_pairs == []
        assert data.fx_rate_on("GBP", "USD", DAYS[3]) is None
        assert "GBPUSD" in data.instrument_identifiers

    def test_a_pair_is_quoted_in_its_quote_currency(self):
        row = {**pair_reference("GBPUSD"), "CURRENCY": "GBP"}

        with pytest.raises(ValueError, match="QUOTE_CURRENCY USD"):
            classify(reference(row))

    def test_a_pair_needs_both_currencies(self):
        row = {**pair_reference("GBPUSD"), "BASE_CURRENCY": None}

        with pytest.raises(ValueError, match="BASE_CURRENCY must be a three-letter"):
            classify(reference(row))

    def test_two_pairs_for_the_same_currencies_are_refused(self):
        with pytest.raises(ValueError, match="Two spot pairs quote GBP in USD"):
            classify(reference(pair_reference("GBPUSD"),
                               pair_reference("GBP WMR", base="GBP", quote="USD")))

    def test_a_currency_row_needs_a_security_type(self):
        row = {**pair_reference("GBPUSD"), "SECURITY_TYPE": None}

        with pytest.raises(ValueError, match="needs a SECURITY_TYPE"):
            classify(reference(row))

    def test_a_reference_row_added_later_makes_a_pair(self):
        data = fetcher([], {"GBPUSD": 1.27})
        assert data.fx_rate_on("GBP", "USD", DAYS[3]) is None

        data.merge_reference_data(pd.DataFrame([pair_reference("GBPUSD")]))

        assert data.fx_rate_on("GBP", "USD", DAYS[3]) == pytest.approx(1.27)


class TestShortRates:

    def test_a_rate_is_read_from_its_close(self):
        data = fetcher([short_rate_reference("SOFR", "usd", tenor="ON")], {"SOFR": 0.053})

        assert data.short_rate("USD").identifier == "SOFR"
        assert data.short_rate_on("USD", DAYS[4]) == pytest.approx(0.053)
        assert data.short_rate_on("USD", DAYS[0] - pd.Timedelta(days=1)) is None
        assert data.short_rate("EUR") is None
        assert len(data.fetch_short_rates("usd")) == len(DAYS)

    def test_one_month_is_preferred_then_overnight(self,
                                                   caplog):
        rows = [short_rate_reference("A 3M", "USD", tenor="3M"),
                short_rate_reference("B ON", "USD", tenor="ON"),
                short_rate_reference("C 1M", "USD", tenor="1m")]

        with caplog.at_level(logging.INFO, logger="beacon.data.kinds"):
            assert classify(reference(*rows)).short_rates["USD"].identifier == "C 1M"

        assert "USD has 3 short rates; reading C 1M" in caplog.text
        assert classify(reference(*rows[:2])).short_rates["USD"].identifier == "B ON"
        assert classify(reference(rows[0])).short_rates["USD"].identifier == "A 3M"

    def test_the_day_count_is_read(self):
        rows = [short_rate_reference("SONIA", "GBP", day_count="act/365"),
                short_rate_reference("SOFR", "USD")]
        rates = classify(reference(*rows)).short_rates

        assert rates["GBP"].day_count == 365.0
        assert rates["USD"].day_count == 360.0

    def test_an_unknown_day_count_is_refused(self):
        with pytest.raises(ValueError, match="DAY_COUNT '30/360'"):
            classify(reference(short_rate_reference("SOFR", "USD", day_count="30/360")))

    def test_a_percent_is_refused(self):
        with pytest.raises(ValueError, match="annual decimals"):
            fetcher([short_rate_reference("SOFR", "USD")], {"SOFR": 5.3})

    def test_a_negative_rate_is_a_rate(self):
        data = fetcher([short_rate_reference("ESTR", "EUR")], {"ESTR": -0.005})

        assert data.short_rate_on("EUR", DAYS[2]) == pytest.approx(-0.005)


class TestTradable:

    def test_pairs_and_short_rates_are_not_tradable(self):
        data = fetcher([pair_reference("GBPUSD"), short_rate_reference("SOFR", "USD")],
                       {"GBPUSD": 1.27, "SOFR": 0.05})

        assert data.instrument_identifiers == ["AAA"]
        assert data.tradable_identifiers == ["AAA"]
        assert not data.is_tradable("SOFR")

    @pytest.mark.parametrize(("value", "tradable"), [("No", False), ("yes", True),
                                                     (False, False), (1, True)])
    def test_the_data_overrides_the_default(self,
                                            value,
                                            tradable):
        kinds = classify(reference({**EQUITY, "IDENTIFIER": "BBB", "TRADABLE": value}))

        assert ("BBB" not in kinds.untradable) is tradable

    def test_a_tradable_value_must_be_a_yes_or_no(self):
        with pytest.raises(ValueError, match="TRADABLE 'maybe'"):
            classify(reference({**EQUITY, "IDENTIFIER": "BBB", "TRADABLE": "maybe"}))

    def test_another_asset_class_is_held_like_an_equity(self):
        kinds = classify(reference({**EQUITY, "IDENTIFIER": "FUND", "ASSET_CLASS": "Fund"}))

        assert "FUND" not in kinds.untradable


class TestTheLegacyColumn:

    def test_a_rate_column_is_refused(self):
        rows = [{"IDENTIFIER": "GBPUSD", "DATE": DAYS[0], "CLOSE": 1.27, "RATE": 1.27}]

        with pytest.raises(ValueError, match=r"RATE column.*CLOSE"):
            DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)))

    def test_a_merge_with_a_rate_column_is_refused(self):
        data = fetcher([], {})

        with pytest.raises(ValueError, match="RATE column"):
            data.merge_market_data(pd.DataFrame([{"IDENTIFIER": "GBPUSD", "DATE": DAYS[0],
                                                  "CLOSE": 1.27, "RATE": 1.27}]))

    def test_a_store_written_with_a_rate_column_is_unreadable(self,
                                                              tmp_path):
        """Reported like any unreadable store, so a server starting on it comes
        up without data rather than not at all."""
        from beacon.data import store
        from beacon.exceptions import StoreFormatError

        path = store.save(fetcher([], {}), tmp_path / "old")
        market = pd.read_csv(path / store.MARKET_FILE)
        market.assign(RATE=float("nan")).to_csv(path / store.MARKET_FILE, index=False)

        assert "before py-beacon 0.7.0" in store.outdated(path)

        with pytest.raises(StoreFormatError, match="RATE column"):
            store.load(path)


class TestImporting:

    MARKET = ("IDENTIFIER,DATE,CLOSE\nAAA,2024-01-02,100\nGBPUSD,2024-01-02,1.27\n"
              "SOFR,2024-01-02,{rate}\n")
    REFERENCE = ("IDENTIFIER,NAME,CURRENCY,EXCHANGE,DATE_FROM,ASSET_CLASS,SECURITY_TYPE,"
                 "BASE_CURRENCY,QUOTE_CURRENCY,DAY_COUNT\n"
                 "AAA,Alpha,USD,{exchange},2020-01-01,Equity,Common Stock,,,\n"
                 "GBPUSD,GBP/USD,USD,,2020-01-01,currency,spot,GBP,USD,\n"
                 "SOFR,SOFR,USD,,2020-01-01,Fixed Income,Short Rate,,,{day_count}\n")

    def load(self,
             folder,
             rate: str = "-0.001",
             exchange: str = "XNYS",
             day_count: str = "act/360"):
        (folder / "market.csv").write_text(self.MARKET.format(rate=rate), encoding="utf-8")
        (folder / "reference.csv").write_text(
            self.REFERENCE.format(exchange=exchange, day_count=day_count), encoding="utf-8")

        return importing.load_files([folder / "market.csv", folder / "reference.csv"])

    def codes(self,
              folder,
              **settings) -> list[tuple]:
        with pytest.raises(importing.DataImportError) as raised:
            self.load(folder, **settings)

        return [(problem.sheet, problem.row, problem.column, problem.code)
                for problem in raised.value.problems]

    def test_every_kind_loads(self,
                              tmp_path):
        data = self.load(tmp_path)

        assert data.fx_pairs == ["GBPUSD"]
        assert data.short_rate_on("USD", "2024-01-02") == pytest.approx(-0.001)
        assert data.tradable_identifiers == ["AAA"]

    def test_an_equity_needs_an_exchange(self,
                                         tmp_path):
        assert self.codes(tmp_path, exchange="") == [("reference", 2, "EXCHANGE",
                                                      "BAD_KIND")]

    def test_a_short_rate_above_one_is_a_percent(self,
                                                 tmp_path):
        assert self.codes(tmp_path, rate="5.3") == [("market", 4, "CLOSE", "OUT_OF_RANGE")]

    def test_a_day_count_must_be_known(self,
                                       tmp_path):
        assert ("reference", 4, "DAY_COUNT", "UNKNOWN_VALUE") in self.codes(
            tmp_path, day_count="30/360")
