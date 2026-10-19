# tests/test_reference_as_of.py
"""BN-325: reference columns for names on dates, each as it stood then.

Three names, one of which moves from Industrials to Technology on
2024-01-02, and a currency pair, in a dataset with no ASSET_CLASS column for
the equities.
"""
import pandas as pd
import pytest

from beacon.data import pair_reference
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.data.reference_as_of import ReferenceAsOf
from beacon.exceptions import InvalidRuleError, NoDataLoadedError
from beacon.server.holding_reference import chosen_columns, holding_reference

MOVED = "2024-01-02"


def fetcher(columns: tuple[str, ...] = ("SECTOR", "COUNTRY_LISTING")) -> DataFetcher:
    """Reference data with a sector move; prices only to satisfy the store."""
    rows = [
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "CURRENCY": "USD", "EXCHANGE": "XNYS",
         "DATE_FROM": "2020-01-01", "DATE_TO": "2024-01-01", "SECTOR": "Industrials",
         "COUNTRY_LISTING": "US", "COUNTRY_DOMICILE": "US"},
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "CURRENCY": "USD", "EXCHANGE": "XNYS",
         "DATE_FROM": MOVED, "DATE_TO": None, "SECTOR": "Technology",
         "COUNTRY_LISTING": "US", "COUNTRY_DOMICILE": "US"},
        {"IDENTIFIER": "BBB", "NAME": "Beta", "CURRENCY": "USD", "EXCHANGE": "XNYS",
         "DATE_FROM": "2022-06-01", "DATE_TO": None, "SECTOR": "Energy",
         "COUNTRY_LISTING": "GB", "COUNTRY_DOMICILE": "IE"},
    ]
    reference = pd.DataFrame([{key: value for key, value in row.items()
                               if key in ("IDENTIFIER", "NAME", "CURRENCY", "EXCHANGE",
                                          "DATE_FROM", "DATE_TO", *columns)}
                              for row in rows] + [pair_reference("USDEUR")])
    prices = pd.DataFrame([{"IDENTIFIER": name, "DATE": "2024-01-02", "CLOSE": 1.0}
                           for name in ("AAA", "BBB", "USDEUR")])

    return DataFetcher(MarketData.from_dataframe(prices), ReferenceData.from_dataframe(reference))


DATA = fetcher()


class TestAsOf:

    def test_a_name_is_read_as_it_stood_on_the_date(self):
        lookup = ReferenceAsOf(DATA, ["AAA"], ["SECTOR"])

        assert lookup.of("AAA", "2023-06-30") == {"sector": "Industrials"}
        assert lookup.of("AAA", MOVED) == {"sector": "Technology"}

    def test_a_window_s_last_day_is_inside_it(self):
        assert ReferenceAsOf(DATA, ["AAA"], ["SECTOR"]).of("AAA", "2024-01-01") == {
            "sector": "Industrials"}

    def test_before_its_first_row_a_name_has_none(self):
        assert ReferenceAsOf(DATA, ["BBB"], ["SECTOR"]).of("BBB", "2021-01-01") is None

    def test_a_name_with_no_reference_has_none(self):
        assert ReferenceAsOf(DATA, ["ZZZ"], ["SECTOR"]).of("ZZZ", MOVED) is None

    def test_the_asset_class_is_read_as_the_engine_reads_it(self):
        """No ASSET_CLASS column: an equity is Equity, the pair is Currency."""
        lookup = ReferenceAsOf(DATA, ["AAA", "USDEUR"], ["ASSET_CLASS", "TRADABLE"])

        assert lookup.of("AAA", MOVED) == {"asset_class": "Equity", "tradable": True}
        assert lookup.of("USDEUR", MOVED) == {"asset_class": "Currency", "tradable": False}

    def test_a_column_the_dataset_lacks_is_null(self):
        assert ReferenceAsOf(DATA, ["AAA"], ["REGION"]).of("AAA", MOVED) == {"region": None}

    def test_many_names_on_one_date(self):
        assert ReferenceAsOf(DATA, ["AAA", "BBB"], ["NAME"]).map_of(["AAA", "BBB"], MOVED) == {
            "AAA": {"name": "Alpha"}, "BBB": {"name": "Beta"}}


class TestChoosingColumns:

    def test_the_default_set_is_name_asset_class_sector_and_country(self):
        assert chosen_columns(DATA, None) == ["NAME", "ASSET_CLASS", "SECTOR",
                                              "COUNTRY_LISTING"]

    def test_domicile_is_the_country_when_the_dataset_has_it(self):
        assert chosen_columns(fetcher(("SECTOR", "COUNTRY_DOMICILE", "COUNTRY_LISTING")),
                              None)[-1] == "COUNTRY_DOMICILE"

    def test_a_dataset_without_a_sector_leaves_it_out_of_the_default(self):
        assert chosen_columns(fetcher(()), None) == ["NAME", "ASSET_CLASS"]

    def test_a_chosen_set_is_taken_as_given(self):
        assert chosen_columns(DATA, " sector, name ,sector") == ["SECTOR", "NAME"]

    def test_an_empty_choice_is_none(self):
        assert chosen_columns(DATA, "") == []

    def test_a_documented_column_the_dataset_lacks_is_allowed(self):
        assert chosen_columns(DATA, "region,country_domicile") == ["REGION",
                                                                   "COUNTRY_DOMICILE"]

    def test_an_unknown_column_is_refused(self):
        with pytest.raises(InvalidRuleError, match="colour"):
            chosen_columns(DATA, "name,colour")

    def test_without_data_columns_asked_for_are_refused(self):
        with pytest.raises(NoDataLoadedError):
            holding_reference(None, "sector", ["AAA"])

    def test_without_data_the_default_is_none(self):
        assert holding_reference(None, None, ["AAA"]).of("AAA", MOVED) is None
