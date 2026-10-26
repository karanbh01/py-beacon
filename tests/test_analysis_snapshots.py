# tests/test_analysis_snapshots.py
"""BN-366: the grouping fields of a run's names, as they stood at each
rebalance.

Three names: one that moves from Industrials to Technology in March, one that
stays in Energy, and one that only lists in April.
"""
import pandas as pd
import pytest

from beacon.analysis.snapshots import (
    GROUPING_FIELDS,
    IDENTIFIER,
    rebalance_dates,
    reference_snapshots,
)
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher

DAYS = pd.bdate_range("2024-01-02", "2024-06-28")
JANUARY, APRIL, MAY = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-04-01"), pd.Timestamp(
    "2024-05-01")


def fetcher() -> DataFetcher:
    rows = [{"IDENTIFIER": name, "DATE": day, "CLOSE": 10.0}
            for name in ("MOVER", "STAYER", "LATE") for day in DAYS]
    common = {"CURRENCY": "USD", "EXCHANGE": "XNYS", "REGION": "North America",
              "COUNTRY_LISTING": "US"}
    reference = pd.DataFrame([
        {"IDENTIFIER": "MOVER", "DATE_FROM": "2020-01-01", "DATE_TO": "2024-02-29",
         "NAME": "Mover", "SECTOR": "Industrials", **common},
        {"IDENTIFIER": "MOVER", "DATE_FROM": "2024-03-01", "NAME": "Mover",
         "SECTOR": "Technology", **common},
        {"IDENTIFIER": "STAYER", "DATE_FROM": "2020-01-01", "NAME": "Stayer",
         "SECTOR": "Energy", **common},
        {"IDENTIFIER": "LATE", "DATE_FROM": "2024-04-15", "NAME": "Late",
         "SECTOR": "Utilities", **common, "EXCHANGE": "XNAS"}])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def sector(table: pd.DataFrame,
           date: pd.Timestamp,
           name: str) -> str:
    rows = table.loc[[date]]

    return str(rows.loc[rows[IDENTIFIER] == name, "SECTOR"].iloc[0])


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return reference_snapshots(fetcher(), ["MOVER", "STAYER", "LATE"], [JANUARY, APRIL, MAY])


class TestPointInTime:

    def test_a_name_s_sector_is_the_one_in_force_on_each_date(self,
                                                              table):
        assert sector(table, JANUARY, "MOVER") == "Industrials"
        assert sector(table, APRIL, "MOVER") == "Technology"

    def test_a_name_not_yet_listed_has_no_row(self,
                                              table):
        assert "LATE" not in set(table.loc[[APRIL], IDENTIFIER])
        assert sector(table, MAY, "LATE") == "Utilities"

    def test_every_field_the_data_has_is_kept(self,
                                              table):
        assert [column for column in table.columns if column != IDENTIFIER] == [
            field for field in GROUPING_FIELDS
            if field in ("SECTOR", "REGION", "COUNTRY_LISTING", "CURRENCY", "EXCHANGE")]
        assert str(table.loc[[MAY]].set_index(IDENTIFIER).loc["LATE", "EXCHANGE"]) == "XNAS"


class TestShape:

    def test_a_row_per_name_per_date(self,
                                     table):
        assert table.index.value_counts().to_dict() == {JANUARY: 2, APRIL: 2, MAY: 3}

    def test_no_reference_data_gives_an_empty_table(self):
        bare = DataFetcher(MarketData.from_dataframe(pd.DataFrame(
            [{"IDENTIFIER": "X", "DATE": DAYS[0], "CLOSE": 1.0}])))

        assert reference_snapshots(bare, ["X"], [JANUARY]).empty


class TestARunsRebalances:

    def test_the_fund_s_and_the_index_s_rebalances_within_the_run(self):
        from test_strategy_futures import index, run

        result = run(index())
        dates = rebalance_dates(result)
        days = result.portfolio.nav.index

        assert dates == sorted(dates)
        assert {pd.Timestamp(row.date) for row in result.rebalance_pricing} <= set(dates)
        assert all(days[0] <= date <= days[-1] for date in dates)
        assert len(dates) >= 4
