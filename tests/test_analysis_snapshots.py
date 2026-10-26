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


class TestCharacteristics:
    """BN-367: market cap, ratios, yield and momentum at each rebalance."""

    SESSIONS = pd.bdate_range("2022-01-03", "2024-06-28")
    ON = pd.Timestamp("2024-06-03")

    def data(self):
        from beacon.data import pair_reference
        from beacon.data.corporate_actions import CorporateActions
        from beacon.data.features import FeatureData

        sessions = self.SESSIONS
        rows = []
        for step, day in enumerate(sessions):
            # GROW climbs 0.1% a session and halves on a 2-for-1 split.
            split = 0.5 if day >= pd.Timestamp("2024-01-02") else 1.0
            rows.append({"IDENTIFIER": "GROW", "DATE": day, "CLOSE": 50.0 * 1.001 ** step * split,
                         "SHARES_OUTSTANDING": 1e6 / split})
            rows.append({"IDENTIFIER": "QUID", "DATE": day, "CLOSE": 400.0,
                         "SHARES_OUTSTANDING": 2e6})
            rows.append({"IDENTIFIER": "GBPUSD", "DATE": day, "CLOSE": 1.25})
            if day <= pd.Timestamp("2023-06-30"):
                rows.append({"IDENTIFIER": "GONE", "DATE": day, "CLOSE": 10.0,
                             "SHARES_OUTSTANDING": 1e6})
        reference = pd.DataFrame([
            {"IDENTIFIER": name, "DATE_FROM": "2020-01-01", "NAME": name, "CURRENCY": currency,
             "EXCHANGE": "XNYS"}
            for name, currency in (("GROW", "USD"), ("QUID", "GBP"), ("GONE", "USD"))]
            + [pair_reference("GBPUSD")])
        actions = pd.DataFrame([
            {"IDENTIFIER": "GROW", "EX_DATE": "2024-01-02", "TYPE": "SPLIT", "VALUE": 2.0},
            {"IDENTIFIER": "QUID", "EX_DATE": "2023-06-03", "TYPE": "DIVIDEND", "VALUE": 99.0},
            {"IDENTIFIER": "QUID", "EX_DATE": "2023-09-01", "TYPE": "DIVIDEND", "VALUE": 6.0},
            {"IDENTIFIER": "QUID", "EX_DATE": "2024-03-01", "TYPE": "DIVIDEND", "VALUE": 10.0},
            {"IDENTIFIER": "QUID", "EX_DATE": "2024-04-01", "TYPE": "SPECIAL_DIVIDEND",
             "VALUE": 50.0}])
        features = pd.DataFrame([
            {"IDENTIFIER": "GROW", "DATE": "2024-05-01", "TYPE": "f", "FIELD": "pe_ratio",
             "VALUE": 21.0},
            {"IDENTIFIER": "GROW", "DATE": "2024-06-04", "TYPE": "f", "FIELD": "pe_ratio",
             "VALUE": 99.0},
            {"IDENTIFIER": "QUID", "DATE": "2021-06-01", "TYPE": "f", "FIELD": "pb_ratio",
             "VALUE": 1.5}])

        return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                           ReferenceData.from_dataframe(reference),
                           CorporateActions.from_dataframe(actions),
                           features=FeatureData.from_dataframe(features))

    @pytest.fixture(scope="class")
    def table(self):
        from beacon.analysis.snapshots import characteristics_snapshots

        frame = characteristics_snapshots(self.data(), ["GROW", "QUID", "GONE"],
                                          [self.ON], "USD")

        return frame.set_index("IDENTIFIER")

    def test_market_cap_is_close_times_shares_in_the_book_s_currency(self,
                                                                    table):
        assert table.loc["QUID", "MARKET_CAP"] == pytest.approx(400.0 * 1.25 * 2e6, rel=1e-12)

    def test_a_ratio_published_after_the_date_is_not_seen(self,
                                                          table):
        assert table.loc["GROW", "PE_RATIO"] == 21.0

    def test_a_ratio_older_than_two_years_is_missing(self,
                                                     table):
        assert pd.isna(table.loc["QUID", "PB_RATIO"])

    def test_the_yield_is_ordinary_dividends_over_the_trailing_year(self,
                                                                    table):
        """The dividend exactly a year old has rolled out; the special is
        not ordinary."""
        assert table.loc["QUID", "DIVIDEND_YIELD"] == pytest.approx(16.0 / 400.0, rel=1e-12)
        assert table.loc["GROW", "DIVIDEND_YIELD"] == 0.0

    def test_momentum_runs_a_year_to_a_month_and_sees_through_a_split(self,
                                                                     table):
        assert table.loc["GROW", "MOMENTUM"] == pytest.approx(1.001 ** (252 - 21) - 1.0,
                                                              rel=1e-9)
        # QUID's price stands still; its return is the cash it paid inside
        # the window, the first dividend falling before it.
        assert table.loc["QUID", "MOMENTUM"] == pytest.approx(
            (1 + 6 / 400) * (1 + 10 / 400) * (1 + 50 / 400) - 1.0, rel=1e-9)

    def test_a_name_that_stopped_trading_has_no_row(self,
                                                    table):
        assert "GONE" not in table.index
