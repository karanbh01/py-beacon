# tests/test_corporate_actions.py
"""BN-98: corporate-action history, TTM aggregates and point-in-time classification."""
import pandas as pd
import pytest

from beacon.data.base import MarketData, ReferenceData
from beacon.data.corporate_actions import (
    CASH_ACTIONS,
    DIVIDEND,
    RATIO_ACTIONS,
    SPECIAL_DIVIDEND,
    SPLIT,
    CorporateActions,
)
from beacon.data.fetcher import UNCLASSIFIED, DataFetcher
from beacon.exceptions import CalculationError

# A hand-checkable history. Ordinary dividends over 2023 total 1.10; adding the
# special makes 2.10. The December 2022 dividend sits just outside a window
# ending 2023-12-31 and just inside one ending 2023-12-14.
HISTORY = [
    {"IDENTIFIER": "AAA", "EX_DATE": "2022-12-15", "TYPE": "DIVIDEND", "VALUE": 0.20},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-03-15", "TYPE": "DIVIDEND", "VALUE": 0.25},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-05-01", "TYPE": "SPLIT", "VALUE": 2.0},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-06-15", "TYPE": "DIVIDEND", "VALUE": 0.25},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-07-01", "TYPE": "SPECIAL_DIVIDEND",
     "VALUE": 1.00},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-09-15", "TYPE": "DIVIDEND", "VALUE": 0.30},
    {"IDENTIFIER": "AAA", "EX_DATE": "2023-12-15", "TYPE": "DIVIDEND", "VALUE": 0.30},
    {"IDENTIFIER": "BBB", "EX_DATE": "2023-06-01", "TYPE": "DIVIDEND", "VALUE": 0.10},
]

ORDINARY_2023 = 0.25 + 0.25 + 0.30 + 0.30
WITH_SPECIAL_2023 = ORDINARY_2023 + 1.00


@pytest.fixture
def actions():
    return CorporateActions.from_dataframe(pd.DataFrame(HISTORY))


@pytest.fixture
def fetcher(actions):
    market = MarketData.from_dataframe(pd.DataFrame({
        "IDENTIFIER": ["AAA"] * 5,
        "DATE": pd.bdate_range("2023-12-25", periods=5),
        "CLOSE": [98.0, 99.0, 100.0, 101.0, 102.0]}))

    reference = ReferenceData.from_dataframe(pd.DataFrame([
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "SECTOR": "Industrials",
         "DATE_FROM": "2020-01-01", "DATE_TO": "2022-12-31"},
        {"IDENTIFIER": "AAA", "NAME": "Alpha", "SECTOR": "Technology",
         "DATE_FROM": "2023-01-01", "DATE_TO": None},
        {"IDENTIFIER": "BBB", "NAME": "Beta", "SECTOR": "Utilities",
         "DATE_FROM": "2020-01-01", "DATE_TO": None}]))

    return DataFetcher(market, reference, actions)


class TestContainer:

    def test_it_loads_the_history(self,
                                  actions):
        assert set(actions.identifiers) == {"AAA", "BBB"}
        assert not actions.is_empty

    def test_an_empty_store_answers_without_complaint(self):
        """Most universes have no action history; None would push the check out."""
        empty = CorporateActions.empty()

        assert empty.is_empty
        assert empty.trailing_dividend("AAA", "2024-01-01") == 0.0
        assert empty.cumulative_ratio("AAA") == 1.0
        assert empty.get("AAA").empty

    def test_missing_columns_are_refused(self):
        with pytest.raises(CalculationError, match="missing required column"):
            CorporateActions.from_dataframe(
                pd.DataFrame([{"IDENTIFIER": "AAA", "EX_DATE": "2023-01-01"}]))

    def test_an_unknown_action_type_is_refused(self):
        """A typo would otherwise sit in the data being silently ignored."""
        with pytest.raises(CalculationError, match="unknown action type"):
            CorporateActions.from_dataframe(pd.DataFrame([
                {"IDENTIFIER": "AAA", "EX_DATE": "2023-01-01",
                 "TYPE": "DIVIDEDN", "VALUE": 0.25}]))

    def test_types_are_normalised_to_upper_case(self):
        store = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": "2023-01-01",
             "TYPE": "dividend", "VALUE": 0.25}]))

        assert store.get("AAA")["TYPE"].iloc[0] == DIVIDEND

    def test_the_whole_history_is_readable_as_a_copy(self,
                                                     actions):
        """A caller mutating what it reads must not disturb the store."""
        frame = actions.data
        frame.loc[frame.index[0], "VALUE"] = -99.0

        assert actions.data["VALUE"].min() >= 0.0
        assert len(frame) == len(HISTORY)

    def test_history_comes_back_oldest_first(self,
                                             actions):
        dates = actions.get("AAA").index.get_level_values("EX_DATE")

        assert list(dates) == sorted(dates)

    def test_an_unknown_identifier_gives_an_empty_frame(self,
                                                        actions):
        """Not an error: a company with no actions and one we do not hold both
        legitimately have nothing to report."""
        result = actions.get("ZZZ")

        assert result.empty
        assert "VALUE" in result.columns

    def test_a_date_window_filters(self,
                                   actions):
        window = actions.get("AAA", "2023-06-01", "2023-09-30")

        assert len(window) == 3

    def test_a_type_filter_applies(self,
                                   actions):
        assert len(actions.get("AAA", types=[SPLIT])) == 1


class TestTrailingCash:
    """The acceptance criterion: a hand-checkable TTM aggregate."""

    def test_ordinary_dividends_over_a_year(self,
                                            actions):
        assert actions.trailing_dividend("AAA", "2023-12-31") == pytest.approx(
            ORDINARY_2023)

    def test_specials_are_excluded_by_default(self,
                                              actions):
        """A special dividend is by definition not expected to repeat, so it
        does not belong in the figure a forward yield is quoted from."""
        assert actions.trailing_dividend("AAA", "2023-12-31") < WITH_SPECIAL_2023

    def test_specials_can_be_asked_for(self,
                                       actions):
        assert actions.trailing_cash(
            "AAA", "2023-12-31", [DIVIDEND, SPECIAL_DIVIDEND]) == pytest.approx(
                WITH_SPECIAL_2023)

    def test_the_window_rolls(self,
                              actions):
        """Ending mid-December: the December 2023 payment is not in yet, and
        the December 2022 one has not rolled out."""
        assert actions.trailing_dividend("AAA", "2023-12-14") == pytest.approx(
            0.20 + 0.25 + 0.25 + 0.30)

    def test_the_window_is_half_open_at_the_anniversary(self,
                                                        actions):
        """An action exactly a year old has rolled out.

        Without this a dividend paid on the anniversary lands in two
        consecutive years' figures, which quietly overstates one of them.
        """
        on_anniversary = actions.trailing_dividend("AAA", "2023-12-15")

        assert on_anniversary == pytest.approx(ORDINARY_2023)

    def test_an_action_dated_today_is_included(self,
                                               actions):
        assert actions.trailing_dividend("AAA", "2023-03-15") == pytest.approx(
            0.20 + 0.25)

    def test_nothing_in_the_window_is_zero(self,
                                           actions):
        assert actions.trailing_dividend("AAA", "2021-01-01") == 0.0

    def test_a_ratio_type_cannot_be_added_to_a_cash_total(self,
                                                          actions):
        """The mistake this module is arranged to prevent."""
        with pytest.raises(CalculationError, match="ratio, not an amount"):
            actions.trailing_cash("AAA", "2023-12-31", [SPLIT])

    def test_the_two_action_families_do_not_overlap(self):
        assert not CASH_ACTIONS & RATIO_ACTIONS


class TestDividendYield:

    def test_it_divides_by_the_price(self,
                                     actions):
        assert actions.trailing_dividend_yield(
            "AAA", "2023-12-31", 100.0) == pytest.approx(ORDINARY_2023 / 100.0)

    def test_a_non_positive_price_is_refused(self,
                                             actions):
        """A yield on a zero price is undefined, not large."""
        with pytest.raises(CalculationError, match="positive price"):
            actions.trailing_dividend_yield("AAA", "2023-12-31", 0.0)


class TestCumulativeRatio:

    def test_a_single_split_gives_its_ratio(self,
                                            actions):
        assert actions.cumulative_ratio("AAA", "2023-01-01", "2023-12-31") == 2.0

    def test_splits_compound(self):
        """Two 2-for-1 splits make a factor of four, not four in any sum."""
        store = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": "2023-01-01", "TYPE": SPLIT,
             "VALUE": 2.0},
            {"IDENTIFIER": "AAA", "EX_DATE": "2023-06-01", "TYPE": SPLIT,
             "VALUE": 2.0}]))

        assert store.cumulative_ratio("AAA") == 4.0

    def test_no_splits_is_the_identity(self,
                                       actions):
        assert actions.cumulative_ratio("BBB") == 1.0

    def test_a_non_positive_ratio_is_refused(self):
        """A zero multiplier would erase a share count."""
        store = CorporateActions.from_dataframe(pd.DataFrame([
            {"IDENTIFIER": "AAA", "EX_DATE": "2023-01-01", "TYPE": SPLIT,
             "VALUE": 0.0}]))

        with pytest.raises(CalculationError, match="erase or invert"):
            store.cumulative_ratio("AAA")


class TestCalculatorHandoff:

    def test_records_match_the_shape_the_calculator_takes(self,
                                                          actions):
        """IndexCalculator wants type/asset/value/ex_date; the mapping lives
        in one place rather than at every call site."""
        records = actions.as_records("AAA", "2023-05-01")

        assert len(records) == 1
        assert set(records[0]) == {"type", "asset", "value", "ex_date"}
        assert records[0]["type"] == SPLIT
        assert records[0]["asset"] == "AAA"

    def test_a_quiet_date_gives_nothing(self,
                                        actions):
        assert actions.as_records("AAA", "2023-05-02") == []


class TestFetcherAccessors:

    def test_actions_come_through_the_fetcher(self,
                                              fetcher):
        assert len(fetcher.fetch_corporate_actions("AAA")) == 7

    def test_the_trailing_dividend_comes_through(self,
                                                 fetcher):
        assert fetcher.fetch_trailing_dividend("AAA", "2023-12-31") == pytest.approx(
            ORDINARY_2023)

    def test_the_yield_prices_itself_off_the_market_data(self,
                                                          fetcher):
        """The close on 2023-12-29 is 102.0."""
        assert fetcher.fetch_trailing_dividend_yield(
            "AAA", "2023-12-31") == pytest.approx(ORDINARY_2023 / 102.0)

    def test_an_explicit_price_overrides(self,
                                         fetcher):
        assert fetcher.fetch_trailing_dividend_yield(
            "AAA", "2023-12-31", price=50.0) == pytest.approx(ORDINARY_2023 / 50.0)

    def test_no_price_gives_no_yield(self,
                                     fetcher):
        """A missing price is a reason to say nothing rather than to guess."""
        assert fetcher.fetch_trailing_dividend_yield("BBB", "2023-12-31") is None

    def test_a_fetcher_without_actions_still_answers(self):
        market = MarketData.from_dataframe(pd.DataFrame({
            "IDENTIFIER": ["AAA"], "DATE": ["2024-01-01"], "CLOSE": [100.0]}))
        bare = DataFetcher(market)

        assert bare.corporate_actions.is_empty
        assert bare.fetch_trailing_dividend("AAA", "2024-01-01") == 0.0


class TestPointInTimeClassification:
    """The other acceptance criterion: lookups respect as-of dates."""

    def test_it_returns_the_classification_in_force_then(self,
                                                          fetcher):
        """Attributing a 2021 return to a sector joined in 2023 is a real way
        to get a breakdown wrong."""
        assert fetcher.fetch_classification("AAA", "2021-06-01") == "Industrials"
        assert fetcher.fetch_classification("AAA", "2024-06-01") == "Technology"

    def test_the_boundary_dates_belong_to_the_right_record(self,
                                                           fetcher):
        assert fetcher.fetch_classification("AAA", "2022-12-31") == "Industrials"
        assert fetcher.fetch_classification("AAA", "2023-01-01") == "Technology"

    def test_no_date_takes_the_open_ended_record(self,
                                                 fetcher):
        assert fetcher.fetch_classification("AAA") == "Technology"

    def test_a_date_before_any_record_is_unknown(self,
                                                 fetcher):
        assert fetcher.fetch_classification("AAA", "2019-01-01") is None

    def test_an_unknown_instrument_is_unknown(self,
                                              fetcher):
        assert fetcher.fetch_classification("ZZZ") is None

    def test_an_unknown_scheme_is_unknown(self,
                                          fetcher):
        assert fetcher.fetch_classification("AAA", scheme="GICS_SUB_INDUSTRY") is None

    def test_a_fetcher_without_reference_data_is_unknown(self):
        market = MarketData.from_dataframe(pd.DataFrame({
            "IDENTIFIER": ["AAA"], "DATE": ["2024-01-01"], "CLOSE": [100.0]}))

        assert DataFetcher(market).fetch_classification("AAA") is None

    def test_bulk_lookup_covers_every_identifier(self,
                                                 fetcher):
        result = fetcher.fetch_classifications(["AAA", "BBB", "ZZZ"], "2024-06-01")

        assert result == {"AAA": "Technology", "BBB": "Utilities", "ZZZ": None}

    def test_grouping_is_ready_for_group_bounds(self,
                                                fetcher):
        groups = fetcher.group_by_classification(["AAA", "BBB"], "2024-06-01")

        assert groups == {"Technology": ["AAA"], "Utilities": ["BBB"]}

    def test_grouping_keeps_unclassified_names_visible(self,
                                                       fetcher):
        """A name missing from every bucket is how a constraint set quietly
        stops covering part of the universe."""
        groups = fetcher.group_by_classification(["AAA", "ZZZ"], "2024-06-01")

        assert groups[UNCLASSIFIED] == ["ZZZ"]

    def test_grouping_follows_the_as_of_date(self,
                                             fetcher):
        early = fetcher.group_by_classification(["AAA"], "2021-06-01")
        late = fetcher.group_by_classification(["AAA"], "2024-06-01")

        assert early == {"Industrials": ["AAA"]}
        assert late == {"Technology": ["AAA"]}
