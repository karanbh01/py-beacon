# tests/test_requirements.py
"""BN-217: every rule and scheme declares the columns it reads, checked up front.

Karan's framing, and it is right: a dataset's columns cannot change during a
run, and the page read each day carries exactly the dataset's columns, so the
question "does this dataset have what the index needs?" can be answered once,
before day one. And the needs are already stated -- a scheme's own inputs are
its declaration, so `use_free_float=False` means FREE_FLOAT is never read and
must never be demanded.

The payoff is the error, not the speed. Without this a missing column failed at
the first read, in terms of one company on one day, and one case did not fail
where the cause was at all:

    no SHARES_OUTSTANDING  ->  "N0 has no positive SHARES_OUTSTANDING on 2024-01-02"
    no VOLUME + liquidity  ->  "index holds nothing on its base date"

The second never mentions volume. Every name was excluded as "not liquid
enough" and a reader's natural move was to loosen the threshold.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from beacon.backtest.engine import BacktestEngine
from beacon.data.base import MarketData, ReferenceData
from beacon.data.fetcher import DataFetcher
from beacon.exceptions import CalculationError
from beacon.expressions import data
from beacon.index import ExpressionRule
from beacon.index.calculation import IndexCalculator
from beacon.index.constructor import IndexDefinition
from beacon.index.feature_rules import FeatureRule
from beacon.index.methodology import (
    EqualWeighted,
    LiquidityRule,
    MarketCapRule,
    MarketCapWeighted,
)
from beacon.index.requirements import required_by
from beacon.testing import index_result_from_weights

NAMES = [f"N{index}" for index in range(10)]
START = "2023-10-02"
BASE = "2024-01-02"
END = "2024-03-28"


def fetcher_with(**columns: float) -> DataFetcher:
    """A store carrying exactly *columns* beside the identifier and date."""
    rows = [{"IDENTIFIER": name, "DATE": date, **columns}
            for name in NAMES
            for date in pd.bdate_range(START, END)]
    reference = pd.DataFrame([{"IDENTIFIER": name, "DATE_FROM": "2020-01-01",
                               "NAME": name, "CURRENCY": "USD",
                               "EXCHANGE": "XNYS"}
                              for name in NAMES])

    return DataFetcher(MarketData.from_dataframe(pd.DataFrame(rows)),
                       ReferenceData.from_dataframe(reference))


def definition(scheme,
               rules=None) -> IndexDefinition:
    return IndexDefinition(index_id="REQ",
                           index_name="Requirements",
                           base_date=BASE,
                           base_value=1000.0,
                           currency="USD",
                           eligibility_rules=rules or [],
                           weighting_scheme=scheme,
                           rebalancing_frequency="MONTHLY",
                           calendar="XNYS",
                           universe_identifiers=NAMES)


class TestEachPartDeclaresWhatItReads:
    """The declarations themselves, derived from each part's inputs."""

    def test_a_plain_market_cap_scheme_does_not_ask_for_free_float(self):
        """Karan's point: the input is the declaration."""
        assert MarketCapWeighted().required_columns() == {
            "CLOSE", "SHARES_OUTSTANDING"}

    def test_a_float_adjusted_one_does(self):
        assert MarketCapWeighted(use_free_float=True).required_columns() == {
            "CLOSE", "SHARES_OUTSTANDING", "FREE_FLOAT"}

    def test_equal_weighting_reads_nothing(self):
        assert EqualWeighted().required_columns() == frozenset()

    def test_a_market_cap_screen_needs_price_and_shares(self):
        assert MarketCapRule(min_market_cap=1e9).required_columns() == {
            "CLOSE", "SHARES_OUTSTANDING"}

    @pytest.mark.parametrize("rule,expected", [
        (LiquidityRule(min_avg_daily_volume=1e5), {"VOLUME"}),
        (LiquidityRule(min_avg_daily_value=1e6), {"CLOSE", "VOLUME"}),
        (LiquidityRule(), set()),
    ])
    def test_a_liquidity_screen_follows_its_thresholds(self,
                                                      rule,
                                                      expected):
        assert rule.required_columns() == expected

    def test_an_expression_derives_its_needs_from_its_tree(self):
        """A derived field is expanded into what it is computed from: a screen
        on `market_cap` needs CLOSE and SHARES_OUTSTANDING, not a column called
        MARKET_CAP that no store has."""
        rule = ExpressionRule.from_expression(
            (data.market.free_float_market_cap > 1e9)
            & (data.market.adv_3m > 1e5))

        assert rule.required_columns() == {
            "CLOSE", "SHARES_OUTSTANDING", "FREE_FLOAT", "VOLUME"}

    def test_a_stored_market_field_names_its_own_column(self):
        rule = ExpressionRule.from_expression(data.market.close > 10)

        assert rule.required_columns() == {"CLOSE"}

    def test_a_reference_field_needs_no_market_column(self):
        """It reads another table, and this check covers market data only."""
        rule = ExpressionRule.from_expression(
            data.reference.sector == "Technology")

        assert rule.required_columns() == frozenset()

    def test_a_feature_rule_needs_no_market_column(self):
        rule = FeatureRule(field="revenue", comparison="gt", threshold=1.0)

        assert rule.required_columns() == frozenset()


class TestTheCheckRunsBeforeAnyWork:

    def test_a_missing_share_count_is_refused_as_the_datasets_problem(self):
        """Was: "N0 has no positive SHARES_OUTSTANDING on 2024-01-02"."""
        calculator = IndexCalculator(definition(MarketCapWeighted()),
                                     fetcher_with(CLOSE=100.0))

        with pytest.raises(CalculationError) as raised:
            calculator.run(end_date=END)

        message = str(raised.value)

        assert "SHARES_OUTSTANDING, for MarketCapWeighted" in message
        assert "whole dataset" in message
        assert "N0" not in message

    def test_a_missing_volume_column_names_volume(self):
        """Was: "index holds nothing on its base date", with no mention of
        volume at all."""
        calculator = IndexCalculator(
            definition(EqualWeighted(), [LiquidityRule(min_avg_daily_volume=1e5)]),
            fetcher_with(CLOSE=100.0))

        with pytest.raises(CalculationError) as raised:
            calculator.run(end_date=END)

        assert "VOLUME, for LiquidityRule" in str(raised.value)

    def test_an_unneeded_optional_column_is_not_demanded(self):
        """A plain market-cap index over a store with no free float runs. The
        scheme said it does not read the column, so nothing asks for it."""
        calculator = IndexCalculator(definition(MarketCapWeighted()),
                                     fetcher_with(CLOSE=100.0,
                                                  SHARES_OUTSTANDING=1e6))

        assert not calculator.run(end_date=END).index_levels.empty

    def test_every_missing_column_is_named_at_once(self):
        """Fixing one and discovering the next is the thing this replaces."""
        calculator = IndexCalculator(
            definition(MarketCapWeighted(use_free_float=True),
                       [LiquidityRule(min_avg_daily_volume=1e5)]),
            fetcher_with(CLOSE=100.0))

        with pytest.raises(CalculationError) as raised:
            calculator.run(end_date=END)

        message = str(raised.value)

        for column in ("SHARES_OUTSTANDING", "FREE_FLOAT", "VOLUME"):
            assert column in message

    def test_it_says_who_needs_each_column(self):
        """"Needs X" says what to load; "for Y" says what to change instead."""
        needs = required_by(definition(MarketCapWeighted(use_free_float=True),
                                       [MarketCapRule(min_market_cap=1.0)]),
                            "CLOSE")

        assert needs["SHARES_OUTSTANDING"] == [
            "MarketCapWeighted (free-float adjusted)", "MarketCapRule"]
        assert needs["CLOSE"][0].startswith("the index calculation")


class TestTheBacktestChecksItsPriceColumn:

    def test_a_missing_price_column_is_refused_before_trading(self):
        fetcher = fetcher_with(VOLUME=1e6)
        schedule = {pd.Timestamp(BASE): {"N0": 1.0}}
        engine = BacktestEngine(start_date=BASE,
                                end_date=END,
                                initial_capital=1e6,
                                data_provider=fetcher,
                                index_result=index_result_from_weights(schedule),
                                calendar="XNYS")

        with pytest.raises(CalculationError, match="CLOSE column"):
            engine.run()


class TestAProviderThatCannotSayIsNotRefused:
    """The provider is an interface; a double need not report its columns."""

    def test_a_mocked_fetcher_skips_the_check(self):
        """A `MagicMock` answers `market_columns` with a Mock, and iterating a
        Mock yields nothing -- read naively as "a dataset with no columns",
        which would refuse every mocked run for a column it was about to
        supply. The check skips instead, and the run fails where it always did
        if it fails at all."""
        calculator = IndexCalculator(definition(MarketCapWeighted()),
                                     MagicMock())

        calculator.require_columns()
