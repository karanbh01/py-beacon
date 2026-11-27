# tests/test_selection_margins.py
"""BN-474: how far each name was from failing each eligibility rule.

The rules are tested on test_index_eligibility's two names, AAA (a cap of
100,000) and ZZZ (1,000); the run is test_strategy_futures' eight names,
caps from about $40bn down to $5bn, behind a $9bn floor, so four pass and
the fourth by little, and two more once prices have risen.
"""
import dataclasses

import pandas as pd
import pytest

from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import columns_of
from beacon.analysis.panels import SELECTION_MARGINS
from beacon.analysis.reports.options import AnalysisOptions, SelectionOptions
from beacon.analysis.reports.statistics import RUN_PREDATES, Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.backtest.breakdown import MARGIN_COLUMNS, selection_margins
from beacon.expressions import data
from beacon.index import IndexDefinition, MarketCapWeighted
from beacon.index.calculation.selection import select_with_provenance
from beacon.index.expression_rules import ExpressionRule
from beacon.index.margins import Measure, margin
from beacon.index.methodology import LiquidityRule, MarketCapRule
from test_index_eligibility import LAST_BAR, build_fetcher, equity, universe

FLOOR = 9e9


class TestTheMargin:

    def test_a_share_of_the_threshold_positive_when_it_passes(self):
        assert margin(6.0, "ge", 5.0) == pytest.approx(0.2)
        assert margin(4.0, "gt", 5.0) == pytest.approx(-0.2)
        assert margin(25.0, "le", 20.0) == pytest.approx(-0.25)
        assert margin(15.0, "lt", 20.0) == pytest.approx(0.25)

    def test_the_difference_against_0_and_none_where_it_does_not_order(self):
        assert margin(0.5, "gt", 0.0) == pytest.approx(0.5)
        assert margin(1.0, "eq", 1.0) is None
        assert margin(None, "ge", 1.0) is None
        assert Measure("x", 6.0, "ge", 5.0, True).margin == pytest.approx(0.2)


class TestEachRuleSaysWhatItMeasured:

    def test_a_market_cap_rule(self):
        rule = MarketCapRule(min_market_cap=50_000, max_market_cap=200_000)
        found = rule.assess(equity("AAA"), LAST_BAR, build_fetcher())

        assert found.eligible and found.eligible == rule.is_eligible(
            equity("AAA"), LAST_BAR, build_fetcher())
        assert [(measure.comparison, measure.threshold) for measure in found.measures] == [
            ("ge", 50_000.0), ("le", 200_000.0)]
        assert found.measures[0].value == pytest.approx(100_000.0)
        assert found.measures[0].margin == pytest.approx(1.0)

    def test_a_name_that_fails(self):
        found = MarketCapRule(min_market_cap=50_000).assess(equity("ZZZ"), LAST_BAR,
                                                             build_fetcher())

        assert not found.eligible
        assert found.measures[0].margin == pytest.approx(1_000 / 50_000 - 1)

    def test_a_liquidity_rule_stops_at_the_minimum_it_failed(self):
        rule = LiquidityRule(min_avg_daily_volume=10_000_000, min_avg_daily_value=1.0,
                             lookback_days=5)
        found = rule.assess(equity("AAA"), LAST_BAR, build_fetcher())

        assert not found.eligible
        assert [measure.quantity for measure in found.measures] == ["avg_daily_volume"]

    def test_an_expression_each_comparison_it_evaluated(self):
        rule = ExpressionRule.from_expression((data.market.close > 50.0)
                                              & (data.market.close < 500.0))
        passed = rule.assess(equity("AAA"), LAST_BAR, build_fetcher())
        # The first comparison fails, so the second is not evaluated.
        failed = rule.assess(equity("ZZZ"), LAST_BAR, build_fetcher())

        assert passed.eligible and [measure.comparison for measure in passed.measures] == [
            "gt", "lt"]
        assert passed.measures[1].margin == pytest.approx((500.0 - 100.0) / 500.0)
        assert not failed.eligible and len(failed.measures) == 1

    def test_a_rule_that_changed_is_eligible_alone_is_asked_that(self):
        class Contrary(MarketCapRule):
            def is_eligible(self,
                            asset,
                            current_date,
                            market_data_provider,
                            context=None) -> bool:
                return asset.asset_id == "ZZZ"

        found = Contrary(min_market_cap=50_000).assess(equity("ZZZ"), LAST_BAR,
                                                        build_fetcher())

        assert found.eligible and found.measures == ()

    def test_selection_keeps_them_by_rule_and_name(self):
        rules = [MarketCapRule(min_market_cap=50_000), MarketCapRule(max_market_cap=1e9)]
        found = select_with_provenance(universe(), rules, LAST_BAR, build_fetcher())

        assert set(found.measures[1]) == {"AAA", "ZZZ"}
        # ZZZ was removed by the first, so the second never tested it.
        assert set(found.measures[2]) == {"AAA"}
        assert found.summary(["AAA", "ZZZ"]).measures == found.measures


@pytest.fixture(scope="module")
def result():
    from test_strategy_futures import NAMES, START, run

    definition = IndexDefinition(index_id="FLOOR", index_name="Over a floor", base_date=START,
                                 base_value=1000.0, currency="USD",
                                 eligibility_rules=[MarketCapRule(min_market_cap=FLOOR)],
                                 weighting_scheme=MarketCapWeighted(),
                                 rebalancing_frequency="QUARTERLY", calendar="XNYS",
                                 universe_identifiers=NAMES)

    return run(definition, transaction_cost_bps=5.0)


class TestTheRunKeepsThem:

    def test_a_row_a_name_a_rebalance(self,
                                      result):
        found = selection_margins(result)

        assert tuple(found.columns) == MARGIN_COLUMNS
        assert len(found) == 8 * len(result.rebalance_steps)
        assert (found["THRESHOLD"] == FLOOR).all() and (found["RULE_NAME"] == "MarketCapRule").all()
        assert found.groupby(level=0)["PASSED"].sum().tolist() == [
            len(step.target) for step in result.rebalance_steps]

    def test_it_is_one_of_the_run_s_tables(self,
                                           result):
        pd.testing.assert_frame_equal(result.run_data.tables[SELECTION_MARGINS],
                                      selection_margins(result))


class TestAName_sSelection:

    def test_each_rebalance_s_margin(self,
                                     result):
        found = result.analytics.instrument.profile("N3").selection

        assert list(found.columns) == ["rule", "rule_name", "measure", "measured",
                                       "comparison", "threshold", "difference", "margin",
                                       "passed"]
        assert found["passed"].all()
        assert found["margin"].tolist() == pytest.approx(
            ((found["measured"] - FLOOR) / FLOOR).tolist())

    def test_the_difference_is_in_the_measure_s_units(self,
                                                      result):
        # BN-485: the value less the threshold, beside the margin.
        found = result.analytics.instrument.profile("N3").selection

        assert found["difference"].tolist() == pytest.approx(
            (found["measured"] - FLOOR).tolist())
        assert (found["difference"] > 0).all()

    def test_the_near_miss_share_comes_with_it(self,
                                               result):
        assert result.analytics.instrument.profile("N3").selection.attrs == {"near_miss": 0.1}

    def test_a_name_that_failed_until_it_passed(self,
                                                result):
        found = result.analytics.instrument.profile("N5").selection

        assert not found["passed"].iloc[0] and found["passed"].iloc[-1]
        assert ((found["margin"] < 0) == ~found["passed"]).all()

    def test_its_labels(self,
                        result):
        columns = {column.name: column
                   for column in columns_of(result.analytics.instrument.profile("N3").selection)}

        assert columns["margin"].unit == "fraction" and columns["threshold"].unit is None
        assert (columns["difference"].label, columns["difference"].unit) == ("Difference", None)
        assert columns["comparison"].labels["ge"] == "At least"

    def test_a_stored_run_s_and_one_kept_before_them(self,
                                                     result,
                                                     tmp_path):
        from beacon.analysis import RunData

        result.run_data.write(tmp_path)
        stored = Analytics(RunData.from_record(tmp_path)).instrument.profile("N3").selection
        older = dataclasses.replace(result.run_data, tables={
            name: table for name, table in result.run_data.tables.items()
            if name != SELECTION_MARGINS})

        pd.testing.assert_frame_equal(stored, result.analytics.instrument.profile("N3").selection,
                                      check_freq=False)

        with pytest.raises(Unavailable) as raised:
            _ = Analytics(older).instrument.profile("N3").selection

        assert raised.value.code == RUN_PREDATES


class TestNearMisses:

    def test_within_the_option_s_margin(self,
                                        result):
        margins = result.analytics.instrument.profile("N3").selection["margin"]
        close = margins.iloc[0] + 1e-6
        options = AnalysisOptions(selection=SelectionOptions(near_miss=close))
        events = Analytics(result.run_data, AnalysisRequest(
            reports={}, statistic_tables={}, options=options)).instrument.profile("N3").events
        missed = events.loc[events["event"] == "near_miss"]

        assert list(missed.index) == list(margins.index[margins < close])
        assert missed["detail"].iloc[0] == "MarketCapRule: market_cap"
        assert missed["amount"].iloc[0] == pytest.approx(margins.iloc[0])

    def test_none_by_default_for_a_wide_pass(self,
                                             result):
        events = result.analytics.instrument.profile("N0").events

        assert "near_miss" not in set(events["event"])
