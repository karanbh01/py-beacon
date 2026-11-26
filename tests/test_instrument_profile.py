# tests/test_instrument_profile.py
"""BN-468: a name's profile over the run, `instrument.profile(name)`: how
each rebalance weighted it and how its weight drifted each day; BN-469: what
it contributed each period.

The run is test_strategy_futures' eight-name market-cap index held through
2024 with each name capped at 30% and positions under 5.5% after the cap
dropped, so N0 is capped and N7 removed at the first rebalance; its total-return version is
the benchmark.
"""
import dataclasses

import pandas as pd
import pytest

from beacon.analysis import RunData
from beacon.analysis.analytics import Analytics
from beacon.analysis.analytics.profile import ACTIVE, LABELS, Profile, columns_of
from beacon.analysis.panels import FUND_WEIGHTS, INDEX_WEIGHTS, REBALANCE_WEIGHTS
from beacon.analysis.reports.options import AnalysisOptions, AttributionOptions
from beacon.analysis.reports.statistics import RUN_PREDATES, Unavailable
from beacon.analysis.request import AnalysisRequest
from beacon.backtest import Implementation
from beacon.backtest.breakdown import REBALANCE_COLUMNS, rebalance_weights
from beacon.backtest.capacity import MinimumPosition, WeightCap

CAP, SMALLEST = 0.3, 0.055


@pytest.fixture(scope="module")
def result():
    from beacon.index.calculation import IndexCalculator
    from test_strategy_futures import DATA, END, START, index, run

    benchmark = IndexCalculator(index("TOTAL_RETURN"), DATA).run(start_date=START, end_date=END)
    implementation = Implementation(caps=[WeightCap(CAP)],
                                    minimum_position=MinimumPosition(weight=SMALLEST))

    return run(index(), implementation=implementation, benchmark=benchmark,
               transaction_cost_bps=5.0)


@pytest.fixture(scope="module")
def tables(result):
    return result.analytics._context.run.tables


def contribution(analytics,
                 start=None,
                 end=None):
    """N0's contribution over the window, from the per-name table."""
    table = analytics.instrument.table(columns=["contribution"], start=start, end=end)

    return table.loc["N0", "contribution"]


def active(analytics,
           start=None,
           end=None):
    """N0's active contribution over the window, from the run's attribution."""
    found = analytics.portfolio.attribution.active_attribution(start=start, end=end)

    return found.loc[found["level"] == "name"].set_index("asset_id").loc["N0", "contribution"]


class TestTheRunKeepsItsRebalances:

    def test_a_row_for_each_name_each_stage_saw(self,
                                                result):
        found = rebalance_weights(result)
        first = result.rebalance_steps[0]
        mine = found.loc[found.index == first.date].set_index("IDENTIFIER")

        assert tuple(found.columns) == REBALANCE_COLUMNS
        assert set(mine.index) == set(first.target) | set(first.removed)
        assert mine.loc["N0", "CAPPED"] == pytest.approx(CAP)
        assert mine.loc["N7", "REMOVED"] == first.removed["N7"]
        assert mine.loc["N7", "WEIGHT"] == 0.0

    def test_it_is_one_of_the_run_s_tables(self,
                                           result,
                                           tables):
        pd.testing.assert_frame_equal(tables[REBALANCE_WEIGHTS], rebalance_weights(result))


class TestTheRebalances:

    @pytest.fixture(scope="class")
    def capped(self,
               result):
        return result.analytics.instrument.profile("N0").rebalance_weights

    def test_each_stage_s_weight(self,
                                 result,
                                 capped):
        steps = result.rebalance_steps

        assert list(capped.index) == [step.date for step in steps]
        assert capped["target"].tolist() == pytest.approx([step.target["N0"] for step in steps])
        assert capped["screened"].tolist() == pytest.approx([step.screened["N0"]
                                                             for step in steps])
        assert capped["weight"].tolist() == pytest.approx([CAP] * len(steps))
        assert capped["capped"].tolist() == pytest.approx([CAP] * len(steps))
        assert capped["removed"].isna().all()

    def test_each_comparator_s_weight_that_day(self,
                                               tables,
                                               capped):
        index = tables[INDEX_WEIGHTS]["N0"]

        assert list(capped.columns[-2:]) == ["index", "benchmark"]
        assert capped["index"].tolist() == pytest.approx(index.reindex(capped.index).tolist())

    def test_a_removed_name(self,
                            result):
        removed = result.analytics.instrument.profile("N7").rebalance_weights
        first = result.rebalance_steps[0]

        assert removed.loc[first.date, "removed"] == first.removed["N7"]
        assert removed.loc[first.date, "weight"] == 0.0
        assert removed.loc[first.date, "target"] == pytest.approx(first.target["N7"])

    def test_over_a_window(self,
                           result):
        found = result.analytics.instrument.profile("N0", start="2024-03-01").rebalance_weights

        assert found.index[0] == pd.Timestamp("2024-04-01")


class TestTheDays:

    def test_the_fund_s_drifted_weight_and_its_active_weight(self,
                                                              result,
                                                              tables):
        found = result.analytics.instrument.profile("N0").daily_weights
        fund = tables[FUND_WEIGHTS]["N0"].fillna(0.0)

        assert list(found.columns) == ["fund", "index", "benchmark", ACTIVE]
        pd.testing.assert_series_equal(found["fund"], fund.reindex(found.index),
                                       check_names=False)
        pd.testing.assert_series_equal(found[ACTIVE], found["fund"] - found["index"],
                                       check_names=False)
        assert found.attrs == {"against": "index"}
        # The cap holds the fund under the index each day.
        assert (found[ACTIVE] < 0).all()

    def test_against_the_benchmark(self,
                                   result):
        found = result.analytics.instrument.profile("N0", against="benchmark").daily_weights

        assert found.attrs == {"against": "benchmark"}
        pd.testing.assert_series_equal(found[ACTIVE], found["fund"] - found["benchmark"],
                                       check_names=False)

    def test_a_day_a_portfolio_did_not_hold_it_is_0(self,
                                                    result):
        found = result.analytics.instrument.profile("N7").daily_weights
        first, second = (step.date for step in result.rebalance_steps[:2])
        out = found.loc[(found.index >= first) & (found.index < second)]

        # Removed at the first rebalance, held by the index throughout.
        assert (out["fund"] == 0.0).all() and (out["index"] > 0.0).all()


class TestItsParts:

    def test_each_column_s_label_and_unit(self,
                                          result):
        profile = result.analytics.instrument.profile("N0")
        columns = {column.name: column for column in columns_of(profile.rebalance_weights)}

        assert columns["target"].unit == "fraction" and columns["removed"].unit is None
        assert columns["capped"].label == LABELS["capped"]
        assert Profile.parts == ("rebalance_weights", "daily_weights", "attribution")
        assert repr(profile) == "Profile('N0': rebalance_weights, daily_weights, attribution)"

    def test_a_stored_run_s_is_the_same(self,
                                        result,
                                        tmp_path):
        result.analytics.run_data.write(tmp_path)
        stored = Analytics(RunData.from_record(tmp_path)).instrument.profile("N7")
        finished = result.analytics.instrument.profile("N7")

        pd.testing.assert_frame_equal(stored.rebalance_weights, finished.rebalance_weights)
        pd.testing.assert_frame_equal(stored.daily_weights, finished.daily_weights)
        pd.testing.assert_frame_equal(stored.attribution, finished.attribution)

    def test_a_run_kept_before_its_rebalances_were(self,
                                                   result,
                                                   tables):
        older = dataclasses.replace(result.analytics.run_data,
                                    tables={name: table for name, table in tables.items()
                                            if name != REBALANCE_WEIGHTS})
        profile = Analytics(older).instrument.profile("N0")

        with pytest.raises(Unavailable) as raised:
            _ = profile.rebalance_weights

        assert raised.value.code == RUN_PREDATES
        assert not profile.daily_weights.empty


class TestItsContribution:

    @pytest.fixture(scope="class")
    def quarters(self,
                 result):
        return result.analytics.instrument.profile("N0").attribution

    @pytest.fixture(scope="class")
    def alone(self,
              result):
        """The analysis with each period linked on its own."""
        options = AnalysisOptions(attribution=AttributionOptions(linking="period"))

        return Analytics(result.analytics.run_data,
                         AnalysisRequest(reports={}, statistic_tables={}, options=options))

    def test_a_row_a_quarter_dated_by_its_last_day(self,
                                                  quarters):
        assert list(quarters.index) == list(pd.to_datetime(["2024-03-28", "2024-06-28",
                                                            "2024-09-30", "2024-12-31"]))
        assert list(quarters.columns) == ["contribution", "contribution_to_date",
                                          "active_contribution", "active_contribution_to_date"]
        assert quarters.attrs == {"against": "index", "by": "quarter", "linking": "window"}

    def test_the_quarters_add_up_to_the_window_s(self,
                                                result,
                                                quarters):
        whole = contribution(result.analytics)

        assert quarters["contribution"].sum() == pytest.approx(whole, rel=1e-12)
        assert quarters["contribution_to_date"].iloc[-1] == pytest.approx(whole, rel=1e-12)
        pd.testing.assert_series_equal(quarters["contribution_to_date"],
                                       quarters["contribution"].cumsum(), check_names=False)
        # The cap holds N0 under the index, so it is a real active figure.
        assert active(result.analytics) < -1e-4
        assert quarters["active_contribution"].sum() == pytest.approx(active(result.analytics),
                                                                      rel=1e-12)

    def test_each_quarter_on_its_own(self,
                                     quarters,
                                     alone):
        found = alone.instrument.profile("N0").attribution
        before = None

        for end in found.index:
            assert found.loc[end, "contribution"] == pytest.approx(
                contribution(alone, before, end), rel=1e-12)
            assert found.loc[end, "contribution_to_date"] == pytest.approx(
                contribution(alone, end=end), rel=1e-12)
            assert found.loc[end, "active_contribution"] == pytest.approx(
                active(alone, before, end), rel=1e-9)
            before = end

        assert found.attrs["linking"] == "period"
        # The last total is the window's either way; the quarters then do not
        # add up to it.
        assert found["contribution_to_date"].iloc[-1] == pytest.approx(
            quarters["contribution_to_date"].iloc[-1], rel=1e-12)
        assert abs(found["contribution"].sum() - quarters["contribution"].sum()) > 1e-4

    def test_by_month_and_over_a_window(self,
                                        result):
        months = result.analytics.instrument.profile("N0", by="month").attribution
        half = result.analytics.instrument.profile("N0", start="2024-06-28").attribution

        assert len(months) == 12 and months.attrs["by"] == "month"
        assert months["contribution"].sum() == pytest.approx(contribution(result.analytics),
                                                             rel=1e-12)
        assert list(half.index) == list(pd.to_datetime(["2024-09-30", "2024-12-31"]))
        assert half["contribution"].sum() == pytest.approx(
            contribution(result.analytics, start="2024-06-28"), rel=1e-12)

    def test_against_the_benchmark(self,
                                   result):
        found = result.analytics.instrument.profile("N0", against="benchmark").attribution

        theirs = result.analytics._context.attribution.active("benchmark").to_frame()

        assert found.attrs["against"] == "benchmark"
        assert found["active_contribution"].sum() == pytest.approx(
            theirs.set_index("asset_id").loc["N0", "contribution"], rel=1e-12)


class TestRefusals:

    def test_a_period_it_does_not_give(self,
                                       result):
        with pytest.raises(ValueError, match="by is month, quarter, year; not 'week'"):
            result.analytics.instrument.profile("N0", by="week")

    def test_a_linking_it_does_not_know(self):
        with pytest.raises(ValueError, match="Unknown linking 'daily'"):
            AttributionOptions(linking="daily")

    def test_a_name_no_portfolio_held(self,
                                      result):
        with pytest.raises(ValueError, match="No portfolio of the run held ZZZ"):
            result.analytics.instrument.profile("ZZZ")

    def test_a_comparator_the_run_lacks(self,
                                        result):
        with pytest.raises(Unavailable):
            result.analytics.instrument.profile("N0", against="optimised")
